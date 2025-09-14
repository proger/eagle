{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}
module Backend
  ( emitTorch         -- :: Prog -> Text
  , emitTorchToFile   -- :: FilePath -> Prog -> IO ()
  ) where

import qualified Data.Text as T
import qualified Data.Text.IO as T
import           Data.List (intercalate, nub)
import           IR

-- Public API
emitTorch :: Prog -> T.Text
emitTorch (Prog defs retV) =
  let params = orderedParams defs
      header = pyHeader
      sig    = pySignature params
      body   = T.unlines (concatMap (emitStmt retV) defs ++ [pyReturn retV])
  in T.unlines [header, sig, body]

emitTorchToFile :: FilePath -> Prog -> IO ()
emitTorchToFile fp p = T.writeFile fp (emitTorch p)

-- Collect parameters in first-appearance order
orderedParams :: [Stmt] -> [T.Text]
orderedParams ss =
  nub [ nm | Let _ (Param _ nm) <- ss ]

-- Python prelude with helpers for collectives & “reshape that may broadcast”
pyHeader :: T.Text
pyHeader = T.unlines
  [ "import torch"
  , "import torch.distributed as dist"
  , ""
  , "def _maybe_reshape_or_expand(x, target_rank, last_dim=None):"
  , "    \"\"\""
  , "    If target_rank == x.dim(): return x"
  , "    If target_rank == x.dim()+1: unsqueeze(-1) and expand last_dim."
  , "    Otherwise, fallback to view/reshape (may fail if sizes mismatch)."
  , "    \"\"\""
  , "    if target_rank == x.dim():"
  , "        return x"
  , "    if target_rank == x.dim() + 1:"
  , "        x2 = x.unsqueeze(-1)"
  , "        if last_dim is None:  # cannot infer; keep as unsqueezed singleton"
  , "            return x2"
  , "        shape = list(x2.shape); shape[-1] = last_dim"
  , "        return x2.expand(*shape)"
  , "    # Fallback: trust reshape with -1 placeholders"
  , "    return x.reshape(*([-1] * target_rank))"
  , ""
  , "def _all_gather_cat(x, dim):"
  , "    if not dist.is_available() or not dist.is_initialized():"
  , "        return x"
  , "    ws = dist.get_world_size()"
  , "    tensors = [torch.empty_like(x) for _ in range(ws)]"
  , "    dist.all_gather(tensors, x)"
  , "    return torch.cat(tensors, dim=dim)"
  , ""
  , "def _all_reduce_sum(x):"
  , "    if not dist.is_available() or not dist.is_initialized():"
  , "        return x"
  , "    dist.all_reduce(x, op=dist.ReduceOp.SUM)"
  , "    return x"
  , ""
  , "def _reduce_scatter_sum_cat(x, dim):"
  , "    if not dist.is_available() or not dist.is_initialized():"
  , "        return x"
  , "    # Split equally along dim and reduce_scatter. Requires equal chunks."
  , "    ws = dist.get_world_size()"
  , "    chunks = list(torch.chunk(x, ws, dim=dim))"
  , "    out = torch.empty_like(chunks[0])"
  , "    dist.reduce_scatter_tensor(out, torch.cat(chunks, dim=0), op=dist.ReduceOp.SUM)"
  , "    return out"
  , ""
  ]

pySignature :: [T.Text] -> T.Text
pySignature pnames =
  let args = T.intercalate ", " pnames
  in T.concat ["def compiled(", args, "):"]

pyReturn :: V -> T.Text
pyReturn v = T.concat ["    return ", vname v]

-- Emit one statement
emitStmt :: V -> Stmt -> [T.Text]
emitStmt _ (Let v op) =
  case op of
    Param _ nm ->
      [ T.concat ["    ", vname v, " = ", nm] ]

    Const _ _ ->
      [ T.concat ["    # ", vname v, " = Const  (initialize in Python if needed)"]
      , T.concat ["    ", vname v, " = None  # TODO: materialize Const if required"] ]

    Copy _ x ->
      [ T.concat ["    ", vname v, " = ", vname x] ]

    Reshape ty x ->
      let (rankSrc, rankDst, lastDim) = (Nothing, shapeRank (placeShapeOf x), lastSymLast ty)
      in [ T.concat [ "    ", vname v, " = _maybe_reshape_or_expand("
                    , vname x, ", ", tshowRank rankDst
                    , case lastDim of
                        Just s  -> T.concat [", last_dim=", s]
                        Nothing -> ""
                    , ")" ] ]

    Transpose _ perm x ->
      [ T.concat ["    ", vname v, " = ", vname x, ".permute(", ints perm, ")"] ]

    Map _ f x ->
      [ T.concat ["    ", vname v, " = ", emitUnary f, "(", vname x, ")"] ]

    ZipWith _ f x y ->
      [ T.concat ["    ", vname v, " = ", emitBinary f (vname x) (vname y)] ]

    Reduce _ r (Axis ax) x ->
      [ T.concat ["    ", vname v, " = ", emitReduce r, "(", vname x
                 , ", dim=", T.pack (show ax), ")"] ]

    MatVec _ a x _ _ ->
      -- In eager, just use matmul; PyTorch handles vector rhs/lhs
      [ T.concat ["    ", vname v, " = torch.matmul(", vname a, ", ", vname x, ")"] ]

    MatMul _ a b _ _ _ ->
      [ T.concat ["    ", vname v, " = torch.matmul(", vname a, ", ", vname b, ")"] ]

    MatMulEpilogue _ a b _ _ _ me ->
      let core = T.concat ["torch.matmul(", vname a, ", ", vname b, ")"]
          withEpi = case me of
            Nothing -> core
            Just (pw, mb) ->
              case (pw, mb) of
                (Add, Just vBias) -> T.concat ["(", core, " + ", vname vBias, ")"]
                (Sub, Just vBias) -> T.concat ["(", core, " - ", vname vBias, ")"]
                (Mul, Just vB)    -> T.concat ["(", core, " * ", vname vB, ")"]
                (Div, Just vB)    -> T.concat ["(", core, " / ", vname vB, ")"]
                (Tanh, _)         -> T.concat ["torch.tanh(", core, ")"]
                (Relu, _)         -> T.concat ["torch.relu(", core, ")"]
                (Neg,  _)         -> T.concat ["(-", core, ")"]
      in [ T.concat ["    ", vname v, " = ", withEpi] ]

    Collective _ coll x ->
      case coll of
        AllReduceSum _ ->
          [ T.concat ["    ", vname v, " = _all_reduce_sum(", vname x, ")"] ]
        AllGather (Axis ax) ->
          [ T.concat ["    ", vname v, " = _all_gather_cat(", vname x, ", dim=", tshow ax, ")"] ]
        ReduceScatter (Axis ax) ->
          [ T.concat ["    ", vname v, " = _reduce_scatter_sum_cat(", vname x, ", dim=", tshow ax, ")"] ]
        BroadcastTo _ ->
          -- Placement cast is a no-op in eager: treat as identity
          [ T.concat ["    ", vname v, " = ", vname x] ]

-- --- helpers to peek shapes for Reshape “expand” heuristic

placeShapeOf :: V -> Shape
placeShapeOf _ = []  -- we don't carry runtime shapes; treat as unknown

shapeRank :: Shape -> Int
shapeRank = length

lastSymLast :: Ty -> Maybe T.Text
lastSymLast (Tensor _ sh _ _) =
  case reverse sh of
    (Sym s : _) -> Just s
    _           -> Nothing

-- small printers
vname :: V -> T.Text
vname (V i) = T.pack ("v" <> show i)

ints :: [Int] -> T.Text
ints xs = T.pack (intercalate "," (map show xs))

emitUnary :: PW -> T.Text
emitUnary = \case
  Tanh -> "torch.tanh"
  Relu -> "torch.relu"
  Neg  -> "(lambda _x: -_x)"
  Add  -> err "unary Add"
  Sub  -> err "unary Sub"
  Mul  -> err "unary Mul"
  Div  -> err "unary Div"
  where err s = error ("bad unary op: " <> s)

emitBinary :: PW -> T.Text -> T.Text -> T.Text
emitBinary = \case
  Add -> \x y -> T.concat [x, " + ", y]
  Sub -> \x y -> T.concat [x, " - ", y]
  Mul -> \x y -> T.concat [x, " * ", y]
  Div -> \x y -> T.concat [x, " / ", y]
  Neg -> \_ _ -> error "bad binary op: Neg"
  Tanh-> \_ _ -> error "bad binary op: Tanh"
  Relu-> \_ _ -> error "bad binary op: ReLU"

emitReduce :: Red -> T.Text
emitReduce = \case
  RSum  -> "torch.sum"
  RMax  -> "torch.amax"
  RMean -> "torch.mean"

tshow :: Show a => a -> T.Text
tshow = T.pack . show

tshowRank :: Int -> T.Text
tshowRank = T.pack . show
