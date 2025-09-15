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
  , "from eagle.runtime import _maybe_reshape_or_expand, _all_gather_cat, _all_reduce_sum, _reduce_scatter_sum_cat, _shard_take"
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
    Param ty nm ->
      let base = [ T.concat ["    ", vname v, " = ", nm] ]
      in case place (oTy op) of
           PConcrete [Split (Axis ax)] ->
             base ++ [ T.concat ["    ", vname v, " = _shard_take(", vname v, ", dim=", tshow ax, ")"] ]
           _ -> base

    Const _ _ ->
      [ T.concat ["    # ", vname v, " = Const  (initialize in Python if needed)"]
      , T.concat ["    ", vname v, " = None  # TODO: materialize Const if required"] ]

    Copy _ x ->
      [ T.concat ["    ", vname v, " = ", vname x] ]

    Reshape ty x ->
      let rankDst = shapeRank (placeShapeOf x)
          lastDim = lastSymLast ty
          reshapeLn = T.concat [ "    ", vname v, " = _maybe_reshape_or_expand("
                               , vname x, ", ", tshowRank rankDst
                               , case lastDim of
                                   Just s  -> T.concat [", last_dim=", s]
                                   Nothing -> ""
                               , ")" ]
          shShard = case place ty of
                      PConcrete [Split (Axis ax)] ->
                        [ T.concat ["    ", vname v, " = _shard_take(", vname v, ", dim=", tshow ax, ")"] ]
                      _ -> []
      in reshapeLn : shShard

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

    Collective ty coll x ->
      case coll of
        AllReduceSum _ ->
          [ T.concat ["    ", vname v, " = _all_reduce_sum(", vname x, ")"] ]
        AllGather (Axis ax) ->
          [ T.concat ["    ", vname v, " = _all_gather_cat(", vname x, ", dim=", tshow ax, ")"] ]
        ReduceScatter (Axis ax) ->
          [ T.concat ["    ", vname v, " = _reduce_scatter_sum_cat(", vname x, ", dim=", tshow ax, ")"] ]
        BroadcastTo p ->
          case p of
            [Split (Axis ax)] ->
              [ T.concat ["    ", vname v, " = _shard_take(", vname x, ", dim=", tshow ax, ")"] ]
            _ -> [ T.concat ["    ", vname v, " = ", vname x] ]

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
