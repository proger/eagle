{-# LANGUAGE OverloadedStrings #-}
module Planner
  ( -- device mesh / target
    Mesh(..), Target(..)
    -- selecting IR values in schedules
  , Sel(..)
    -- schedule commands
  , SchedCmd(..)
    -- handy placements
  , replicated, splitRow, splitCol, partial
    -- main entry
  , specializePlacement
  ) where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import           IR

--------------------------------------------------------------------------------
-- device mesh / target

-- Minimal placeholder: we only need rank to reason about SBP vectors.
data Mesh   = Mesh { meshRank :: Int } deriving (Eq, Show)
data Target = Target { mesh :: Mesh }  deriving (Eq, Show)

--------------------------------------------------------------------------------
-- schedule DSL

data Sel = ByName T.Text | ByVar V
  deriving (Eq, Show)

data SchedCmd
  = Place Sel Placement   -- set concrete placement for a value
  | Want  Sel Placement   -- soft preference for consumer view (used by align)
  deriving (Eq, Show)

-- convenience placements (assume 2D mesh by convention: row=0, col=1)
replicated :: Placement
replicated = []

splitRow, splitCol :: Placement
splitRow = [Split (Axis 0)]
splitCol = [Split (Axis 1)]

-- denote partial sums (e.g., from split-K matmul). Resolved later by align/consumers.
partial :: Placement
partial = [PartialSum]

--------------------------------------------------------------------------------
-- planner state

data PlanEnv = PlanEnv
  { tgt      :: Target
  , names    :: M.Map T.Text V   -- param name -> var
  , placeOfV :: M.Map V Placement
  , wantOfV  :: M.Map V Placement
  }

emptyEnv :: Target -> [Stmt] -> PlanEnv
emptyEnv t defs = PlanEnv t nameMap M.empty M.empty
  where
    nameMap = M.fromList [ (nm, v) | Let v (Param _ nm) <- defs ]

selVar :: PlanEnv -> Sel -> V
selVar env = \case
  ByVar v   -> v
  ByName nm ->
    case M.lookup nm (names env) of
      Just v  -> v
      Nothing -> error ("Planner: unknown ByName " <> T.unpack nm)

setPlace :: PlanEnv -> V -> Placement -> PlanEnv
setPlace env v p = env { placeOfV = M.insert v p (placeOfV env) }

setWant :: PlanEnv -> V -> Placement -> PlanEnv
setWant env v p  = env { wantOfV  = M.insert v p (wantOfV  env) }

lookupPlace :: PlanEnv -> V -> Placement
lookupPlace env v = M.findWithDefault replicated v (placeOfV env)

lookupWant :: PlanEnv -> V -> Maybe Placement
lookupWant env v = M.lookup v (wantOfV env)

-- seed placements/wants from schedule
applySched :: [SchedCmd] -> PlanEnv -> PlanEnv
applySched cmds env0 = foldl step env0 cmds
  where
    step env (Place s p) = setPlace env (selVar env s) p
    step env (Want  s p) = setWant  env (selVar env s) p

--------------------------------------------------------------------------------
-- entry point

specializePlacement :: Target -> [SchedCmd] -> Prog -> Prog
specializePlacement tgt cmds (Prog defs ret) =
  let env0            = applySched cmds (emptyEnv tgt defs)
      (defs', env1)   = planBlock env0 [] defs
      defsFinal       = finalizeTypes env1 defs'
  in Prog defsFinal ret

--------------------------------------------------------------------------------
-- block traversal

planBlock :: PlanEnv -> [Stmt] -> [Stmt] -> ([Stmt], PlanEnv)
planBlock env acc []             = (acc, env)
planBlock env acc (Let v op:xs)  =
  let (acc', env', op') = planOp env acc v op
  in planBlock env' (acc' ++ [Let v op']) xs

-- set the final PConcrete placement on every op's type
finalizeTypes :: PlanEnv -> [Stmt] -> [Stmt]
finalizeTypes env = map (\(Let v op) -> Let v (op { oTy = (oTy op){ place = PConcrete (lookupPlace env v) } }))

--------------------------------------------------------------------------------
-- op-by-op planning & communication insertion

planOp :: PlanEnv -> [Stmt] -> V -> Op -> ([Stmt], PlanEnv, Op)
planOp env acc v = \case
  -- params: honor schedule placement or keep replicated by default
  op@(Param ty nm) ->
    let p0   = case place ty of
                 PConcrete p -> p
                 _           -> M.findWithDefault replicated v (placeOfV env)
        env' = setPlace env v p0
    in (acc, env', op { oTy = ty { place = PConcrete p0 } })

  -- layout-neutral ops: inherit input placement
  op@(Copy ty x) ->
    inherit1 env acc v op ty x

  op@(Reshape ty x) ->
    inherit1 env acc v op ty x

  op@(Transpose ty _ x) ->
    inherit1 env acc v op ty x

  -- elementwise unary: inherit
  op@(Map ty _ x) ->
    inherit1 env acc v op ty x

  -- elementwise binary: align inputs
  op@(ZipWith ty _ x y) ->
    let px = lookupPlace env x
        py = lookupPlace env y
        want = lookupWant env v <|> lookupWant env x <|> lookupWant env y
        (acc1, x', y', pOut) = align acc (x,px) (y,py) want
        env' = setPlace env v pOut
    in (acc1, env', op { oTy = ty { place = PConcrete pOut }, x = x', y = y' })

  -- reductions: if reducing across mesh-split axis, insert allreduce and replicate
  op@(Reduce ty _ (Axis redAx) x) ->
    let px             = lookupPlace env x
        (acc1, x', p') = reduceRule acc (Axis redAx) (x,px)
        env'           = setPlace env v p'
    in (acc1, env', op { oTy = ty { place = PConcrete p' }, x = x' })

  -- matvec: treat as matmul for placement purposes (row/col rules apply)
  op@(MatVec ty a x _ _) ->
    let pa = lookupPlace env a
        px = lookupPlace env x
        (acc1, a', x', pOut) = matmulRule acc Nothing (a,pa) (x,px)
        env' = setPlace env v pOut
    in (acc1, env', op { oTy = ty { place = PConcrete pOut }, a = a', v = x' })

  -- matmul core cases
  op@(MatMul ty a b _ _ _) ->
    let pa = lookupPlace env a
        pb = lookupPlace env b
        want = lookupWant env v
        (acc1, a', b', pOut) = matmulRule acc want (a,pa) (b,pb)
        env' = setPlace env v pOut
    in (acc1, env', op { oTy = ty { place = PConcrete pOut }, a = a', b = b' })

  -- matmul with epilogue: same as matmul
  op@(MatMulEpilogue ty a b _ _ _ _) ->
    let pa = lookupPlace env a
        pb = lookupPlace env b
        want = lookupWant env v
        (acc1, a', b', pOut) = matmulRule acc want (a,pa) (b,pb)
        env' = setPlace env v pOut
    in (acc1, env', op { oTy = ty { place = PConcrete pOut }, a = a', b = b' })

  -- collectives: trust placement already encoded by the node
  op@(Collective ty _ _) ->
    let p  = case place ty of PConcrete q -> q; _ -> replicated
        env' = setPlace env v p
    in (acc, env', op)

  where
    inherit1 e a vv theOp ty x =
      let px  = lookupPlace e x
          e'  = setPlace e vv px
      in (a, e', theOp { oTy = ty { place = PConcrete px } })

--------------------------------------------------------------------------------
-- alignment and transformation rules

import           Data.Maybe (fromMaybe)
import           Control.Applicative ((<|>))

-- align two inputs to a common placement, optionally guided by a desired output placement
align :: [Stmt]
      -> (V, Placement) -> (V, Placement)
      -> Maybe Placement
      -> ([Stmt], V, V, Placement)
align acc (vx,px) (vy,py) wantOut
  | px == py =
      let pout = fromMaybe px wantOut
      in (acc, vx, vy, pout)
  | otherwise =
      case (px, py, wantOut) of
        -- replicate broadcast to match split
        ([], s, _) ->
          let (vx', acc1) = insertBroadcast acc vx s
          in (acc1, vx', vy, s)
        (s, [], _) ->
          let (vy', acc1) = insertBroadcast acc vy s
          in (acc1, vx, vy', s)

        -- both Split on same mesh axis: keep that split
        ([Split ax1], [Split ax2], _) | ax1 == ax2 ->
          (acc, vx, vy, px)

        -- prefer requested placement when possible
        (_, _, Just want) ->
          case want of
            [] -> let (vx', acc1) = insertAllGather acc vx (guessAxis px)
                      (vy', acc2) = insertAllGather acc1 vy (guessAxis py)
                  in (acc2, vx', vy', [])
            [Split ax] ->
                  let (vx', acc1) = ensureSplit acc vx px ax
                      (vy', acc2) = ensureSplit acc1 vy py ax
                  in (acc2, vx', vy', [Split ax])
            _ -> (acc, vx, vy, px)  -- fallback

        -- default: gather right to left placement
        _ ->
          let (vy', acc1) = case px of
                              []            -> insertAllGather acc vy (guessAxis py)
                              [Split axDst] -> ensureSplit acc vy py axDst
                              _             -> (vy, acc)
          in (acc1, vx, vy', px)

-- reduction: if reducing along the same axis as a split, allreduce to replicate
reduceRule :: [Stmt]
           -> Axis
           -> (V, Placement)
           -> ([Stmt], V, Placement)
reduceRule acc redAxis (vx, px) =
  case px of
    [Split ax] | ax == redAxis ->
      let (v', acc1) = insertAllReduce acc vx redAxis
      in (acc1, v', replicated)
    [PartialSum] ->
      let (v', acc1) = insertAllReduce acc vx redAxis
      in (acc1, v', replicated)
    _ -> (acc, vx, px)

-- matmul placements:
--  A(rep), B(split col axN) -> C(split col axN)             (1D col parallel)
--  A(split K axK), B(rep)   -> C(PartialSum)                (needs later reduction)
--  A(split row axR), B(split col axC) -> C(split row+col)   (2D style)
--  guided by wantOut: if wantOut is replicated, we may insert collectives to realize it
matmulRule :: [Stmt]
           -> Maybe Placement
           -> (V, Placement)
           -> (V, Placement)
           -> ([Stmt], V, V, Placement)
matmulRule acc wantOut (va,pa) (vb,pb) =
  case (pa, pb) of
    ([], [Split axN]) ->
      -- classic column-parallel: follow B
      let pout = chooseOut wantOut [Split axN]
      in (acc, va, vb, pout)

    ([Split _axK], []) ->
      -- split-K partials: mark partial; consumer resolves to allreduce or reduce-scatter
      let pout = chooseOut wantOut partial
      in (acc, va, vb, pout)

    ([Split axR], [Split axC]) | axR /= axC ->
      let pout = chooseOut wantOut [Split axR, Split axC]
      in (acc, va, vb, pout)

    -- if want replicated, gather both sides
    (pa', pb') ->
      case wantOut of
        Just [] ->
          let (va', acc1) = insertAllGather acc va (guessAxis pa')
              (vb', acc2) = insertAllGather acc1 vb (guessAxis pb')
          in (acc2, va', vb', [])
        _ -> (acc, va, vb, pb')  -- default: follow B

chooseOut :: Maybe Placement -> Placement -> Placement
chooseOut (Just want) _ = want
chooseOut Nothing      p = p

--------------------------------------------------------------------------------
-- collective insertion helpers

-- NOTE: these helper constructors *prepend* new Let bindings into the
-- accumulator and return the new Var produced so downstream users can refer to it.

insertBroadcast :: [Stmt] -> V -> Placement -> (V, [Stmt])
insertBroadcast acc vx p =
  let ty  = tyOfVar acc vx
      ty' = ty { place = PConcrete p }
      (v', acc1) = bind (Collective ty' (BroadcastTo p) vx) acc
  in (v', acc1)

insertAllReduce :: [Stmt] -> V -> Axis -> (V, [Stmt])
insertAllReduce acc vx ax =
  let ty  = tyOfVar acc vx
      ty' = ty { place = PConcrete replicated }
      (v', acc1) = bind (Collective ty' (AllReduceSum ax) vx) acc
  in (v', acc1)

insertAllGather :: [Stmt] -> V -> Axis -> (V, [Stmt])
insertAllGather acc vx ax =
  let ty  = tyOfVar acc vx
      -- simple: after gather, treat as replicated (safe default)
      ty' = ty { place = PConcrete replicated }
      (v', acc1) = bind (Collective ty' (AllGather ax) vx) acc
  in (v', acc1)

ensureSplit :: [Stmt] -> V -> Placement -> Axis -> (V, [Stmt])
ensureSplit acc vx p ax =
  case p of
    [Split ax'] | ax' == ax -> (vx, acc)
    []                      -> -- from replicated to split: model as gather+split cast
                                let (v', acc1) = insertBroadcast acc vx [Split ax]
                                in (v', acc1)
    _                       -> -- mismatch: gather to replicated, then split
                                let (v1, acc1) = insertAllGather acc vx (guessAxis p)
                                    (v2, acc2) = insertBroadcast acc1 v1 [Split ax]
                                in (v2, acc2)

-- Heuristic: pick the first axis mentioned; otherwise default axis 0
guessAxis :: Placement -> Axis
guessAxis []               = Axis 0
guessAxis (Split ax : _ )  = ax
guessAxis (PartialSum: _ ) = Axis 0
guessAxis (_:xs)           = guessAxis xs

-- read the type of an existing var from the current acc block (must exist)
tyOfVar :: [Stmt] -> V -> Ty
tyOfVar defs v =
  case lookupLet v defs of
    Just (Let _ op) -> oTy op
    Nothing         -> error ("Planner: unknown var " <> show v)
