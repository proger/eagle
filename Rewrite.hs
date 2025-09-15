{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TupleSections #-}
module Rewrite
  ( -- pipeline
    runPipeline
  , passFixpoint
  , passCSE
  , passDCE
    -- individual local rewrites
  , rewriteMapAfterMatMul
  , rewriteMatVecToMatMul
  , rewritePullTranspose
  ) where

import qualified Data.Map.Strict as M
import qualified Data.Set        as S
import           Data.Maybe      (fromMaybe)
import           IR

--------------------------------------------------------------------------------
-- Small utilities over an ANF block

-- build use-def indices (only what we need here)
type Defs = M.Map V Op

defsOf :: [Stmt] -> Defs
defsOf ss = M.fromList [ (v, op) | Let v op <- ss ]

lookupDef :: Defs -> V -> Maybe Op
lookupDef m v = M.lookup v m

--------------------------------------------------------------------------------
-- 0) Worklist-style driver for local rewrites that work over a whole block

type LocalRewrite = [Stmt] -> ([Stmt], Bool)

passFixpoint :: [LocalRewrite] -> [Stmt] -> [Stmt]
passFixpoint rules = go
  where
    go ss = case stepAny rules ss of
      (ss', True)  -> go ss'
      (ss', False) -> ss'
    stepAny []     ss = (ss, False)
    stepAny (r:rs) ss = case r ss of
                          (ss', True)  -> (ss', True)
                          (ss', False) -> stepAny rs ss'

--------------------------------------------------------------------------------
-- 1) Local rewrites

-- Fuse: Map f (MatMul a b)  ==>  MatMulEpilogue a b (epi=f[, maybe bias])
rewriteMapAfterMatMul :: LocalRewrite
rewriteMapAfterMatMul ss = go [] ss
  where
    go acc [] = (reverse acc, False)
    go acc (Let vz (Map t f vy) : rest) =
      case findMatMul acc vy of
        Just (vy', (tmm, a, b, ta, tb, mma)) | vy' == vy ->
          let acc' = dropByVar vy acc
              epi  = MatMulEpilogue t a b ta tb mma (Just (f, Nothing))
          in go (Let vz epi : acc') rest
        _ -> go (Let vz (Map t f vy) : acc) rest
    go acc (s:rest) = go (s:acc) rest

    findMatMul :: [Stmt] -> V -> Maybe (V, (Ty,V,V,Bool,Bool,MMA))
    findMatMul acc v =
      case lookupLet v acc of
        Just (Let v' (MatMul t a b ta tb m)) -> Just (v', (t,a,b,ta,tb,m))
        _                                    -> Nothing

    dropByVar v = filter (\(Let v' _) -> v' /= v)

-- Pull Transpose through MatMul where possible:
--   MatMul (Transpose A) B  -> MatMul A B  with flipped transA
--   MatMul A (Transpose B)  -> MatMul A B  with flipped transB
rewritePullTranspose :: LocalRewrite
rewritePullTranspose ss =
  let defs = defsOf ss
  in go defs [] False ss
  where
    go _ acc changed [] = (reverse acc, changed)
    go m acc changed (s@(Let v (MatMul t a b ta tb mma)) : rest) =
      case (lookupDef m a, lookupDef m b) of
        (Just (Transpose _ [1,0] a0), _) ->
          let s' = Let v (MatMul t a0 b (not ta) tb mma)
          in go m (s':acc) True rest
        (_, Just (Transpose _ [1,0] b0)) ->
          let s' = Let v (MatMul t a b0 ta (not tb) mma)
          in go m (s':acc) True rest
        _ -> go m (s:acc) changed rest
    go m acc changed (s:rest) = go m (s:acc) changed rest

-- Lift MatVec with batched RHS into a MatMul. This keeps Surface simple.
rewriteMatVecToMatMul :: LocalRewrite
rewriteMatVecToMatMul ss =
  let defs = defsOf ss
  in go defs [] False ss
  where
    rhsRank m v =
      case lookupDef m v of
        Just op -> length (shape (oTy op))
        Nothing -> 1 -- if unknown, assume vector to be safe
    go _ acc changed [] = (reverse acc, changed)
    go m acc changed (Let v (MatVec t a x ta mma) : rest)
      | rhsRank m x > 1 =
          let s' = Let v (MatMul t a x ta False mma)
          in go m (s':acc) True rest
      | otherwise = go m (Let v (MatVec t a x ta mma) : acc) changed rest
    go m acc changed (s:rest) = go m (s:acc) changed rest

--------------------------------------------------------------------------------
-- 2) CSE via simple value numbering / hash-consing

-- Keys ignore destination var names; include types (safe & simple).
data OpKey
  = KConst Ty [Double]
  | KParam Ty String
  | KCopy Ty V
  | KReshape Ty V
  | KTranspose Ty [Int] V
  | KMap Ty PW V
  | KZip Ty PW V V
  | KReduce Ty Red Axis V
  | KMatVec Ty V V Bool MMA
  | KMatMul Ty V V Bool Bool MMA
  | KMatMulEpi Ty V V Bool Bool MMA (Maybe (PW, Maybe V))
  | KColl Ty Collective V
  | KGradTanh Ty V V
  deriving (Eq, Ord, Show)

keyOf :: Op -> OpKey
keyOf = \case
  Const t bs                    -> KConst t bs
  Param t nm                    -> KParam t (show nm)
  Copy t a                      -> KCopy t a
  Reshape t a                   -> KReshape t a
  Transpose t p a               -> KTranspose t p a
  Map t f a                     -> KMap t f a
  ZipWith t f a b               -> KZip t f a b
  Reduce t r ax a               -> KReduce t r ax a
  MatVec t a b ta m             -> KMatVec t a b ta m
  MatMul t a b ta tb m          -> KMatMul t a b ta tb m
  MatMulEpilogue t a b ta tb m e-> KMatMulEpi t a b ta tb m e
  Collective t k a              -> KColl t k a
  GradTanh t y gy               -> KGradTanh t y gy

-- rename operands in an op (using IR helper)
applySub :: M.Map V V -> Op -> Op
applySub = renameVarInOp

passCSE :: [Stmt] -> [Stmt]
passCSE = go M.empty M.empty []
  where
    -- kv: OpKey -> canonical var ; sub: var -> canonical var ; acc: emitted stmts
    go _ _ acc [] = reverse acc
    go kv sub acc (Let v op0 : rest) =
      let op1 = applySub sub op0
          k   = keyOf op1
      in case M.lookup k kv of
           Just vCan ->
             -- drop this Let, and redirect future uses of v -> vCan
             let sub' = M.insert v vCan sub
             in go kv sub' acc rest
           Nothing   ->
             let kv'  = M.insert k v kv
                 acc' = Let v op1 : acc
             in go kv' sub acc' rest

--------------------------------------------------------------------------------
-- 3) DCE (backward liveness from program return)

passDCE :: Prog -> Prog
passDCE (Prog defs ret) =
  let defMap = defsOf defs
      live   = lfix (S.singleton ret) step
      step s = S.union s (S.fromList (concatMap inputsOf (S.toList s)))
      inputsOf v =
        case lookupDef defMap v of
          Nothing  -> []
          Just op  -> case op of
            Const{}                  -> []
            Param{}                  -> []
            Copy _ a                 -> [a]
            Reshape _ a              -> [a]
            Transpose _ _ a          -> [a]
            Map _ _ a                -> [a]
            ZipWith _ _ a b          -> [a,b]
            Reduce _ _ _ a           -> [a]
            MatVec _ a b _ _         -> [a,b]
            MatMul _ a b _ _ _       -> [a,b]
            MatMulEpilogue _ a b _ _ _ me ->
              [a,b] ++ maybe [] (\(_,mb)-> maybe [] (:[]) mb) me
            Collective _ _ a         -> [a]
      keep (Let v _) = S.member v live
      defs' = filter keep defs
  in Prog defs' ret

lfix :: (Eq a) => a -> (a -> a) -> a
lfix x f = let x' = f x in if x' == x then x else lfix x' f

--------------------------------------------------------------------------------
-- 4) Full pipeline

runPipeline :: Prog -> Prog
runPipeline (Prog defs ret) =
  let defs1 = passFixpoint
                [ rewriteMatVecToMatMul
                , rewritePullTranspose
                , rewriteMapAfterMatMul
                ] defs
      defs2 = passCSE defs1
  in passDCE (Prog defs2 ret)
