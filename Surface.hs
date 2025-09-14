{-# LANGUAGE OverloadedStrings #-}
module Surface
  ( -- builder
    Build, runBuild, ret
  , Arr(..), param
  , addBatchView, dropBatchView
  , mapA, zipAdd, tanhA
  , matmul, matvec     -- matvec is a SMART constructor (lifts to matmul)
  , -- vmap specialized for rnnStep shape convention
    vmapRnnStep
  ) where

import           Control.Monad.State.Strict
import qualified Data.Text as T
import           IR

--------------------------------------------------------------------------------
-- Builder monad over ANF blocks

type Build a = State [Stmt] a

runBuild :: Build V -> Prog
runBuild m =
  let defs = execState (m >> pure ()) []
      outV = case defs of [] -> V 0; _ -> let Let v _ = last defs in v
  in Prog defs outV

ret :: Arr -> Build V
ret (Arr v) = pure v

newtype Arr = Arr { unA :: V } deriving (Eq, Ord, Show)

-- fetch current type of a var from defs
arrTy :: Arr -> Build Ty
arrTy (Arr v) = do
  defs <- get
  case lookupLet v defs of
    Just (Let _ op) -> pure (oTy op)
    _               -> error "arrTy: unknown var"

arrShape :: Arr -> Build Shape
arrShape a = shape <$> arrTy a

withBind :: Op -> Build Arr
withBind op = do
  defs <- get
  let (v, defs') = bind op defs
  put defs'
  pure (Arr v)

-- small helper for shape pairs
take2 :: [a] -> (a, a)
take2 [a,b] = (a,b)
take2 _     = error "matmul/matvec: rank mismatch"

--------------------------------------------------------------------------------
-- Parameters

param :: Ty -> T.Text -> Build Arr
param ty nm = withBind (Param ty nm)

--------------------------------------------------------------------------------
-- Views that change logical shape only (IR Reshape nodes)

addBatchView :: Dim -> Arr -> Build Arr
addBatchView b a = do
  ty@(Tensor dt sh lo _) <- arrTy a
  let ty' = Tensor dt (sh ++ [b]) lo (place ty)  -- keep placement annotation
  withBind (Reshape ty' (unA a))

dropBatchView :: Arr -> Build Arr
dropBatchView a = do
  ty@(Tensor dt sh lo _) <- arrTy a
  case sh of
    []   -> error "dropBatchView: rank-0"
    _: _ ->
      let ty' = Tensor dt (init sh) lo (place ty)
      in withBind (Reshape ty' (unA a))

--------------------------------------------------------------------------------
-- Elementwise / simple ops

mapA :: PW -> Arr -> Build Arr
mapA f a = do
  ty <- arrTy a
  withBind (Map ty f (unA a))

zipAdd :: Arr -> Arr -> Build Arr
zipAdd x y = do
  tx <- arrTy x
  -- output type follows broadcasted shape; we keep it simple and reuse tx
  withBind (ZipWith tx Add (unA x) (unA y))

tanhA :: Arr -> Build Arr
tanhA = mapA Tanh

--------------------------------------------------------------------------------
-- Linear algebra

-- Plain matmul (caller supplies correct shapes)
matmul :: Arr -> Arr -> Build Arr
matmul a b = do
  Tensor dt shA _ plA <- arrTy a
  Tensor _  shB _ plB <- arrTy b
  -- minimal shape calc: (m×k)·(k×n) -> (m×n)
  let (m,k1) = take2 shA; (k2,n) = take2 shB
      _check = if show k1 == show k2 then () else error "matmul: K mismatch"
      tOut   = Tensor dt [m,n] RowMajor (mergePlace plA plB)
  withBind (MatMul tOut (unA a) (unA b) False False MMA16x16x16)

-- SMART matvec: if RHS has batch, lift to matmul; else emit MatVec
matvec :: Arr -> Arr -> Build Arr
matvec a x = do
  ta <- arrTy a
  tx <- arrTy x
  case (ta, tx) of
    (Tensor dtA [h,k] _ plA, Tensor _ (k':batch) _ plX) -> do
      let _   = if k == k' then () else error "matvec: K mismatch"
          tyY = Tensor dtA (h:batch) RowMajor (mergePlace plA plX)
      withBind (MatVec tyY (unA a) (unA x) False MMA16x16x16)
    _ -> error "matvec: ill-shaped operands"
  where
    take2 [a,b] = (a,b)
    take2 _     = error "matmul/matvec: rank mismatch"

-- simple placement merge policy (kept the same concrete/unknown)
mergePlace :: PlaceAnn -> PlaceAnn -> PlaceAnn
mergePlace (PConcrete p) (PConcrete _) = PConcrete p
mergePlace pa _ = pa

--------------------------------------------------------------------------------
-- vmap specialized for rnnStep (5-arg step): adds batch on x, hprev, bias; leaves weights as-is

-- rnnStep has type:
--   Wx[h×x] -> Wh[h×h] -> b[h] or b[h×1]/[h×b] -> x[k] -> hprev[h] -> h[h] (or with batch axes)
--
-- vmapRnnStep b step lifts it to operate on batches:
--   x  becomes [x×b]
--   h  becomes [h×b]
--   b  becomes [h×b] (view)
vmapRnnStep
  :: Dim
  -> (Arr -> Arr -> Arr -> Arr -> Arr -> Build Arr)  -- step Wx Wh b x hprev
  ->  Arr -> Arr -> Arr -> Arr -> Arr -> Build Arr   -- returns H (batched)
vmapRnnStep b step wx wh bBias x hprev = do
  -- add a trailing batch axis to x, hprev, bias (weights stay 2D)
  xB     <- addBatchView b x
  hprevB <- addBatchView b hprev
  bB     <- ensureBiasBatched b bBias
  step wx wh bB xB hprevB
  where
    ensureBiasBatched :: Dim -> Arr -> Build Arr
    ensureBiasBatched b a = do
      Tensor _ sh _ _ <- arrTy a
      case sh of
        [ _h ]       -> addBatchView b a           -- [h] -> [h×b]
        [ _h, _one ] -> addBatchView b a           -- [h×1] -> [h×1×b] (simple view)
        [ _h, _b  ]  -> pure a                     -- already [h×b]
        _            -> addBatchView b a           -- fallback: add last axis
