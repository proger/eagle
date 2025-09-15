{-# LANGUAGE OverloadedStrings #-}
module Autodiff
  ( gradAll  -- :: Prog -> (Prog, [(T.Text, V)])
  , gradAllWithSeed -- :: Maybe V -> Prog -> (Prog, [(T.Text, V)])
  , gradOfParam -- :: T.Text -> Prog -> Prog
  ) where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import           IR

-- Build def map
defMap :: [Stmt] -> M.Map V Op
defMap = M.fromList . map (\(Let v op) -> (v, op))

tyOfVar :: [Stmt] -> V -> Ty
tyOfVar defs v =
  case lookupLet v defs of
    Just (Let _ op) -> oTy op
    _               -> error ("Autodiff: unknown var " <> show v)

-- transpose 2D tensor type (swap last two dims)
transposeTy :: Ty -> Ty
transposeTy (Tensor dt [a,b] lo pl) = Tensor dt [b,a] lo pl
transposeTy t = t

-- For simplicity, we accumulate gradients with Maybe and ZipWith Add when needed

-- Entry: compute grads for all Params, seeding with a Param dOut for the program return
gradAll :: Prog -> (Prog, [(T.Text, V)])
gradAll = gradAllWithSeed Nothing

-- If a seed var is provided (e.g., seed=y for 0.5*||y||^2), use it.
-- Otherwise, insert a Param "dOut" with the same type as the return.
gradAllWithSeed :: Maybe V -> Prog -> (Prog, [(T.Text, V)])
gradAllWithSeed mSeed (Prog defs ret) =
  let (vSeed, defs1) = case mSeed of
                         Just v  -> (v, defs)
                         Nothing -> let tyOut = tyOfVar defs ret
                                        (s, d1) = bind (Param tyOut "dOut") defs
                                    in (s, d1)
      -- reverse pass
      (defsG, gmap) = backprop defs1 (M.singleton ret vSeed)
      -- collect grads for params by first mention order
      params = [ (nm, v) | Let v (Param _ nm) <- defs ]
      grads  = [ (nm, M.findWithDefault vZero v gmap)
               | (nm, v) <- params ]
      vZero  = V (-1)
  in (Prog defsG ret, grads)
  where
    backprop :: [Stmt] -> M.Map V V -> ([Stmt], M.Map V V)
    backprop ds g0 = foldr step (ds, g0) ds

    step :: Stmt -> ([Stmt], M.Map V V) -> ([Stmt], M.Map V V)
    step (Let v op) (accDefs, gmap) =
      case M.lookup v gmap of
        Nothing -> (accDefs, gmap)
        Just gv -> case op of
          -- pass-throughs
          Copy _ x -> add gmap x gv accDefs
          Reshape _ x ->
            let tyX = tyOfVar accDefs x
                (vR, acc1) = bind (Reshape tyX gv) accDefs
            in add gmap x vR acc1
          Transpose _ [1,0] x ->
            let tyX = tyOfVar accDefs x
                (vT, acc1) = bind (Transpose tyX [1,0] gv) accDefs
            in add gmap x vT acc1
          Map _ Tanh x ->
            let tyX = tyOfVar accDefs x
                (vy, _) = case lookupLet v accDefs of Just (Let _ (Map _ _ _)) -> (v, accDefs); _ -> (v, accDefs)
                (vG, acc1) = bind (GradTanh tyX vy gv) accDefs
            in add gmap x vG acc1
          Collective _ _ x ->
            -- For now, treat collectives as pass-through in reverse
            add gmap x gv accDefs
          ZipWith _ Add x y ->
            let (acc1, g1) = add gmap x gv accDefs
                (acc2, g2) = add g1   y gv acc1
            in (acc2, g2)
          MatMul _ a b ta tb _ ->
            let tyA = tyOfVar accDefs a
                tyB = tyOfVar accDefs b
                -- dA'
                tyA' = if ta then transposeTy tyA else tyA
                (vDAp, acc1) = bind (MatMul tyA' gv b False (not tb) MMA16x16x16) accDefs
                (vDA, acc2)  = if ta
                               then let (vT, accT) = bind (Transpose tyA [1,0] vDAp) acc1 in (vT, accT)
                               else (vDAp, acc1)
                -- dB'
                tyB' = if tb then transposeTy tyB else tyB
                (vDBp, acc3) = bind (MatMul tyB' a gv (not ta) False MMA16x16x16) acc2
                (vDB, acc4)  = if tb
                               then let (vT, accT) = bind (Transpose tyB [1,0] vDBp) acc3 in (vT, accT)
                               else (vDBp, acc3)
                (acc5, g1) = add gmap a vDA acc4
                (acc6, g2) = add g1   b vDB acc5
            in (acc6, g2)
          _ -> (accDefs, gmap)
      where
        add gmap' var delta acc =
          case M.lookup var gmap' of
            Nothing -> (acc, M.insert var delta gmap')
            Just old ->
              let tyX = tyOfVar acc var
                  (vAdd, acc1) = bind (ZipWith tyX Add old delta) acc
              in (acc1, M.insert var vAdd gmap')

-- Convenience: produce a program that returns the gradient of a given parameter name.
gradOfParam :: T.Text -> Prog -> Prog
gradOfParam pname p =
  let (Prog defs ret, pairs) = gradAll p
  in case lookup pname pairs of
       Just vG -> Prog defs vG
       Nothing -> error ("gradOfParam: unknown parameter " <> T.unpack pname)
