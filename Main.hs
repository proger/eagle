{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import           Control.Monad.State.Strict (get)
import           IR
import           Planner
import           Surface
import           Rewrite
import           Backend (emitTorch, emitTorchToFile)
import           Autodiff (gradAllWithSeed)

-- Build a single batched RNN step in the surface DSL
buildBatchedRnnStep :: Prog
buildBatchedRnnStep =
  runBuild $ do
    let dt = BF16
        h  = sym "h"; x = sym "x"; b = sym "b"
        tWx = Tensor dt [h,x] RowMajor (PUnknown "p_Wx")
        tWh = Tensor dt [h,h] RowMajor (PUnknown "p_Wh")
        tB  = Tensor dt [h]   RowMajor (PUnknown "p_b")
        tX  = Tensor dt [x]   RowMajor (PUnknown "p_x")
        tH  = Tensor dt [h]   RowMajor (PUnknown "p_h")

    wx <- param tWx "Wx"
    wh <- param tWh "Wh"
    bb <- param tB  "b"
    xv <- param tX  "x"
    hp <- param tH  "hprev"

    -- vmap-style: add trailing batch dim 'b' to x, hprev, b (weights stay 2D)
    hNext <- vmapRnnStep b rnnStep wx wh bb xv hp
    ret hNext
  where
    -- the unbatched step in the surface DSL
    rnnStep wx wh b x hprev = do
      u  <- matvec wx x
      v  <- matvec wh hprev
      s  <- zipAdd u v
      sb <- zipAdd s b
      tanhA sb

printProg :: String -> Prog -> IO ()
printProg title (Prog defs _) = do
  putStrLn ("\n" <> title)
  mapM_ (putStrLn . ppStmt) defs

main :: IO ()
main = do
  let progAlg = buildBatchedRnnStep

  -- Two schedules: replicated vs. split-by-columns (mesh rank=2)
  let tgt       = Target (Mesh 2)
      schedRep  = []  -- everything replicated by default
      schedCol  = [ Place (ByName "Wx") [Split (Axis 1)]
                  , Place (ByName "Wh") [Split (Axis 1)]
                  , Place (ByName "b")  [Split (Axis 1)]
                  , Place (ByName "hprev") [Split (Axis 1)]
                  ]

  -- Specialize placements
  let progRep  = specializePlacement tgt schedRep  progAlg
      progCol  = specializePlacement tgt schedCol  progAlg

  -- Optimize (fusion + CSE + DCE)
  let progRepOpt = runPipeline progRep
      progColOpt = runPipeline progCol

  -- Show before/after
  printProg "=== Base ===" progAlg
  printProg "=== Replicated (before optimize) ===" progRep
  printProg "=== Replicated (after  optimize) ===" progRepOpt
  printProg "=== Split-by-columns (before optimize) ===" progCol
  printProg "=== Split-by-columns (after  optimize) ===" progColOpt

  -- Emit PyTorch eager code using Backend.hs (before and after optimize)
  putStrLn "\n=== PyTorch (Replicated, before optimize) ==="
  TIO.putStrLn (emitTorch progRep)
  emitTorchToFile "compiled_rnn_step.replicated.before.py" progRep

  putStrLn "\n=== PyTorch (Split-by-columns, before optimize) ==="
  TIO.putStrLn (emitTorch progCol)
  emitTorchToFile "compiled_rnn_step.before.py" progCol

  putStrLn "\n=== PyTorch (Split-by-columns, after optimize) ==="
  TIO.putStrLn (emitTorch progColOpt)
  emitTorchToFile "compiled_rnn_step.after.py" progColOpt
  putStrLn "Wrote compiled_rnn_step.replicated.before.py, compiled_rnn_step.before.py and compiled_rnn_step.after.py"

  -- Gradients of all parameters for objective 0.5*||h||^2 (seed = y)
  -- We compute grads on the split-by-columns optimized program.
  let Prog defsY y = progColOpt
  let (progWithGrads, grads) = gradAllWithSeed (Just y) progColOpt
  putStrLn "\n=== Gradients (Split-by-columns, after optimize) ==="
  mapM_ (\(nm, v) -> do
            let p = case progWithGrads of Prog ds _ -> Prog ds v
            putStrLn ("-- grad of " <> T.unpack nm)
            TIO.putStrLn (emitTorch p)
            emitTorchToFile ("compiled_grad_" <> T.unpack nm <> ".py") p
        ) grads
  putStrLn "Wrote compiled_grad_<param>.py files for all params"

-- vmap specialized for rnnStep (5-arg step): adds batch on x, hprev, bias; leaves weights as-is
-- rnnStep has type:
--   Wx[h×x] -> Wh[h×h] -> b[h] or b[h×1]/[h×b] -> x[k] -> hprev[h] -> h[h] (or with batch axes)
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
    -- local type query (mirror of Surface.arrTy)
    arrTyM :: Arr -> Build Ty
    arrTyM (Arr v) = do
      defs <- get
      case lookupLet v defs of
        Just (Let _ op) -> pure (oTy op)
        _               -> error "arrTy: unknown var"

    ensureBiasBatched :: Dim -> Arr -> Build Arr
    ensureBiasBatched b a = do
      Tensor _ sh _ _ <- arrTyM a
      case sh of
        [ _h ]       -> addBatchView b a           -- [h] -> [h×b]
        [ _h, _one ] -> addBatchView b a           -- [h×1] -> [h×1×b] (simple view)
        [ _h, _b  ]  -> pure a                     -- already [h×b]
        _            -> addBatchView b a           -- fallback: add last axis
