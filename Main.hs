{-# LANGUAGE OverloadedStrings #-}
module Main where

import qualified Data.Text as T
import           IR
import           Planner
import           Surface
import           Rewrite

-- Build a single batched RNN step in the surface DSL
buildBatchedRnnStep :: Prog
buildBatchedRnnStep =
  runBuild $ do
    let dt = BF16
        h  = sym "h"; x = sym "x"; b = sym "b"
        tWx = Tensor dt [h,x] RowMajor (PUnknown "α_Wx")
        tWh = Tensor dt [h,h] RowMajor (PUnknown "α_Wh")
        tB  = Tensor dt [h]   RowMajor (PUnknown "α_b")
        tX  = Tensor dt [x]   RowMajor (PUnknown "α_x")
        tH  = Tensor dt [h]   RowMajor (PUnknown "α_h")

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
  printProg "=== Replicated (before optimize) ===" progRep
  printProg "=== Replicated (after  optimize) ===" progRepOpt

  printProg "=== Split-by-columns (before optimize) ===" progCol
  printProg "=== Split-by-columns (after  optimize) ===" progColOpt

  -- If you have a PyTorch emitter (e.g., EmitterTorch.emitTorchToFile), you can enable:
  -- import EmitterTorch (emitTorchToFile)
  -- emitTorchToFile "compiled_rnn_step.py" progColOpt
  -- putStrLn "Wrote compiled_rnn_step.py"
