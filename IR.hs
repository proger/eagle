{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

module IR
  ( -- * Scalar/data types, shapes, layout
    DType(..)
  , Dim(..), Shape
  , Layout(..)

    -- * Placement (abstract, then specialized)
  , Axis(..)
  , SBP(..)
  , Placement
  , PlaceAnn(..)

    -- * Tensor type
  , Ty(..)

    -- * Variables, statements, programs
  , V(..)
  , Stmt(..)
  , Prog(..)

    -- * Operators
  , MMA(..)
  , Collective(..)
  , PW(..)
  , Red(..)
  , Op(..)

    -- * Pattern synonyms for pleasant matching
  , pattern PMap
  , pattern PZip
  , pattern PReduce
  , pattern PMatMul
  , pattern PMatMulEpi
  , pattern PColl

    -- * Helpers
  , tyOf
  , shOf
  , freshen
  , bind
  , lookupLet
  , dropLet
  , renameVarInOp

    -- * Pretty printers
  , ppTy
  , ppStmt
  ) where

import GHC.Generics (Generic)
import Data.List (intercalate)
import qualified Data.Text as T
import qualified Data.Map.Strict as M

--------------------------------------------------------------------------------
-- Types & shapes

data DType = F16 | BF16 | F32 | I32
  deriving (Eq, Ord, Show, Generic)

-- symbolic-friendly dimension
data Dim = static !Int | sym !T.Text
  deriving (Eq, Ord, Generic)
instance Show Dim where
  show (static n) = show n
  show (sym s)    = T.unpack s

type Shape = [Dim]

data Layout = RowMajor | ColMajor
  deriving (Eq, Ord, Show, Generic)

--------------------------------------------------------------------------------
-- Placement (algorithm/schedule separation)

-- Tensor axis index (for ops like Reduce over axis=Int)
newtype Axis = Axis { unAxis :: Int }
  deriving (Eq, Ord, Show, Generic)

-- Split/Broadcast/Partial over *device-mesh* axes (e.g., row=0, col=1)
data SBP = Split Axis | Broadcast | PartialSum
  deriving (Eq, Ord, Show, Generic)

type Placement = [SBP]      -- length == mesh rank; [] ~ replicated

data PlaceAnn
  = PUnknown T.Text         -- symbolic placement var, solved by planner (e.g., "α_Wx")
  | PConcrete Placement     -- specialized/known placement after planning
  deriving (Eq, Ord, Show, Generic)

--------------------------------------------------------------------------------
-- Tensor type

data Ty = Tensor
  { dtype  :: !DType
  , shape  :: !Shape
  , layout :: !Layout
  , place  :: !PlaceAnn
  } deriving (Eq, Ord, Show, Generic)

--------------------------------------------------------------------------------
-- Vars, statements, programs (A-normal form)

newtype V = V { vId :: Int } deriving (Eq, Ord)
instance Show V where show (V i) = "%" ++ show i

-- A statement binds the result of an Op to a Var
data Stmt = Let !V !Op
  deriving (Eq, Ord, Show, Generic)

-- A program is a sequence of lets, with a designated return var
data Prog = Prog
  { pDefs   :: ![Stmt]
  , pReturn :: !V
  } deriving (Eq, Ord, Show, Generic)

--------------------------------------------------------------------------------
-- Ops

-- side metadata for codegen/tiling (kept minimal here)
data MMA = MMA16x16x16 | MMA32x8x16
  deriving (Eq, Ord, Show, Generic)

-- Distributed collectives (the planner inserts these)
data Collective
  = AllReduceSum Axis      -- allreduce(sum) along a mesh axis
  | AllGather   Axis       -- concatenate shards along a mesh axis
  | ReduceScatter Axis     -- reduce-scatter along a mesh axis
  | BroadcastTo Placement  -- convert to a concrete placement (cast)
  deriving (Eq, Ord, Show, Generic)

-- Pointwise function id (codegen decides lowering)
data PW
  = Add | Sub | Mul | Div | Neg | Tanh | Relu
  deriving (Eq, Ord, Show, Generic)

-- Reductions
data Red = RSum | RMax | RMean
  deriving (Eq, Ord, Show, Generic)

-- Core operators (matchable)
data Op
  = Const      { oTy :: !Ty, oBytes :: [Double] }             -- tiny demo const
  | Param      { oTy :: !Ty, oName :: T.Text }                -- graph inputs
  | Copy       { oTy :: !Ty, from :: V }                      -- explicit copy / layout cast
  | Reshape    { oTy :: !Ty, x :: V }
  | Transpose  { oTy :: !Ty, perm :: [Int], x :: V }

  | Map        { oTy :: !Ty, fun :: PW, x :: V }              -- unary map
  | ZipWith    { oTy :: !Ty, fun2 :: PW, x :: V, y :: V }     -- binary map (e.g., Add)
  | Reduce     { oTy :: !Ty, rop :: Red, axis :: Axis, x :: V }

  | MatVec     { oTy :: !Ty, a :: V, v :: V, transA :: !Bool, mma :: !MMA }
  | MatMul     { oTy :: !Ty, a :: V, b :: V
               , transA :: !Bool, transB :: !Bool, mma :: !MMA }

  -- Fused epilogue (e.g., add bias then activation)
  | MatMulEpilogue
      { oTy :: !Ty, a :: V, b :: V, transA :: !Bool, transB :: !Bool
      , mma :: !MMA, epilogue :: Maybe (PW, Maybe V) }

  -- Explicit collectives (inserted by planner)
  | Collective { oTy :: !Ty, coll :: !Collective, x :: V }

  deriving (Eq, Ord, Show, Generic)

--------------------------------------------------------------------------------
-- Pattern synonyms to make matching pleasant

pattern PMap :: Ty -> PW -> V -> Op
pattern PMap t f v           = Map t f v

pattern PZip :: Ty -> PW -> V -> V -> Op
pattern PZip t f u v         = ZipWith t f u v

pattern PReduce :: Ty -> Red -> Axis -> V -> Op
pattern PReduce t r ax v     = Reduce t r ax v

pattern PMatMul :: Ty -> V -> V -> Bool -> Bool -> MMA -> Op
pattern PMatMul t a b ta tb m = MatMul t a b ta tb m

pattern PMatMulEpi :: Ty -> V -> V -> Bool -> Bool -> MMA -> Maybe (PW, Maybe V) -> Op
pattern PMatMulEpi t a b ta tb m e = MatMulEpilogue t a b ta tb m e

pattern PColl :: Ty -> Collective -> V -> Op
pattern PColl t k v          = Collective t k v

{-# COMPLETE
    Const, Param, Copy, Reshape, Transpose
  , Map, ZipWith, Reduce
  , MatVec, MatMul, MatMulEpilogue
  , Collective
  #-}

--------------------------------------------------------------------------------
-- Helpers

tyOf :: Op -> Ty
tyOf = oTy

shOf :: Op -> Shape
shOf = shape . oTy

-- Fresh var (based on current block length)
freshen :: [Stmt] -> V
freshen defs = V (length defs)

-- Append a binding to the block
bind :: Op -> [Stmt] -> (V, [Stmt])
bind op defs = let v = freshen defs in (v, defs ++ [Let v op])

-- Lookup a definition by var in a block
lookupLet :: V -> [Stmt] -> Maybe Stmt
lookupLet v = go
  where
    go [] = Nothing
    go (s@(Let v' _):xs)
      | v == v'   = Just s
      | otherwise = go xs

-- Drop a binding by var
dropLet :: V -> [Stmt] -> [Stmt]
dropLet v = filter (\(Let v' _) -> v' /= v)

-- Rename vars inside an Op (used by CSE / rewriting)
renameVarInOp :: M.Map V V -> Op -> Op
renameVarInOp sub = go
  where
    r v = M.findWithDefault v v sub
    go (Copy t a)                        = Copy t (r a)
    go (Reshape t a)                     = Reshape t (r a)
    go (Transpose t p a)                 = Transpose t p (r a)
    go (Map t f a)                       = Map t f (r a)
    go (ZipWith t f a b)                 = ZipWith t f (r a) (r b)
    go (Reduce t red ax a)               = Reduce t red ax (r a)
    go (MatVec t a b ta m)               = MatVec t (r a) (r b) ta m
    go (MatMul t a b ta tb m)            = MatMul t (r a) (r b) ta tb m
    go (MatMulEpilogue t a b ta tb m me) = MatMulEpilogue t (r a) (r b) ta tb m
                                           (fmap (\(pw, mb)->(pw, fmap r mb)) me)
    go (Collective t k a)                = Collective t k (r a)
    go op@Const{}                        = op
    go op@Param{}                        = op

--------------------------------------------------------------------------------
-- Pretty printing

ppTy :: Ty -> String
ppTy (Tensor dt sh lo pl) =
  show dt <> " [" <> intercalate "×" (map show sh) <> "] "
  <> show lo <> " @" <> show pl

ppStmt :: Stmt -> String
ppStmt (Let v op) = show v <> " = " <> case op of
  Const t _             -> "Const :" <> ppTy t
  Param t nm            -> "Param " <> T.unpack nm <> " :" <> ppTy t
  Copy t a              -> "Copy " <> show a <> " :" <> ppTy t
  Reshape t a           -> "Reshape " <> show a <> " :" <> ppTy t
  Transpose t p a       -> "Transpose " <> show p <> " " <> show a <> " :" <> ppTy t
  Map t f a             -> "Map " <> show f <> " " <> show a <> " :" <> ppTy t
  ZipWith t f a b       -> "ZipWith " <> show f <> " " <> show a <> " " <> show b <> " :" <> ppTy t
  Reduce t r (Axis ax) a->
    "Reduce " <> show r <> " axis=" <> show ax <> " " <> show a <> " :" <> ppTy t
  MatVec t a b ta m     -> "MatVec " <> show a <> "·" <> show b <> " ta=" <> show ta <> " " <> show m <> " :" <> ppTy t
  MatMul t a b ta tb m  -> "MatMul " <> show a <> "·" <> show b <> " ta=" <> show ta <> " tb=" <> show tb
                           <> " " <> show m <> " :" <> ppTy t
  MatMulEpilogue t a b _ _ m e ->
    "MatMulEpilogue " <> show a <> "·" <> show b <> " " <> show m
    <> " epi=" <> show e <> " :" <> ppTy t
  Collective t k a      -> "Collective " <> show k <> " " <> show a <> " :" <> ppTy t
