import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { Layout } from './site/Layout'
import { Home } from './pages/Home'
import './site.css'

// 章节页全部按需加载（lazy）——首页只下载外壳 + Home，点哪节才拉哪节的代码。
// 页面是具名导出，所以用 .then 取出对应的命名成员当 default。
const L = <T,>(p: Promise<T>, k: keyof T) =>
  lazy(() => p.then((m) => ({ default: m[k] as React.ComponentType })))

const Ch01Vectors = L(import('./pages/Ch01Vectors'), 'Ch01Vectors')
const BowEmbedding = L(import('./pages/BowEmbedding'), 'BowEmbedding')
const VectorArithmetic = L(import('./pages/VectorArithmetic'), 'VectorArithmetic')
const DotProduct = L(import('./pages/DotProduct'), 'DotProduct')
const Projection = L(import('./pages/Projection'), 'Projection')
const Norms = L(import('./pages/Norms'), 'Norms')
const Ch02MatrixMult = L(import('./pages/Ch02MatrixMult'), 'Ch02MatrixMult')
const MatrixAsTransform = L(import('./pages/MatrixAsTransform'), 'MatrixAsTransform')
const MatmulViews = L(import('./pages/MatmulViews'), 'MatmulViews')
const TransposeShape = L(import('./pages/TransposeShape'), 'TransposeShape')
const Batching = L(import('./pages/Batching'), 'Batching')
const Determinant = L(import('./pages/Determinant'), 'Determinant')
const Rank = L(import('./pages/Rank'), 'Rank')
const Inverse = L(import('./pages/Inverse'), 'Inverse')
const Eigen = L(import('./pages/Eigen'), 'Eigen')
const Spectral = L(import('./pages/Spectral'), 'Spectral')
const OrthogonalRotation = L(import('./pages/OrthogonalRotation'), 'OrthogonalRotation')
const OrthogonalProjection = L(import('./pages/OrthogonalProjection'), 'OrthogonalProjection')
const GramSchmidt = L(import('./pages/GramSchmidt'), 'GramSchmidt')
const RegressionProjection = L(import('./pages/RegressionProjection'), 'RegressionProjection')
const SVD = L(import('./pages/SVD'), 'SVD')
const LowRank = L(import('./pages/LowRank'), 'LowRank')
const PCA = L(import('./pages/PCA'), 'PCA')
const Gradient = L(import('./pages/Gradient'), 'Gradient')
const ChainRule = L(import('./pages/ChainRule'), 'ChainRule')
const GradientDescent = L(import('./pages/GradientDescent'), 'GradientDescent')
const Regularization = L(import('./pages/Regularization'), 'Regularization')
const Normalization = L(import('./pages/Normalization'), 'Normalization')
const Softmax = L(import('./pages/Softmax'), 'Softmax')
const CrossEntropy = L(import('./pages/CrossEntropy'), 'CrossEntropy')
const SelfAttention = L(import('./pages/SelfAttention'), 'SelfAttention')
const MultiHeadAttention = L(import('./pages/MultiHeadAttention'), 'MultiHeadAttention')
const Rope = L(import('./pages/Rope'), 'Rope')
const TransformerBlock = L(import('./pages/TransformerBlock'), 'TransformerBlock')
const LoraFinetuning = L(import('./pages/LoraFinetuning'), 'LoraFinetuning')
const SamplingDecoding = L(import('./pages/SamplingDecoding'), 'SamplingDecoding')
const Quantization = L(import('./pages/Quantization'), 'Quantization')
const ForwardPass = L(import('./pages/ForwardPass'), 'ForwardPass')

function PageFallback() {
  return <div className="page-loading" role="status" aria-live="polite">加载中…</div>
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route
            path="ch/*"
            element={
              <Suspense fallback={<PageFallback />}>
                <Routes>
                  <Route path="vectors" element={<Ch01Vectors />} />
                  <Route path="bow-to-embedding" element={<BowEmbedding />} />
                  <Route path="vector-arithmetic" element={<VectorArithmetic />} />
                  <Route path="dot-product" element={<DotProduct />} />
                  <Route path="projection" element={<Projection />} />
                  <Route path="norms" element={<Norms />} />
                  <Route path="matrix-as-transform" element={<MatrixAsTransform />} />
                  <Route path="matrix-mult" element={<Ch02MatrixMult />} />
                  <Route path="matmul-views" element={<MatmulViews />} />
                  <Route path="transpose-shape" element={<TransposeShape />} />
                  <Route path="batching" element={<Batching />} />
                  <Route path="determinant" element={<Determinant />} />
                  <Route path="rank" element={<Rank />} />
                  <Route path="inverse" element={<Inverse />} />
                  <Route path="eigen" element={<Eigen />} />
                  <Route path="spectral" element={<Spectral />} />
                  <Route path="orthogonal-rotation" element={<OrthogonalRotation />} />
                  <Route path="orthogonal-projection" element={<OrthogonalProjection />} />
                  <Route path="gram-schmidt" element={<GramSchmidt />} />
                  <Route path="regression-projection" element={<RegressionProjection />} />
                  <Route path="svd" element={<SVD />} />
                  <Route path="low-rank" element={<LowRank />} />
                  <Route path="pca" element={<PCA />} />
                  <Route path="gradient" element={<Gradient />} />
                  <Route path="chain-rule" element={<ChainRule />} />
                  <Route path="gradient-descent" element={<GradientDescent />} />
                  <Route path="regularization" element={<Regularization />} />
                  <Route path="normalization" element={<Normalization />} />
                  <Route path="softmax" element={<Softmax />} />
                  <Route path="cross-entropy" element={<CrossEntropy />} />
                  <Route path="self-attention" element={<SelfAttention />} />
                  <Route path="multi-head" element={<MultiHeadAttention />} />
                  <Route path="rope" element={<Rope />} />
                  <Route path="transformer-block" element={<TransformerBlock />} />
                  <Route path="lora-finetuning" element={<LoraFinetuning />} />
                  <Route path="sampling-decoding" element={<SamplingDecoding />} />
                  <Route path="quantization" element={<Quantization />} />
                  <Route path="forward-pass" element={<ForwardPass />} />
                </Routes>
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
