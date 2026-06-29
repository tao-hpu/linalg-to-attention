import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './site/Layout'
import { Home } from './pages/Home'
import { Ch01Vectors } from './pages/Ch01Vectors'
import { BowEmbedding } from './pages/BowEmbedding'
import { VectorArithmetic } from './pages/VectorArithmetic'
import { DotProduct } from './pages/DotProduct'
import { Projection } from './pages/Projection'
import { Norms } from './pages/Norms'
import { Ch02MatrixMult } from './pages/Ch02MatrixMult'
import { MatrixAsTransform } from './pages/MatrixAsTransform'
import { MatmulViews } from './pages/MatmulViews'
import { TransposeShape } from './pages/TransposeShape'
import { Batching } from './pages/Batching'
import { Determinant } from './pages/Determinant'
import { Rank } from './pages/Rank'
import { Inverse } from './pages/Inverse'
import { Eigen } from './pages/Eigen'
import { Spectral } from './pages/Spectral'
import { OrthogonalRotation } from './pages/OrthogonalRotation'
import { OrthogonalProjection } from './pages/OrthogonalProjection'
import { GramSchmidt } from './pages/GramSchmidt'
import { RegressionProjection } from './pages/RegressionProjection'
import { SVD } from './pages/SVD'
import { LowRank } from './pages/LowRank'
import { PCA } from './pages/PCA'
import { Gradient } from './pages/Gradient'
import { ChainRule } from './pages/ChainRule'
import { GradientDescent } from './pages/GradientDescent'
import { Regularization } from './pages/Regularization'
import { Normalization } from './pages/Normalization'
import { Softmax } from './pages/Softmax'
import { CrossEntropy } from './pages/CrossEntropy'
import { SelfAttention } from './pages/SelfAttention'
import { MultiHeadAttention } from './pages/MultiHeadAttention'
import { TransformerBlock } from './pages/TransformerBlock'
import { LoraFinetuning } from './pages/LoraFinetuning'
import { SamplingDecoding } from './pages/SamplingDecoding'
import { Quantization } from './pages/Quantization'
import './site.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="ch/vectors" element={<Ch01Vectors />} />
          <Route path="ch/bow-to-embedding" element={<BowEmbedding />} />
          <Route path="ch/vector-arithmetic" element={<VectorArithmetic />} />
          <Route path="ch/dot-product" element={<DotProduct />} />
          <Route path="ch/projection" element={<Projection />} />
          <Route path="ch/norms" element={<Norms />} />
          <Route path="ch/matrix-as-transform" element={<MatrixAsTransform />} />
          <Route path="ch/matrix-mult" element={<Ch02MatrixMult />} />
          <Route path="ch/matmul-views" element={<MatmulViews />} />
          <Route path="ch/transpose-shape" element={<TransposeShape />} />
          <Route path="ch/batching" element={<Batching />} />
          <Route path="ch/determinant" element={<Determinant />} />
          <Route path="ch/rank" element={<Rank />} />
          <Route path="ch/inverse" element={<Inverse />} />
          <Route path="ch/eigen" element={<Eigen />} />
          <Route path="ch/spectral" element={<Spectral />} />
          <Route path="ch/orthogonal-rotation" element={<OrthogonalRotation />} />
          <Route path="ch/orthogonal-projection" element={<OrthogonalProjection />} />
          <Route path="ch/gram-schmidt" element={<GramSchmidt />} />
          <Route path="ch/regression-projection" element={<RegressionProjection />} />
          <Route path="ch/svd" element={<SVD />} />
          <Route path="ch/low-rank" element={<LowRank />} />
          <Route path="ch/pca" element={<PCA />} />
          <Route path="ch/gradient" element={<Gradient />} />
          <Route path="ch/chain-rule" element={<ChainRule />} />
          <Route path="ch/gradient-descent" element={<GradientDescent />} />
          <Route path="ch/regularization" element={<Regularization />} />
          <Route path="ch/normalization" element={<Normalization />} />
          <Route path="ch/softmax" element={<Softmax />} />
          <Route path="ch/cross-entropy" element={<CrossEntropy />} />
          <Route path="ch/self-attention" element={<SelfAttention />} />
          <Route path="ch/multi-head" element={<MultiHeadAttention />} />
          <Route path="ch/transformer-block" element={<TransformerBlock />} />
          <Route path="ch/lora-finetuning" element={<LoraFinetuning />} />
          <Route path="ch/sampling-decoding" element={<SamplingDecoding />} />
          <Route path="ch/quantization" element={<Quantization />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
