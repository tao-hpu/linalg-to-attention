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
          <Route path="ch/matrix-mult" element={<Ch02MatrixMult />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
