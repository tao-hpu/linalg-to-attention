import { useState, useRef, type Dispatch, type SetStateAction } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Color palette ─────────────────────────────────────────────────────────────
const RUST = '#c75b39'
const IKB  = '#002fa7'

// ── SVG layout constants ──────────────────────────────────────────────────────
const SIZE  = 380
const RANGE = 3.0
const CX    = SIZE / 2   // 190 — SVG center = data centroid
const CY    = SIZE / 2   // 190
const UNIT  = SIZE / 2 / RANGE   // px per data unit ≈ 63.3

const toSX = (x: number): number => CX + x * UNIT
const toSY = (y: number): number => CY - y * UNIT   // SVG y flipped

const f2 = (n: number): string => {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── Preset cloud data ─────────────────────────────────────────────────────────
// Raw 2-D points in data units; centroid ≈ (0,0) — centering done in computePCA.
// Presets are just convenient *starting points*; every point is draggable.
type PresetKey = 'strong' | 'weak' | 'iso'

// A draggable data point lives in raw data coordinates (NOT pre-centered).
type DataPoint = { id: number; x: number; y: number }

const toDataPoints = (raw: readonly [number, number][]): DataPoint[] =>
  raw.map(([x, y], id) => ({ id, x, y }))

const CLOUDS: Record<PresetKey, [number, number][]> = {
  // 强相关: elongated cloud at ≈35°
  strong: [
    [-2.4, -1.5], [-2.0, -1.1], [-1.6, -0.8], [-1.3, -0.6], [-0.9, -0.3],
    [-0.5, -0.1], [-0.1,  0.2], [ 0.3,  0.4], [ 0.7,  0.3], [ 1.0,  0.7],
    [ 1.4,  0.9], [ 1.7,  1.1], [ 2.1,  1.4], [ 2.4,  1.6], [-0.8,  0.5],
  ],
  // 弱相关: moderate spread in both directions
  weak: [
    [-1.6, -0.8], [-1.3, -1.2], [-1.0, -0.3], [-0.7,  0.6], [-0.4, -0.9],
    [-0.2,  0.8], [ 0.1, -0.5], [ 0.4,  0.7], [ 0.6, -0.4], [ 0.9,  1.0],
    [ 1.1, -0.6], [ 1.4,  0.5], [ 1.7, -0.2], [-1.9,  0.4], [ 1.9,  0.9],
  ],
  // 各向同性: roughly circular cloud, no dominant direction
  iso: [
    [ 0.8,  1.2], [-1.1,  0.9], [ 1.3, -0.4], [-0.6, -1.3], [ 0.2,  1.5],
    [-1.4, -0.2], [ 1.0, -1.1], [-0.3,  0.7], [ 0.9,  0.4], [-1.0, -0.8],
    [ 0.5, -1.4], [-0.8,  1.1], [ 1.2,  0.7], [-1.3, -0.5], [ 0.3, -0.6],
  ],
}

const PRESET_META: { key: PresetKey; label: string }[] = [
  { key: 'strong', label: '强相关' },
  { key: 'weak',   label: '弱相关' },
  { key: 'iso',    label: '各向同性圆' },
]

// ── PCA computation (exact closed form for 2×2 symmetric covariance) ──────────
interface PCAResult {
  mx: number             // centroid x (data units)
  my: number             // centroid y (data units)
  cxx: number
  cxy: number
  cyy: number
  lam1: number
  lam2: number
  v1: [number, number]   // unit eigenvector for lam1 (PC1, max variance)
  v2: [number, number]   // unit eigenvector for lam2 (PC2, ⊥ PC1)
}

function computePCA(pts: readonly DataPoint[]): PCAResult {
  const n  = pts.length
  const mx = pts.reduce((s, p) => s + p.x, 0) / n
  const my = pts.reduce((s, p) => s + p.y, 0) / n

  // 1. Covariance matrix C = (1/n) XᵀX over centered points (xᵢ − x̄, yᵢ − ȳ)
  const cxx = pts.reduce((s, p) => s + (p.x - mx) * (p.x - mx), 0) / n
  const cyy = pts.reduce((s, p) => s + (p.y - my) * (p.y - my), 0) / n
  const cxy = pts.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0) / n

  // 2. Eigenvalues of symmetric [[cxx, cxy], [cxy, cyy]]
  //    λ = (cxx+cyy)/2 ± sqrt(((cxx−cyy)/2)² + cxy²)
  const mid  = (cxx + cyy) / 2
  const disc = Math.sqrt(((cxx - cyy) / 2) ** 2 + cxy * cxy)
  const lam1 = mid + disc
  const lam2 = Math.max(0, mid - disc)   // clamp for floating-point safety (C is PSD)

  // 3. Eigenvectors
  let v1: [number, number], v2: [number, number]
  if (Math.abs(cxy) < 1e-10) {
    // Diagonal matrix — standard-basis eigenvectors ordered by eigenvalue
    if (cxx >= cyy) { v1 = [1, 0]; v2 = [0, 1] }
    else             { v1 = [0, 1]; v2 = [1, 0] }
  } else {
    // General: [cxy, λ₁ − cxx] is eigenvector for λ₁
    const rx = cxy, ry = lam1 - cxx
    const len = Math.hypot(rx, ry)
    v1 = [rx / len, ry / len]
    v2 = [-v1[1], v1[0]]   // 90° rotation → always ⊥ v1
  }

  return { mx, my, cxx, cxy, cyy, lam1, lam2, v1, v2 }
}

// ── Principal-component axis arrow (bidirectional, scaled to √λ = std dev) ────
function PCAxis({
  v, sdScale, color, label, origin,
}: {
  v: [number, number]
  sdScale: number
  color: string
  label: string
  origin: [number, number]   // axis origin in data units (the centroid)
}) {
  const [vx, vy] = v
  if (sdScale < 1e-6) return null

  const [ox, oy] = origin
  const orgX = toSX(ox),               orgY = toSY(oy)
  const tipX = toSX(ox + vx * sdScale), tipY = toSY(oy + vy * sdScale)
  const negX = toSX(ox - vx * sdScale), negY = toSY(oy - vy * sdScale)

  // Arrowhead
  const dx = tipX - orgX, dy = tipY - orgY
  const dlen = Math.hypot(dx, dy) || 1
  const ux = dx / dlen, uy = dy / dlen
  const AH = 9, AW = 4.5
  const bx = tipX - AH * ux, by = tipY - AH * uy
  const ahPts = `${tipX},${tipY} ${bx - AW * uy},${by + AW * ux} ${bx + AW * uy},${by - AW * ux}`

  // Label: pushed outward beyond the tip
  const lx = tipX + ux * 22
  const ly = tipY + uy * 22 + 4
  const anchor: 'start' | 'middle' | 'end' =
    ux > 0.15 ? 'start' : ux < -0.15 ? 'end' : 'middle'

  return (
    <g>
      {/* Dashed negative half */}
      <line
        x1={orgX} y1={orgY} x2={negX} y2={negY}
        stroke={color} strokeWidth={1.5} strokeDasharray="4 3"
        strokeLinecap="round" opacity={0.4}
      />
      {/* Solid shaft (stops at arrowhead base) */}
      <line
        x1={orgX} y1={orgY} x2={bx} y2={by}
        stroke={color} strokeWidth={2.5} strokeLinecap="round"
      />
      <polygon points={ahPts} fill={color} />
      <text
        x={lx} y={ly}
        fill={color} fontSize={11} fontWeight="bold"
        textAnchor={anchor}
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {label}
      </text>
    </g>
  )
}

// ── Interactive scatter + PCA canvas ─────────────────────────────────────────
function PCACanvas({
  pts, setPts, mx, my, v1, v2, lam1, lam2, showProj,
}: {
  pts: DataPoint[]
  setPts: Dispatch<SetStateAction<DataPoint[]>>
  mx: number
  my: number
  v1: [number, number]
  v2: [number, number]
  lam1: number
  lam2: number
  showProj: boolean
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dragId, setDragId] = useState<number | null>(null)

  const sd1 = Math.sqrt(lam1)
  const sd2 = Math.sqrt(lam2)

  // Convert a pointer's client position → data coordinates, robust to any CSS
  // scaling of the SVG. getScreenCTM() maps user-space (viewBox) → screen, so
  // its inverse maps screen → viewBox; we then undo toSX/toSY. Falls back to
  // bounding-rect scaling (clientX·SIZE/rect.width) if no CTM is available.
  function clientToData(clientX: number, clientY: number): { x: number; y: number } {
    const svg = svgRef.current
    let vx: number, vy: number
    if (svg && svg.getScreenCTM()) {
      const ctm = svg.getScreenCTM()!
      const p = new DOMPoint(clientX, clientY).matrixTransform(ctm.inverse())
      vx = p.x; vy = p.y
    } else if (svg) {
      const rect = svg.getBoundingClientRect()
      vx = (clientX - rect.left) * SIZE / rect.width
      vy = (clientY - rect.top)  * SIZE / rect.height
    } else {
      vx = CX; vy = CY
    }
    // viewBox px → data units, then clamp to the visible window
    const clamp = (v: number) => Math.max(-RANGE, Math.min(RANGE, v))
    return {
      x: Math.round(clamp((vx - CX) / UNIT) * 20) / 20,   // 0.05 step
      y: Math.round(clamp((CY - vy) / UNIT) * 20) / 20,
    }
  }

  function onMove(clientX: number, clientY: number) {
    if (dragId === null) return
    const { x, y } = clientToData(clientX, clientY)
    setPts(prev => prev.map(p => (p.id === dragId ? { ...p, x, y } : p)))
  }

  // Grid lines
  const gridLines: JSX.Element[] = []
  for (let k = -3; k <= 3; k++) {
    const gStroke = k === 0 ? '#9aa5b0' : '#e6e8ea'
    const gSw     = k === 0 ? 1.5 : 1
    gridLines.push(
      <line key={`gv${k}`}
        x1={toSX(k)} y1={toSY(-RANGE)} x2={toSX(k)} y2={toSY(RANGE)}
        stroke={gStroke} strokeWidth={gSw}
      />,
      <line key={`gh${k}`}
        x1={toSX(-RANGE)} y1={toSY(k)} x2={toSX(RANGE)} y2={toSY(k)}
        stroke={gStroke} strokeWidth={gSw}
      />,
    )
  }

  // Orthogonal projection of each point onto the PC1 line through the centroid
  const projData = pts.map(p => {
    const dx = p.x - mx, dy = p.y - my
    const t = dx * v1[0] + dy * v1[1]            // scalar coord along PC1
    return { px: p.x, py: p.y, qx: mx + t * v1[0], qy: my + t * v1[1] }
  })

  return (
    <svg
      ref={svgRef}
      width={SIZE} height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      style={{
        width: '100%', height: 'auto', maxWidth: SIZE, display: 'block',
        touchAction: 'none', userSelect: 'none',
        cursor: dragId !== null ? 'grabbing' : 'default',
      }}
      aria-label="PCA 主成分散点图（数据点可拖动）"
      onPointerMove={(e) => onMove(e.clientX, e.clientY)}
      onPointerUp={() => setDragId(null)}
      onPointerLeave={() => setDragId(null)}
    >
      <defs>
        <clipPath id="pca-clip">
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
      </defs>

      <g clipPath="url(#pca-clip)">
        {/* Grid */}
        {gridLines}

        {/* PC2 axis drawn first (lower z-order) */}
        <PCAxis v={v2} sdScale={sd2} color={IKB}  label={`PC2  λ₂=${f2(lam2)}`} origin={[mx, my]} />
        {/* PC1 axis on top of PC2 */}
        <PCAxis v={v1} sdScale={sd1} color={RUST} label={`PC1  λ₁=${f2(lam1)}`} origin={[mx, my]} />

        {/* Projection drop-lines + collapsed 1-D points on PC1 */}
        {showProj && projData.map((d, i) => (
          <g key={`drop-${i}`}>
            <line
              x1={toSX(d.px)} y1={toSY(d.py)}
              x2={toSX(d.qx)} y2={toSY(d.qy)}
              stroke="#aab0ba" strokeWidth={1} strokeDasharray="3 2"
            />
            <circle
              cx={toSX(d.qx)} cy={toSY(d.qy)} r={4.5}
              fill={RUST} opacity={0.88}
            />
          </g>
        ))}

        {/* Centroid marker (data mean) — axes pivot here */}
        <circle cx={toSX(mx)} cy={toSY(my)} r={4} fill="#5b6168" />
        <circle cx={toSX(mx)} cy={toSY(my)} r={8} fill="none" stroke="#5b6168" strokeWidth={1} opacity={0.5} />

        {/* Original data points (IKB) — draggable */}
        {pts.map(p => (
          <circle
            key={`pt-${p.id}`}
            cx={toSX(p.x)} cy={toSY(p.y)} r={dragId === p.id ? 7.5 : 6}
            fill={IKB} fillOpacity={showProj ? 0.42 : 0.82}
            stroke="#fff" strokeWidth={1.2}
            style={{ cursor: dragId !== null ? 'grabbing' : 'grab' }}
            onPointerDown={(e) => {
              ;(e.target as Element).setPointerCapture(e.pointerId)
              setDragId(p.id)
            }}
          />
        ))}
      </g>
    </svg>
  )
}

// ── Python code snippet ───────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

# 数据矩阵 X: n 行 d 列（每行一个样本；这里 d = 2）
X = np.array([...])

# 1. 中心化：减去每列均值
X_c = X - X.mean(axis=0)

# 2. 协方差矩阵 C = (1/n) XᵀX   ← 对称正半定阵
n = len(X_c)
C = X_c.T @ X_c / n          # (d × d)

# 3. 谱分解（eigh 专为实对称阵，保证实数 eigenvalue，升序返回）
eigvals, eigvecs = np.linalg.eigh(C)   # eigvecs[:, k] = 单位 PC 方向

# 4. 降序排列（eigh 升序 → PCA 要大的在前）
idx     = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]        # λ₁ ≥ λ₂ ≥ 0
eigvecs = eigvecs[:, idx]     # 第 0 列 = PC1

# 5. Explained variance ratio
ratio = eigvals / eigvals.sum()        # 例如 [0.82, 0.18]

# 6. 降维：投影到前 k 个 principal component
X_1d = X_c @ eigvecs[:, :1]           # (n, 1)  ← 降到 1-D

# 注：PCA == svd(X_centered)
# U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
# eigvals == S**2 / n  ；  eigvecs == Vt.T`

// ── Main page component ───────────────────────────────────────────────────────
export function PCA() {
  const [preset,   setPreset]   = useState<PresetKey>('strong')
  const [pts,      setPts]      = useState<DataPoint[]>(() => toDataPoints(CLOUDS.strong))
  const [showProj, setShowProj] = useState(false)

  const { mx, my, cxx, cxy, cyy, lam1, lam2, v1, v2 } = computePCA(pts)

  // Load a preset as a fresh, fully-editable starting point
  const loadPreset = (key: PresetKey) => {
    setPreset(key)
    setPts(toDataPoints(CLOUDS[key]))
    setShowProj(false)
  }

  const totalVar  = lam1 + lam2
  const explained = totalVar > 1e-12 ? lam1 / totalVar : 1

  const me = findChapter('pca')!
  const { prev, next } = neighbors('pca')

  // ── Preset button style ──────────────────────────────────────────────────
  const presetStyle = (active: boolean) => ({
    padding: '0.35rem 0.9rem',
    border: `1.5px solid ${IKB}`,
    borderRadius: 4,
    background: active ? IKB : 'white',
    color: active ? 'white' : IKB,
    cursor: 'pointer' as const,
    fontSize: '0.85rem',
    fontFamily: 'inherit',
    fontWeight: active ? 600 : 400,
  })

  return (
    <article className="page">

      {/* ── Masthead ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第五部分 · 降维：抓住主要矛盾
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          PCA 主成分分析
          <span className="zh-sub">最大方差的方向，就是信息最密集的方向</span>
        </h1>
        <p className="lede">
          PCA 是最常用的降维方法，几何上就是这件事：
          把数据<strong>中心化</strong>（减去均值），构造{' '}
          <strong>covariance matrix</strong>{' '}
          <code>C = (1/n) XᵀX</code>，
          对它做<strong>谱分解（16 节）</strong>——
          eigenvector 就是 <strong>principal component（主成分）</strong>，
          eigenvalue 就是沿该方向的 <strong>variance（方差）</strong>。
          PC1 = 数据散布最宽的方向；PC2 ⊥ PC1 且方差次之。
          <strong>降维</strong>就是只保留 PC1，把点投影上去，
          用一条轴留住尽可能多的信息。下面亲眼看一下。
        </p>
      </header>

      {/* ── Controls ── */}
      <section className="controls">

        {/* Cloud preset selector */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">起始数据云</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>选一个起点，然后直接拖动散点图里的蓝点</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESET_META.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => loadPreset(key)}
                style={presetStyle(preset === key)}
              >
                {label}
              </button>
            ))}
          </div>
          <p style={{ fontSize: 13, color: '#5b6168', margin: '8px 0 0', maxWidth: '52ch' }}>
            预设只是<strong>懒得手调时的起点</strong>。拖动任意数据点，
            协方差矩阵 C、特征值 λ₁/λ₂、主轴方向和解释方差比都会<strong>实时重算</strong>；
            想从头来就再点一次形状按钮重置。
          </p>
        </div>

        {/* Dimensionality-reduction toggle */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">降维操作</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>投影到 PC1（从 2-D 降到 1-D）</span>
          </div>
          <div style={{ paddingTop: '0.4rem' }}>
            <button
              onClick={() => setShowProj(v => !v)}
              style={{
                padding: '0.35rem 0.9rem',
                border: `1.5px solid ${RUST}`,
                borderRadius: 4,
                background: showProj ? RUST : 'white',
                color: showProj ? 'white' : RUST,
                cursor: 'pointer',
                fontSize: '0.85rem',
                fontFamily: 'inherit',
                fontWeight: showProj ? 600 : 400,
              }}
            >
              {showProj ? '正在投影到 PC1 · 点击关闭' : '投影到 PC1（降到 1 维）'}
            </button>
            {showProj && (
              <p style={{ fontSize: 13, color: '#5b6168', margin: '8px 0 0', maxWidth: '52ch' }}>
                灰色虚线 = 各点向 PC1 的正交投影（⊥ PC1，见 18 节）；
                {' '}<span style={{ color: RUST, fontWeight: 600 }}>锈色点</span>
                {' '}= 降维后每个样本在 PC1 上的 1-D 坐标。蓝色原点已淡化。
              </p>
            )}
          </div>
        </div>

      </section>

      {/* ── Canvas ── */}
      <section
        className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}
      >
        <div style={{
          border: '1.5px solid #e6e8ea',
          borderRadius: 8,
          overflow: 'hidden',
          background: '#fff',
          boxShadow: '0 1px 6px rgba(0,0,0,0.06)',
          maxWidth: '100%',
          width: SIZE,
        }}>
          <PCACanvas
            pts={pts} setPts={setPts}
            mx={mx} my={my}
            v1={v1} v2={v2}
            lam1={lam1} lam2={lam2}
            showProj={showProj}
          />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0, textAlign: 'center', maxWidth: '52ch' }}>
          <strong>拖动蓝点</strong>试试 ·
          {' '}<span style={{ color: '#5b6168', fontWeight: 700 }}>⊙</span> = 数据重心（centroid，随点移动）；
          {' '}<span style={{ color: RUST, fontWeight: 700 }}>—— PC1</span>
          {' '}= 最大方差方向，箭头长度 = 标准差 √λ₁；
          {' '}<span style={{ color: IKB, fontWeight: 700 }}>—— PC2</span>
          {' '}⊥ PC1，长度 = √λ₂。主轴始终穿过重心。
        </p>
      </section>

      {/* ── Readouts ── */}
      <section className="readouts">
        <div style={{
          display: 'flex', gap: '2rem', flexWrap: 'wrap',
          alignItems: 'flex-start', justifyContent: 'center',
        }}>

          {/* Covariance matrix C */}
          <div className="matrix">
            <span className="matrix-name">C</span>
            <span className="bracket">[</span>
            <span className="matrix-rows">
              <span>
                <span style={{ color: RUST, fontWeight: 700 }}>{f2(cxx)}</span>
                {'  '}
                <span style={{ color: '#555' }}>{f2(cxy)}</span>
              </span>
              <span>
                <span style={{ color: '#555' }}>{f2(cxy)}</span>
                {'  '}
                <span style={{ color: IKB, fontWeight: 700 }}>{f2(cyy)}</span>
              </span>
            </span>
            <span className="bracket">]</span>
          </div>

          {/* Eigenvalues */}
          <div style={{ fontFamily: 'monospace', fontSize: '0.95rem', lineHeight: 2.2 }}>
            <div>λ₁ = <span style={{ color: RUST, fontWeight: 700 }}>{f2(lam1)}</span></div>
            <div>λ₂ = <span style={{ color: IKB,  fontWeight: 700 }}>{f2(lam2)}</span></div>
          </div>

          {/* Explained variance ratio */}
          <div style={{
            fontFamily: 'monospace', fontSize: '0.92rem', lineHeight: 1.9,
            background: '#fafbfc', border: '1px solid #e4e6e9',
            padding: '6px 16px', borderRadius: 4,
          }}>
            <div style={{ fontSize: '0.78rem', color: '#666', marginBottom: 2, fontFamily: 'var(--sans)' }}>
              explained variance ratio
            </div>
            <div>
              PC1:{' '}
              <span style={{ color: RUST, fontWeight: 700 }}>
                {(explained * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              PC2:{' '}
              <span style={{ color: IKB, fontWeight: 700 }}>
                {((1 - explained) * 100).toFixed(1)}%
              </span>
            </div>
          </div>

        </div>

        <p style={{
          textAlign: 'center', color: '#888',
          fontSize: '0.82rem', marginTop: '1rem',
        }}>
          C 对角线 = 各原始轴的 variance（<span style={{ color: RUST }}>cₓₓ</span>，
          <span style={{ color: IKB }}>c_yy</span>）；
          off-diagonal cₓᵧ = covariance（相关程度）；
          eigenvalue = 沿 PC 方向的 variance。
        </p>
      </section>

      {/* ── Insight box ── */}
      <section className="verdict verdict--eq">
        <p>
          <strong>PCA 选出「主要矛盾」。</strong>{' '}
          当数据强相关时（cₓᵧ 大），λ₁ ≫ λ₂——
          PC1 一条轴能解释绝大部分 variance，降维损失极小。
          各向同性圆里 λ₁ ≈ λ₂，没有哪个方向特别值得保留，降维就有实质损失。
          投影后的 1-D 坐标（锈色点在 PC1 上的位置）就是降维的结果：
          你扔掉了 PC2，只留下每个点「在最重要方向上走了多远」。
          数学上这就是 section 18 的正交投影——降维 = 投影 + 舍弃余维度。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            PCA 是<strong>把 768 维 embedding 拍到 2-D 可视化</strong>的标准工具——
            在 NLP 里最常见的用法就是：
            用前两个 principal component 画散点图，看语义聚类。
            「表征里真正有信息的只有少数几个方向」这一直觉，
            也是 <strong>低秩近似（22 节）</strong>和 <strong>LoRA 微调</strong>的基础：
            如果梯度更新矩阵本质上是低秩的，就只需要更新那几个 principal component 方向。
          </p>
          <p>
            数学上，PCA = 中心化数据的 <strong>SVD（21 节）</strong>：
            <code>X_c = U Σ Vᵀ</code>，Vᵀ 的行就是 PC 方向，Σ²/n 就是 eigenvalue。
            covariance matrix 是<strong>对称正半定阵</strong>——正是
            <strong>谱分解（16 节）</strong>能保证实数 eigenvalue 和正交 eigenvector 的那类矩阵。
            三件事（谱分解、SVD、PCA）在这里汇成一条线。
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：从中心化到降维，逐步骤</h2>
        <CodeBlock code={SNIPPET} language="python" title="pca.py" />
      </section>

      {/* ── Pager ── */}
      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一章</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : <span />}
        {next ? (
          <Link
            className="pager-link next"
            to={next.status === 'live' ? `/ch/${next.slug}` : '/'}
          >
            <span className="pager-dir">下一章 →</span>
            <span className="pager-title">
              {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
            </span>
          </Link>
        ) : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
