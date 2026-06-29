import { useState } from 'react'
import { Link } from 'react-router-dom'
import { apply, multiply, nearlyEqual, type Mat2, type Vec2 } from '../linalg'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Canvas constants ──────────────────────────────────────────────────────────
const SIZE = 360
const RANGE = 3
const UNIT = SIZE / 2 / RANGE   // 60 px per math unit
const CX = SIZE / 2
const CY = SIZE / 2

const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT

const RUST = '#c75b39'
const IKB  = '#002fa7'
const GRID = '#e6e8ea'

const fmt = (n: number): string => {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── Presets ───────────────────────────────────────────────────────────────────
type Preset = { name: string; a: number; b: number; d: number }
const PRESETS: Preset[] = [
  { name: '各向同性（圆）',       a: 1, b: 0,  d: 1   },
  { name: '主轴拉伸',            a: 2, b: 0,  d: 0.5 },
  { name: '含剪切的对称阵',      a: 2, b: 1,  d: 2   },
  { name: '一个负 λ（轴翻转）',  a: 1, b: 0,  d: -1  },
]

// ── Eigen computation for symmetric [[a,b],[b,d]] ────────────────────────────
// λ = (a+d)/2 ± sqrt(((a−d)/2)² + b²)  — always real
// Eigenvectors: for λ₁, direction ∝ [b, λ₁−a]  (b≠0);  rotate 90° for λ₂
interface EigenResult {
  lam1: number
  lam2: number
  v1: Vec2    // unit eigenvector for lam1
  v2: Vec2    // unit eigenvector for lam2 (v2 ⊥ v1)
}

function computeEigen(a: number, b: number, d: number): EigenResult {
  const mid  = (a + d) / 2
  const disc = Math.sqrt(((a - d) / 2) ** 2 + b * b)
  const lam1 = mid + disc
  const lam2 = mid - disc

  let v1: Vec2, v2: Vec2
  if (Math.abs(b) < 1e-10) {
    // Diagonal matrix — standard-basis eigenvectors, ordered by eigenvalue
    if (a >= d) { v1 = [1, 0]; v2 = [0, 1] }
    else         { v1 = [0, 1]; v2 = [1, 0] }
  } else {
    // [b, λ₁−a] is an eigenvector for λ₁
    const rx  = b
    const ry  = lam1 - a
    const len = Math.hypot(rx, ry)
    v1 = [rx / len, ry / len]
    v2 = [-v1[1], v1[0]]   // 90° rotation → always ⊥ v1
  }

  return { lam1, lam2, v1, v2 }
}

// ── EigenArrow: bidirectional principal-axis arrow ────────────────────────────
function EigenArrow({
  v, color, label,
}: { v: Vec2; color: string; label: string }) {
  const [x, y] = v
  if (Math.hypot(x, y) < 1e-6) return null

  const tipX = toSx(x);  const tipY = toSy(y)
  const orgX = toSx(0);  const orgY = toSy(0)
  const negX = toSx(-x); const negY = toSy(-y)

  // Arrowhead at positive tip (same geometry as MatrixAsTransform BasisArrow)
  const ang = Math.atan2(orgY - tipY, tipX - orgX)
  const ah = 10, aw = 5
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)

  // Label: pushed outward along arrow direction from tip
  const dx = tipX - orgX
  const dy = tipY - orgY
  const dlen = Math.hypot(dx, dy) || 1
  const labelX = tipX + (dx / dlen) * 20
  const labelY = tipY + (dy / dlen) * 20 + 4
  const anchor = dx > 5 ? 'start' : (dx < -5 ? 'end' : 'middle')

  return (
    <g>
      {/* Negative half — dashed, shows full axis */}
      <line
        x1={orgX} y1={orgY} x2={negX} y2={negY}
        stroke={color} strokeWidth={1.5} strokeDasharray="4 3"
        strokeLinecap="round" opacity={0.4}
      />
      {/* Positive half — solid */}
      <line
        x1={orgX} y1={orgY} x2={tipX} y2={tipY}
        stroke={color} strokeWidth={2.5} strokeLinecap="round"
      />
      <polygon
        points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`}
        fill={color}
      />
      <text
        x={labelX} y={labelY}
        fill={color} fontSize={12} fontWeight="bold"
        textAnchor={anchor}
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {label}
      </text>
    </g>
  )
}

// ── SpectralCanvas ────────────────────────────────────────────────────────────
function SpectralCanvas({ a, b, d }: { a: number; b: number; d: number }) {
  const M: Mat2 = [a, b, b, d]
  const { lam1, lam2, v1, v2 } = computeEigen(a, b, d)

  // Ellipse: transform 120 points on the unit circle through M
  const N = 120
  const ellipsePts = Array.from({ length: N }, (_, i) => {
    const theta = (2 * Math.PI * i) / N
    const [ex, ey] = apply(M, [Math.cos(theta), Math.sin(theta)])
    return `${toSx(ex).toFixed(1)},${toSy(ey).toFixed(1)}`
  }).join(' ')

  // Background grid (undeformed reference)
  const gridLines: JSX.Element[] = []
  for (let idx = 0; idx <= RANGE * 2; idx++) {
    const k = idx - RANGE
    const stroke = k === 0 ? '#9aa5b0' : GRID
    const sw = k === 0 ? 1.5 : 1
    gridLines.push(
      <line key={`v${k}`}
        x1={toSx(k)} y1={toSy(-RANGE)} x2={toSx(k)} y2={toSy(RANGE)}
        stroke={stroke} strokeWidth={sw} />,
      <line key={`h${k}`}
        x1={toSx(-RANGE)} y1={toSy(k)} x2={toSx(RANGE)} y2={toSy(k)}
        stroke={stroke} strokeWidth={sw} />,
    )
  }

  // Principal axes scaled to |λ|
  const axis1: Vec2 = [v1[0] * Math.abs(lam1), v1[1] * Math.abs(lam1)]
  const axis2: Vec2 = [v2[0] * Math.abs(lam2), v2[1] * Math.abs(lam2)]

  return (
    <svg
      width={SIZE} height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      style={{ display: 'block' }}
    >
      <defs>
        <clipPath id="spectral-clip">
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
      </defs>

      {/* Grid */}
      <g clipPath="url(#spectral-clip)">{gridLines}</g>

      {/* Unit circle (input reference, dashed) */}
      <circle
        cx={CX} cy={CY} r={UNIT}
        fill="none" stroke="#c4c8cc"
        strokeWidth={1} strokeDasharray="4 3"
        clipPath="url(#spectral-clip)"
      />

      {/* Transformed ellipse (M applied to unit circle) */}
      <polygon
        clipPath="url(#spectral-clip)"
        points={ellipsePts}
        fill="rgba(0,47,167,0.07)"
        stroke={IKB}
        strokeWidth={2}
        strokeLinejoin="round"
      />

      {/* Principal-axis arrows */}
      <g clipPath="url(#spectral-clip)">
        <EigenArrow v={axis1} color={RUST} label={`λ₁ = ${fmt(lam1)}`} />
        <EigenArrow v={axis2} color={IKB}  label={`λ₂ = ${fmt(lam2)}`} />
      </g>
    </svg>
  )
}

// ── Compact matrix display ────────────────────────────────────────────────────
function MiniMatrix({
  label, m, col1Color, col2Color,
}: {
  label: string
  m: Mat2
  col1Color?: string
  col2Color?: string
}) {
  const cell = (v: number, color?: string) => (
    <span style={color ? { color, fontWeight: 700 } : {}}>{fmt(v)}</span>
  )
  return (
    <div className="matrix">
      <span className="matrix-name">{label}</span>
      <span className="bracket">[</span>
      <span className="matrix-rows">
        <span>{cell(m[0], col1Color)}{'  '}{cell(m[1], col2Color)}</span>
        <span>{cell(m[2], col1Color)}{'  '}{cell(m[3], col2Color)}</span>
      </span>
      <span className="bracket">]</span>
    </div>
  )
}

// ── Python code snippet ───────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

M = np.array([[2., 1.],
              [1., 2.]])   # 对称阵：M == M.T

# eigh 专为对称/Hermitian 矩阵设计，保证实数特征值，升序返回
lam, Q = np.linalg.eigh(M)
print("λ  :", lam)           # [1. 3.]
print("Q  :\\n", Q)          # 列 = 单位特征向量

# 正交性验证
print("Q.T @ Q :\\n", Q.T @ Q)   # ≈ 单位阵 — 列互相垂直

# 谱分解：重建 M = Q @ diag(λ) @ Q.T
M_rec = Q @ np.diag(lam) @ Q.T
print("重建误差:", np.abs(M_rec - M).max())  # ≈ 0.0`

// ── Main page component ───────────────────────────────────────────────────────
export function Spectral() {
  const [a, setA] = useState(2)
  const [b, setB] = useState(1)
  const [d, setD] = useState(2)

  // Symmetric matrix M = [[a,b],[b,d]]
  const M: Mat2 = [a, b, b, d]

  const { lam1, lam2, v1, v2 } = computeEigen(a, b, d)

  // Spectral decomposition pieces
  // Q: columns = unit eigenvectors.  Mat2 = [a,b,c,d] means [[a,b],[c,d]]
  // col1 = v1, col2 = v2  →  Q[0]=v1[0], Q[1]=v2[0], Q[2]=v1[1], Q[3]=v2[1]
  const Q: Mat2      = [v1[0], v2[0], v1[1], v2[1]]
  const Lambda: Mat2 = [lam1, 0, 0, lam2]
  const QT: Mat2     = [v1[0], v1[1], v2[0], v2[1]]   // Qᵀ (transpose of Q)

  // Reconstruct: Q Λ Qᵀ should equal M
  const recon = multiply(multiply(Q, Lambda), QT)
  const ok    = nearlyEqual(recon, M)

  const me = findChapter('spectral')!
  const { prev, next } = neighbors('spectral')

  return (
    <article className="page">

      {/* ── Header ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第三部分 · 方阵的秘密
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          对称矩阵与谱分解
          <span className="zh-sub">最温顺的矩阵，隐藏最美的结构</span>
        </h1>
        <p className="lede">
          对称矩阵 <code>M = Mᵀ</code>——对角线两侧的数互相镜像——是线性代数里最「规矩」的一类矩阵。
          它有两个神奇保证：eigenvalue（特征值）永远是<strong>实数</strong>；
          eigenvector（特征向量）永远<strong>两两 orthogonal（垂直）</strong>。
          这让它可以被彻底拆开：<code>M = Q Λ Qᵀ</code>——先旋转到特征轴坐标系，
          按 λ 缩放各轴，再旋转回来。这就是 spectral decomposition（谱分解）。
          调节下面三个滑块，亲眼看单位圆变成椭圆，主轴始终与特征向量重合。
        </p>
      </header>

      {/* ── Sliders + Presets ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">对称矩阵参数</span>
            <code style={{ color: '#666', fontSize: '0.85rem' }}>[[a, b], [b, d]]</code>
          </div>
          <label className="slider-row">
            <span style={{ width: '1.8rem', display: 'inline-block', color: RUST, fontWeight: 700 }}>a</span>
            <input
              type="range" min={-3} max={3} step={0.1}
              value={a} onChange={(e) => setA(Number(e.target.value))}
            />
            <span className="param-val">{fmt(a)}</span>
          </label>
          <label className="slider-row">
            <span style={{ width: '1.8rem', display: 'inline-block', fontWeight: 700, color: '#555' }}>b</span>
            <input
              type="range" min={-2} max={2} step={0.1}
              value={b} onChange={(e) => setB(Number(e.target.value))}
            />
            <span className="param-val">{fmt(b)}</span>
          </label>
          <label className="slider-row">
            <span style={{ width: '1.8rem', display: 'inline-block', color: IKB, fontWeight: 700 }}>d</span>
            <input
              type="range" min={-3} max={3} step={0.1}
              value={d} onChange={(e) => setD(Number(e.target.value))}
            />
            <span className="param-val">{fmt(d)}</span>
          </label>
        </div>

        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>点击快速切换</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => { setA(p.a); setB(p.b); setD(p.d) }}
                style={{
                  padding: '0.35rem 0.9rem',
                  border: `1.5px solid ${IKB}`,
                  borderRadius: 4,
                  background: 'white',
                  color: IKB,
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontFamily: 'inherit',
                }}
              >
                {p.name}
              </button>
            ))}
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
        }}>
          <SpectralCanvas a={a} b={b} d={d} />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0, textAlign: 'center' }}>
          虚线圆 = 单位圆（输入）；实线椭圆 = M 变换后的像；
          {' '}<span style={{ color: RUST, fontWeight: 700 }}>● 轴₁</span> 和
          {' '}<span style={{ color: IKB, fontWeight: 700 }}>● 轴₂</span>
          {' '}= 两个特征向量主轴，始终互相垂直
        </p>
      </section>

      {/* ── Readouts ── */}
      <section className="readouts">
        {/* Row 1: M and eigenvalues */}
        <div style={{
          display: 'flex', gap: '2.5rem', flexWrap: 'wrap',
          alignItems: 'flex-start', justifyContent: 'center',
        }}>
          <MiniMatrix label="M" m={M} />
          <div style={{ fontFamily: 'monospace', fontSize: '0.95rem', lineHeight: 2.2 }}>
            <div>λ₁ = <span style={{ color: RUST, fontWeight: 700 }}>{fmt(lam1)}</span></div>
            <div>λ₂ = <span style={{ color: IKB,  fontWeight: 700 }}>{fmt(lam2)}</span></div>
          </div>
        </div>

        {/* Row 2: Q Λ Qᵀ = M */}
        <div style={{
          display: 'flex', gap: '0.6rem', flexWrap: 'wrap',
          alignItems: 'center', justifyContent: 'center',
          marginTop: '1.4rem',
        }}>
          <MiniMatrix label="Q" m={Q} col1Color={RUST} col2Color={IKB} />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>·</span>
          <MiniMatrix label="Λ" m={Lambda} />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>·</span>
          <MiniMatrix label="Qᵀ" m={QT} />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>=</span>
          <MiniMatrix label="重建" m={recon} />
          {ok && (
            <span style={{ color: '#1a7a3a', fontSize: '1.5rem', fontWeight: 700 }}>✓</span>
          )}
        </div>
        <p style={{
          textAlign: 'center', color: '#888',
          fontSize: '0.82rem', marginTop: '0.5rem',
        }}>
          Q 的列 = 单位特征向量（orthogonal）；Λ = 对角特征值矩阵；Q Λ Qᵀ 精确还原 M
        </p>
      </section>

      {/* ── Insight box ── */}
      <section className="verdict verdict--eq">
        <p>
          <strong>三个参数，三件事。</strong>{' '}
          a、d 控制沿坐标轴方向的本征拉伸，b 把两根主轴斜向耦合。
          λ 全正 → 椭圆是真正的放大；有负 λ → 那根主轴翻转，椭圆在对应方向被「折回」。
          λ₁ = λ₂ → 各向同性，圆映射到圆。
          Q 的两列始终垂直——这是 symmetric 对你的承诺，无论 a、b、d 怎么取。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            对称矩阵无处不在：<strong>covariance matrix（协方差矩阵）</strong>{' '}
            <code>XᵀX</code>、<strong>Gram matrix</strong> <code>XXᵀ</code>，以及注意力里
            <code> QKᵀ</code> 的对称化版本都是对称的，因此都有实数 eigenvalue 和
            orthogonal eigenvector。
          </p>
          <p>
            Spectral decomposition = <strong>PCA 的引擎</strong>（第 22–23 节）：
            主成分分析就是在求协方差矩阵的特征向量，把数据投影到方差最大的方向上。
            它也是理解 <strong>SVD</strong>（第 21 节）的台阶——SVD 把谱分解推广到任意非方阵：
            <code> M = U Σ Vᵀ</code>，U、V 的列都 orthogonal，Σ 是奇异值对角阵。
            看懂这一页，SVD 就只是谱分解的自然延伸。
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：用 NumPy 做 spectral decomposition</h2>
        <CodeBlock code={SNIPPET} language="python" title="spectral.py" />
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
        ) : (
          <span />
        )}
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
        ) : (
          <span />
        )}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
