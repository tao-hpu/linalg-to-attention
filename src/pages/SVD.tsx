import { useState } from 'react'
import { apply, multiply, nearlyEqual, type Mat2, type Vec2 } from '../linalg'
import { ChapterShell } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'

// ── Canvas constants ──────────────────────────────────────────────────────────
const SIZE  = 200
const RANGE = 3
const UNIT  = SIZE / 2 / RANGE   // px per math unit
const CX    = SIZE / 2
const CY    = SIZE / 2

const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT

const RUST = '#c75b39'
const IKB  = '#002fa7'

const IDENTITY: Mat2 = [1, 0, 0, 1]

const fmt = (n: number): string => {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── SVD (2×2, no library) ─────────────────────────────────────────────────────
// Mat2 row-major: [a,b,c,d] = [[a,b],[c,d]]
// S = MᵀM is always symmetric PSD → use closed-form symmetric eigen (from Spectral.tsx)
// σᵢ = sqrt(λᵢ(S)) ≥ 0, sorted desc
// Right singular vectors: eigenvectors of S → columns of V
// Left singular vectors:  uᵢ = M vᵢ / σᵢ  (fallback when σᵢ ≈ 0)

interface SVDResult {
  sigma1: number
  sigma2: number
  v1: Vec2
  v2: Vec2
  u1: Vec2
  u2: Vec2
  U:   Mat2   // columns u1, u2
  Sig: Mat2   // diag(σ1, σ2)
  Vt:  Mat2   // rows v1ᵀ, v2ᵀ
  V:   Mat2   // columns v1, v2
}

function computeSVD(M: Mat2): SVDResult {
  const [a, b, c, d] = M

  // S = MᵀM
  const s00 = a * a + c * c
  const s01 = a * b + c * d
  const s11 = b * b + d * d

  // Eigenvalues of symmetric [[s00,s01],[s01,s11]]
  const mid  = (s00 + s11) / 2
  const disc = Math.sqrt(Math.max(0, ((s00 - s11) / 2) ** 2 + s01 * s01))
  const lam1 = mid + disc            // λ₁ ≥ λ₂
  const lam2 = Math.max(0, mid - disc)

  const sigma1 = Math.sqrt(lam1)
  const sigma2 = Math.sqrt(lam2)

  // Eigenvectors of S → right singular vectors
  let v1: Vec2, v2: Vec2
  if (Math.abs(s01) < 1e-10) {
    // Diagonal S → standard basis
    if (s00 >= s11) { v1 = [1, 0]; v2 = [0, 1] }
    else             { v1 = [0, 1]; v2 = [1, 0] }
  } else {
    const rx  = s01
    const ry  = lam1 - s00
    const len = Math.hypot(rx, ry)
    v1 = [rx / len, ry / len]
    v2 = [-v1[1], v1[0]]   // 90° rotation → always ⊥ v1
  }

  // Left singular vectors: uᵢ = M vᵢ / σᵢ
  const mv1 = apply(M, v1)
  const u1: Vec2 = sigma1 > 1e-10
    ? [mv1[0] / sigma1, mv1[1] / sigma1]
    : [1, 0]

  let u2: Vec2
  if (sigma2 > 1e-10) {
    const mv2 = apply(M, v2)
    u2 = [mv2[0] / sigma2, mv2[1] / sigma2]
  } else {
    u2 = [-u1[1], u1[0]]   // fallback: ⊥ u1
  }

  const U:   Mat2 = [u1[0], u2[0], u1[1], u2[1]]
  const Sig: Mat2 = [sigma1, 0, 0, sigma2]
  const Vt:  Mat2 = [v1[0], v1[1], v2[0], v2[1]]
  const V:   Mat2 = [v1[0], v2[0], v1[1], v2[1]]

  return { sigma1, sigma2, v1, v2, u1, u2, U, Sig, Vt, V }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
// Apply T to 120 points on the unit circle → SVG polygon points
function ellipsePoints(T: Mat2, N = 120): string {
  return Array.from({ length: N }, (_, i) => {
    const theta = (2 * Math.PI * i) / N
    const [ex, ey] = apply(T, [Math.cos(theta), Math.sin(theta)])
    return `${toSx(ex).toFixed(1)},${toSy(ey).toFixed(1)}`
  }).join(' ')
}

// Cap vector length at maxLen for display (so arrows don't dominate the canvas)
function cap(v: Vec2, maxLen = 2.5): Vec2 {
  const len = Math.hypot(v[0], v[1])
  if (len <= maxLen) return v
  return [v[0] / len * maxLen, v[1] / len * maxLen] as Vec2
}

// ── Arrow (origin → v) with arrowhead and label ───────────────────────────────
function SvdArrow({ v, color, label }: { v: Vec2; color: string; label: string }) {
  const [x, y] = v
  if (Math.hypot(x, y) < 1e-6) return null
  const tipX = toSx(x);  const tipY = toSy(y)
  const orgX = toSx(0);  const orgY = toSy(0)
  const ang  = Math.atan2(orgY - tipY, tipX - orgX)
  const ah = 8, aw = 4
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)
  const dx    = tipX - orgX
  const dy    = tipY - orgY
  const dlen  = Math.hypot(dx, dy) || 1
  const labelX  = tipX + (dx / dlen) * 15
  const labelY  = tipY + (dy / dlen) * 15 + 4
  const anchor  = dx > 3 ? 'start' : (dx < -3 ? 'end' : 'middle')
  return (
    <g>
      <line x1={orgX} y1={orgY} x2={tipX} y2={tipY}
        stroke={color} strokeWidth={2} strokeLinecap="round" />
      <polygon points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`} fill={color} />
      <text x={labelX} y={labelY} fill={color} fontSize={10} fontWeight="bold"
        textAnchor={anchor} style={{ pointerEvents: 'none', userSelect: 'none' }}>
        {label}
      </text>
    </g>
  )
}

// ── Stage panel ───────────────────────────────────────────────────────────────
type ArrowSpec = { v: Vec2; color: string; label: string }

function StagePanel({ panelId, T, label, sublabel, arrows = [] }: {
  panelId: string
  T: Mat2
  label: string
  sublabel: string
  arrows?: ArrowSpec[]
}) {
  const pts    = ellipsePoints(T)
  const clipId = `svd-clip-${panelId}`

  const gridLines: JSX.Element[] = []
  for (let k = -RANGE; k <= RANGE; k++) {
    const stroke = k === 0 ? '#9aa5b0' : '#e6e8ea'
    const sw     = k === 0 ? 1.5 : 1
    gridLines.push(
      <line key={`v${k}`} x1={toSx(k)} y1={toSy(-RANGE)} x2={toSx(k)} y2={toSy(RANGE)}
        stroke={stroke} strokeWidth={sw} />,
      <line key={`h${k}`} x1={toSx(-RANGE)} y1={toSy(k)} x2={toSx(RANGE)} y2={toSy(k)}
        stroke={stroke} strokeWidth={sw} />,
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
      <div style={{
        border: '1.5px solid #e6e8ea', borderRadius: 8,
        overflow: 'hidden', background: '#fff',
        boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
      }}>
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}
          style={{ display: 'block' }}>
          <defs>
            <clipPath id={clipId}>
              <rect x={0} y={0} width={SIZE} height={SIZE} />
            </clipPath>
          </defs>
          <g clipPath={`url(#${clipId})`}>{gridLines}</g>
          {/* Unit circle reference (dashed) */}
          <circle cx={CX} cy={CY} r={UNIT}
            fill="none" stroke="#d0d4d8"
            strokeWidth={1} strokeDasharray="4 3"
            clipPath={`url(#${clipId})`} />
          {/* Transformed shape */}
          <polygon clipPath={`url(#${clipId})`} points={pts}
            fill="rgba(0,47,167,0.08)" stroke={IKB}
            strokeWidth={2} strokeLinejoin="round" />
          {/* Singular-vector arrows */}
          <g clipPath={`url(#${clipId})`}>
            {arrows.map((arr) => (
              <SvdArrow key={arr.label} v={arr.v} color={arr.color} label={arr.label} />
            ))}
          </g>
        </svg>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#1b1f24' }}>{label}</div>
        <div style={{ fontSize: '0.75rem', color: '#888', fontFamily: 'monospace' }}>{sublabel}</div>
      </div>
    </div>
  )
}

// ── Matrix readout ────────────────────────────────────────────────────────────
function MiniMatrix({ label, m, c1, c2 }: {
  label: string; m: Mat2; c1?: string; c2?: string
}) {
  const cell = (v: number, col?: string) => (
    <span style={col ? { color: col, fontWeight: 700 } : {}}>{fmt(v)}</span>
  )
  return (
    <div className="matrix">
      <span className="matrix-name">{label}</span>
      <span className="bracket">[</span>
      <span className="matrix-rows">
        <span>{cell(m[0], c1)}{'  '}{cell(m[1], c2)}</span>
        <span>{cell(m[2], c1)}{'  '}{cell(m[3], c2)}</span>
      </span>
      <span className="bracket">]</span>
    </div>
  )
}

// ── Presets ───────────────────────────────────────────────────────────────────
type Preset = { name: string; a: number; b: number; c: number; d: number }
const PRESETS: Preset[] = [
  { name: '对角拉伸',        a: 3,   b: 0,   c: 0,   d: 2   },
  { name: '对称含剪切',      a: 2,   b: 1,   c: 1,   d: 2   },
  { name: '一般切变',        a: 1,   b: 1,   c: 0,   d: 1   },
  { name: '几乎降秩 σ₂≈0',  a: 2,   b: 1,   c: 4,   d: 2.1 },
]

// ── Code snippet ──────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

M = np.array([[2., 1.],
              [1., 2.]])

U, S, Vt = np.linalg.svd(M)
# S  奇异值按降序排列  →  [3. 1.]
# U  左奇异向量（列 = uᵢ），输出方向
# Vt 右奇异向量的转置（行 = vᵢᵀ），输入方向
# 椭圆半轴长 = 奇异值 S

# 重建 M = U @ diag(S) @ Vt
M_rec = U @ np.diag(S) @ Vt
print("重建误差:", np.abs(M_rec - M).max())  # ≈ 0.0

# 秩-1 近似（只保留最大奇异值）
M_rank1 = S[0] * np.outer(U[:, 0], Vt[0, :])

# Eckart–Young：丢掉的那部分，谱范数恰好等于第一个被丢掉的奇异值
print("秩-1 误差:", np.linalg.norm(M_rank1 - M, 2))  # = σ₂ = 1.0
# 换成 np.abs(...).max() 得到的是逐元素最大值 0.5，不是范数，别混用`

// ── Main component ────────────────────────────────────────────────────────────
export function SVD() {
  const [a, setA] = useState(2)
  const [b, setB] = useState(1)
  const [c, setC] = useState(1)
  const [d, setD] = useState(2)

  const M: Mat2 = [a, b, c, d]
  const { sigma1, sigma2, v1, v2, u1, u2, U, Sig, Vt, V } = computeSVD(M)

  // Cumulative stage transformations
  const sigVt = multiply(Sig, Vt)          // Σ·Vᵀ
  const recon = multiply(U, sigVt)         // U·Σ·Vᵀ ≈ M
  const ok    = nearlyEqual(recon, M)

  // Arrows per stage
  const arrows1: ArrowSpec[] = [
    { v: v1, color: RUST, label: 'v₁' },
    { v: v2, color: IKB,  label: 'v₂' },
  ]
  const arrows2: ArrowSpec[] = [
    { v: [1, 0] as Vec2, color: RUST, label: 'e₁' },
    { v: [0, 1] as Vec2, color: IKB,  label: 'e₂' },
  ]
  const arrows3: ArrowSpec[] = [
    { v: cap([sigma1, 0] as Vec2), color: RUST, label: 'σ₁' },
    { v: cap([0, sigma2] as Vec2), color: IKB,  label: 'σ₂' },
  ]
  const arrows4: ArrowSpec[] = [
    { v: cap([u1[0] * sigma1, u1[1] * sigma1] as Vec2), color: RUST, label: 'u₁' },
    { v: cap([u2[0] * sigma2, u2[1] * sigma2] as Vec2), color: IKB,  label: 'u₂' },
  ]

  return (
      <ChapterShell
        slug="svd"
        part="第五部分 · 降维：抓住主要矛盾"
        sub="旋转 → 拉伸 → 旋转：任意矩阵的万能拆解"
        lede={
          <>
        无论什么矩阵——方阵、长方阵、切变、投影——都能写成
        {' '}<code>M = U Σ Vᵀ</code>：
        先用 <code>Vᵀ</code> 旋转输入轴，再用 <code>Σ</code> 按
        singular value（奇异值）沿各轴拉伸，最后用 <code>U</code> 旋转到输出方向。
        几何上：M 总是把单位圆变成一个<strong>椭圆</strong>，椭圆的两个半轴长
        正是 <strong>σ₁ ≥ σ₂ ≥ 0</strong>。
        拖动滑块，看四步分解如何把圆变成椭圆。
          </>
        }
      >


      {/* ── Controls ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">矩阵参数</span>
            <code style={{ color: '#666', fontSize: '0.85rem' }}>M = [[a, b], [c, d]]</code>
          </div>
          <label className="slider-row">
            <span style={{ width: '1.4rem', display: 'inline-block', color: RUST, fontWeight: 700 }}>a</span>
            <input type="range" min={-3} max={3} step={0.1}
              value={a} onChange={(e) => setA(Number(e.target.value))} />
            <span className="param-val">{fmt(a)}</span>
          </label>
          <label className="slider-row">
            <span style={{ width: '1.4rem', display: 'inline-block', fontWeight: 700, color: '#555' }}>b</span>
            <input type="range" min={-3} max={3} step={0.1}
              value={b} onChange={(e) => setB(Number(e.target.value))} />
            <span className="param-val">{fmt(b)}</span>
          </label>
          <label className="slider-row">
            <span style={{ width: '1.4rem', display: 'inline-block', fontWeight: 700, color: '#555' }}>c</span>
            <input type="range" min={-3} max={3} step={0.1}
              value={c} onChange={(e) => setC(Number(e.target.value))} />
            <span className="param-val">{fmt(c)}</span>
          </label>
          <label className="slider-row">
            <span style={{ width: '1.4rem', display: 'inline-block', color: IKB, fontWeight: 700 }}>d</span>
            <input type="range" min={-3} max={3} step={0.1}
              value={d} onChange={(e) => setD(Number(e.target.value))} />
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
              <button key={p.name}
                onClick={() => { setA(p.a); setB(p.b); setC(p.c); setD(p.d) }}
                style={{
                  padding: '0.35rem 0.9rem',
                  border: `1.5px solid ${IKB}`,
                  borderRadius: 4, background: 'white',
                  color: IKB, cursor: 'pointer',
                  fontSize: '0.85rem', fontFamily: 'inherit',
                }}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Stage: 4 panels ── */}
      <section className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
        <div style={{
          display: 'flex', alignItems: 'flex-start',
          flexWrap: 'wrap', gap: '0.3rem', justifyContent: 'center',
        }}>
          <StagePanel panelId="p1" T={IDENTITY}
            label="① 输入：单位圆" sublabel="I"
            arrows={arrows1} />
          <div style={{ alignSelf: 'center', color: '#bbb', fontSize: '1.3rem', padding: '0 0.15rem' }}>→</div>
          <StagePanel panelId="p2" T={Vt}
            label="② Vᵀ 旋转" sublabel="Vᵀ"
            arrows={arrows2} />
          <div style={{ alignSelf: 'center', color: '#bbb', fontSize: '1.3rem', padding: '0 0.15rem' }}>→</div>
          <StagePanel panelId="p3" T={sigVt}
            label="③ Σ 拉伸" sublabel="Σ·Vᵀ"
            arrows={arrows3} />
          <div style={{ alignSelf: 'center', color: '#bbb', fontSize: '1.3rem', padding: '0 0.15rem' }}>→</div>
          <StagePanel panelId="p4" T={M}
            label="④ U 旋转 = M" sublabel="U·Σ·Vᵀ"
            arrows={arrows4} />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0, textAlign: 'center' }}>
          虚线圆 = 单位圆参考；实线多边形 = 各步累积变换结果；
          {' '}<span style={{ color: RUST, fontWeight: 700 }}>■</span> v₁/u₁（σ₁ 方向）·
          {' '}<span style={{ color: IKB,  fontWeight: 700 }}>■</span> v₂/u₂（σ₂ 方向）
        </p>
      </section>

      {/* ── Readouts ── */}
      <section className="readouts">
        {/* σ values + M */}
        <div style={{
          display: 'flex', gap: '2rem', flexWrap: 'wrap',
          alignItems: 'flex-start', justifyContent: 'center',
        }}>
          <div style={{ fontFamily: 'monospace', fontSize: '0.95rem', lineHeight: 2.2 }}>
            <div>σ₁ = <span style={{ color: RUST, fontWeight: 700 }}>{fmt(sigma1)}</span></div>
            <div>σ₂ = <span style={{ color: IKB,  fontWeight: 700 }}>{fmt(sigma2)}</span></div>
          </div>
          <MiniMatrix label="M" m={M} />
        </div>

        {/* U · Σ · Vᵀ = recon */}
        <div style={{
          display: 'flex', gap: '0.6rem', flexWrap: 'wrap',
          alignItems: 'center', justifyContent: 'center',
          marginTop: '1.4rem',
        }}>
          <MiniMatrix label="U"  m={U}   c1={RUST} c2={IKB} />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>·</span>
          <MiniMatrix label="Σ"  m={Sig} />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>·</span>
          <MiniMatrix label="Vᵀ" m={Vt}  />
          <span style={{ fontSize: '1.3rem', color: '#777' }}>=</span>
          <MiniMatrix label="重建" m={recon} />
          {ok && <span style={{ color: '#1a7a3a', fontSize: '1.5rem', fontWeight: 700 }}>✓</span>}
        </div>
        <p style={{
          textAlign: 'center', color: '#888',
          fontSize: '0.82rem', marginTop: '0.5rem',
        }}>
          U、V 的列均正交（orthogonal）；Σ 对角；U Σ Vᵀ 精确还原 M
        </p>

        {/* V for reference */}
        <div style={{
          display: 'flex', gap: '1.5rem', flexWrap: 'wrap',
          alignItems: 'center', justifyContent: 'center',
          marginTop: '0.8rem',
        }}>
          <MiniMatrix label="V" m={V} c1={RUST} c2={IKB} />
          <span style={{ fontSize: '0.82rem', color: '#888' }}>
            V 的列 = right singular vector（v₁, v₂）= 输入空间的奇异轴
          </span>
        </div>
      </section>

      {/* ── Insight ── */}
      <section className="verdict verdict--eq">
        <p>
          <strong>三个矩阵，三件事。</strong>
          Vᵀ 选定输入空间里哪两个方向将被拉伸（right singular vector v₁, v₂）；
          Σ 决定每个方向的增益（σ₁ ≥ σ₂ ≥ 0，永远非负且降序）；
          U 把拉伸结果送到输出空间的最终朝向（left singular vector u₁, u₂）。
          验证：对角阵 [[3,0],[0,2]] → σ = 3, 2，U = V = I；
          对称阵 [[2,1],[1,2]] → σ = 3, 1，U = V（即特征向量矩阵）；
          几乎降秩的矩阵 [[2,1],[4,2.1]] → σ₂ ≈ 0.04，椭圆被压成一根细长条，
          离「压成一条线」只差一点点——把 d 从 2.1 拨到 2.0，σ₂ 就精确归零了。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            SVD 是「万能分解」：任意权重矩阵都能拆成旋转-拉伸-旋转，
            <strong>奇异值大小 = 每个方向上的「增益」</strong>。
            最大的几个奇异值抓住矩阵的主要作用——这正是
            {' '}<strong>22 低秩近似 / LoRA</strong> 和 <strong>23 PCA</strong> 的全部底气：
            扔掉小奇异值，矩阵的行为几乎不变。
          </p>
          <p>
            σ → 0 的方向就是第 16 节谱分解里「快被压扁的维度」，也是第 13 节「秩」的直接体现。
            LoRA 的核心：不直接训练整个权重矩阵 W，而是训练
            {' '}<code>W + AB</code>（A、B 低秩），利用的正是
            W 的奇异值分布集中在少数几个大奇异值上这一事实。
          </p>
        </div>
      </section>

      {/* ── Code ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：用 NumPy 计算 SVD</h2>
        <CodeBlock code={SNIPPET} language="python" title="svd.py" />
      </section>

      {/* ── Pager ── */}
      </ChapterShell>
  )
}
