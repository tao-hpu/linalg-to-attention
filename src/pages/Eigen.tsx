import { useState, useRef } from 'react'
import { apply, type Mat2, type Vec2 } from '../linalg'
import { ChRef } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'

// ── Canvas constants (same coordinate system as MatrixAsTransform) ────────────
const SIZE = 360
const RANGE = 3
const UNIT = SIZE / 2 / RANGE   // 60 px per math unit
const CX = SIZE / 2
const CY = SIZE / 2

const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT
const toMx = (px: number) => (px - CX) / UNIT
const toMy = (py: number) => (CY - py) / UNIT

// Keep probe inside canvas with a small margin
const clampVec = (v: number) => Math.max(-(RANGE - 0.3), Math.min(RANGE - 0.3, v))

// ── Colors ────────────────────────────────────────────────────────────────────
const RUST = '#c75b39'           // probe vector v
const IKB = '#002fa7'            // transformed vector Mv
const EIGEN1_COLOR = '#b8860b'   // dark goldenrod — λ₁ guide line
const EIGEN2_COLOR = '#6a994e'   // muted green    — λ₂ guide line
const GRID_COLOR = '#e6e8ea'

// ── Inline 2×2 eigen math (closed form via trace / det) ──────────────────────
// λ = t/2 ± √((t/2)² − D),  t = trace = a+d,  D = det = ad−bc
// eigenvector for real λ: solve (M−λI)v=0 → v=[b, λ−a] or [λ−d, c]

type EigenResult =
  | { kind: 'complex' }
  | { kind: 'real'; λ1: number; λ2: number; e1: Vec2; e2: Vec2; repeated: boolean }

function normalize(v: Vec2): Vec2 {
  const len = Math.hypot(v[0], v[1])
  return len < 1e-9 ? [1, 0] : [v[0] / len, v[1] / len]
}

function eigenvecFor(M: Mat2, λ: number): Vec2 {
  const [a, b, c, d] = M
  // Try first row of (M − λI): v = [b, λ−a]
  const v1: Vec2 = [b, λ - a]
  if (Math.hypot(v1[0], v1[1]) > 1e-9) return normalize(v1)
  // Fallback to second row: v = [λ−d, c]
  const v2: Vec2 = [λ - d, c]
  if (Math.hypot(v2[0], v2[1]) > 1e-9) return normalize(v2)
  return [1, 0]
}

export function computeEigen(M: Mat2): EigenResult {
  const [a, b, c, d] = M
  const t = a + d               // trace  = λ₁ + λ₂
  const D = a * d - b * c       // det    = λ₁ · λ₂
  const halfT = t / 2
  const disc = halfT * halfT - D
  if (disc < -1e-9) return { kind: 'complex' }
  const sqrtDisc = Math.sqrt(Math.max(0, disc))
  const λ1 = halfT + sqrtDisc
  const λ2 = halfT - sqrtDisc
  const repeated = sqrtDisc < 1e-9
  return {
    kind: 'real',
    λ1, λ2,
    e1: eigenvecFor(M, λ1),
    e2: eigenvecFor(M, λ2),
    repeated,
  }
}

// Angle in degrees between two vectors (0–90, treats parallel/anti-parallel equally)
function angleDeg(u: Vec2, v: Vec2): number {
  const lu = Math.hypot(u[0], u[1])
  const lv = Math.hypot(v[0], v[1])
  if (lu < 1e-9 || lv < 1e-9) return 90
  const cosAbs = Math.abs((u[0] * v[0] + u[1] * v[1]) / (lu * lv))
  return (Math.acos(Math.min(1, cosAbs)) * 180) / Math.PI
}

// Signed eigenvalue estimate at probe direction: λ ≈ (v · Mv) / |v|²
function eigenScalar(probe: Vec2, Mv: Vec2): number {
  const len2 = probe[0] * probe[0] + probe[1] * probe[1]
  if (len2 < 1e-9) return 0
  return (probe[0] * Mv[0] + probe[1] * Mv[1]) / len2
}

// ── Threshold for "parallel enough" detection ─────────────────────────────────
const PARALLEL_DEG = 8

// ── SVG sub-components ────────────────────────────────────────────────────────

function GridLines() {
  return (
    <>
      {Array.from({ length: RANGE * 2 + 1 }, (_, i) => i - RANGE).flatMap(k => {
        const isAxis = k === 0
        const stroke = isAxis ? '#9aa5b0' : GRID_COLOR
        const sw = isAxis ? 1.5 : 1
        return [
          <line key={`v${k}`}
            x1={toSx(k)} y1={toSy(-RANGE)} x2={toSx(k)} y2={toSy(RANGE)}
            stroke={stroke} strokeWidth={sw}
          />,
          <line key={`h${k}`}
            x1={toSx(-RANGE)} y1={toSy(k)} x2={toSx(RANGE)} y2={toSy(k)}
            stroke={stroke} strokeWidth={sw}
          />,
        ]
      })}
    </>
  )
}

function Arrow({ tip, color, width = 2.5 }: { tip: Vec2; color: string; width?: number }) {
  const [x, y] = tip
  if (Math.hypot(x, y) < 1e-6) return null
  const tx = toSx(x)
  const ty = toSy(y)
  const ox = toSx(0)
  const oy = toSy(0)
  const ang = Math.atan2(oy - ty, tx - ox)
  const ah = 9, aw = 4.5
  const b1x = tx - ah * Math.cos(ang) - aw * Math.sin(ang)
  const b1y = ty + ah * Math.sin(ang) - aw * Math.cos(ang)
  const b2x = tx - ah * Math.cos(ang) + aw * Math.sin(ang)
  const b2y = ty + ah * Math.sin(ang) + aw * Math.cos(ang)
  return (
    <g>
      <line x1={ox} y1={oy} x2={tx} y2={ty}
        stroke={color} strokeWidth={width} strokeLinecap="round" />
      <polygon points={`${tx},${ty} ${b1x},${b1y} ${b2x},${b2y}`} fill={color} />
    </g>
  )
}

// Dashed guide line through origin along direction e
function EigenLine({ e, color }: { e: Vec2; color: string }) {
  const scale = RANGE * 1.4
  return (
    <line
      x1={toSx(-e[0] * scale)} y1={toSy(-e[1] * scale)}
      x2={toSx(e[0] * scale)} y2={toSy(e[1] * scale)}
      stroke={color} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.7}
    />
  )
}

// ── Interactive eigen canvas ───────────────────────────────────────────────────
function EigenCanvas({
  M,
  probe,
  onProbe,
}: {
  M: Mat2
  probe: Vec2
  onProbe: (v: Vec2) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const isDragging = useRef(false)

  const Mv = apply(M, probe)
  const eigen = computeEigen(M)
  const angle = angleDeg(probe, Mv)
  const probeLen = Math.hypot(probe[0], probe[1])
  const MvLen = Math.hypot(Mv[0], Mv[1])
  const isParallel = probeLen > 0.15 && MvLen > 0.05 && angle < PARALLEL_DEG
  const λ = eigenScalar(probe, Mv)

  // Normalized probe direction (for the celebration highlight line)
  const probeDir: Vec2 = probeLen > 0
    ? [probe[0] / probeLen, probe[1] / probeLen]
    : [1, 0]

  const handleMove = (clientX: number, clientY: number) => {
    if (!svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return
    // 窄屏下 SVG 被 CSS 等比缩小，先换算回 viewBox 内坐标再除以 UNIT。
    const px = (clientX - rect.left) * (SIZE / rect.width)
    const py = (clientY - rect.top) * (SIZE / rect.height)
    const mx = clampVec(toMx(px))
    const my = clampVec(toMy(py))
    if (Math.hypot(mx, my) > 0.15) onProbe([mx, my])
  }

  // Label x offset: flip left if near right edge
  const labelX = (sx: number, w: number) => sx > SIZE - w - 10 ? -(w + 4) : 14

  return (
    <div style={{
      border: `1.5px solid ${isParallel ? '#2e7d32' : '#e6e8ea'}`,
      borderRadius: 8,
      overflow: 'hidden',
      background: '#fff',
      boxShadow: isParallel
        ? '0 0 0 3px rgba(46,125,50,0.20), 0 1px 8px rgba(0,0,0,0.08)'
        : '0 1px 6px rgba(0,0,0,0.06)',
      transition: 'box-shadow 0.25s, border-color 0.25s',
    }}>
      <svg
        ref={svgRef}
        width={SIZE} height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        style={{ display: 'block', touchAction: 'none', cursor: 'crosshair' }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          isDragging.current = true
          handleMove(e.clientX, e.clientY)
        }}
        onPointerMove={(e) => {
          if (!isDragging.current) return
          handleMove(e.clientX, e.clientY)
        }}
        onPointerUp={() => { isDragging.current = false }}
        onPointerCancel={() => { isDragging.current = false }}
      >
        <defs>
          <clipPath id="eigen-canvas-clip">
            <rect x={0} y={0} width={SIZE} height={SIZE} />
          </clipPath>
        </defs>

        <g clipPath="url(#eigen-canvas-clip)">
          <GridLines />

          {/* Eigenvector guide lines (dashed) — faint hints to hunt for */}
          {eigen.kind === 'real' && (
            <>
              <EigenLine e={eigen.e1} color={EIGEN1_COLOR} />
              {!eigen.repeated && (
                <EigenLine e={eigen.e2} color={EIGEN2_COLOR} />
              )}
            </>
          )}

          {/* Mv arrow (IKB blue) — result of the transformation */}
          <Arrow tip={Mv} color={IKB} width={2.8} />

          {/* v arrow (rust) — the probe vector */}
          <Arrow tip={probe} color={RUST} width={2.8} />

          {/* v label */}
          <text
            x={toSx(probe[0]) + labelX(toSx(probe[0]), 10)}
            y={toSy(probe[1]) + 5}
            fill={RUST} fontSize={13} fontWeight="bold"
            style={{ pointerEvents: 'none', userSelect: 'none' }}
          >v</text>

          {/* Mv label (only when Mv is long enough to see) */}
          {MvLen > 0.1 && (
            <text
              x={toSx(Mv[0]) + labelX(toSx(Mv[0]), 24)}
              y={toSy(Mv[1]) + 5}
              fill={IKB} fontSize={13} fontWeight="bold"
              style={{ pointerEvents: 'none', userSelect: 'none' }}
            >Mv</text>
          )}

          {/* Draggable probe handle */}
          <circle
            cx={toSx(probe[0])} cy={toSy(probe[1])}
            r={9} fill={RUST} stroke="white" strokeWidth={2.5}
            style={{ cursor: 'grab' }}
          />

          {/* Celebration overlay when v aligns with an eigenvector */}
          {isParallel && (
            <>
              {/* Highlight the eigenline */}
              <line
                x1={toSx(-probeDir[0] * RANGE * 1.4)}
                y1={toSy(-probeDir[1] * RANGE * 1.4)}
                x2={toSx(probeDir[0] * RANGE * 1.4)}
                y2={toSy(probeDir[1] * RANGE * 1.4)}
                stroke="#2e7d32" strokeWidth={2.5} opacity={0.45}
              />
              {/* λ readout near Mv tip */}
              <text
                x={toSx(Mv[0]) + labelX(toSx(Mv[0]), 60)}
                y={toSy(Mv[1]) - 10}
                fill="#2e7d32" fontSize={12} fontWeight="bold"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >λ ≈ {λ.toFixed(2)}</text>
            </>
          )}
        </g>
      </svg>
    </div>
  )
}

// ── Matrix display ────────────────────────────────────────────────────────────
function MatrixDisplay({ M }: { M: Mat2 }) {
  const f = (n: number) => {
    const r = Math.round(n * 100) / 100
    return (Object.is(r, -0) ? 0 : r).toFixed(2)
  }
  return (
    <div className="matrix">
      <span className="matrix-name">M</span>
      <span className="bracket">[</span>
      <span className="matrix-rows">
        <span>{f(M[0])}{'  '}{f(M[1])}</span>
        <span>{f(M[2])}{'  '}{f(M[3])}</span>
      </span>
      <span className="bracket">]</span>
    </div>
  )
}

// ── Preset matrices ───────────────────────────────────────────────────────────
type PresetKey = 'diagonal' | 'symmetric' | 'rotation' | 'shear'

interface Preset {
  key: PresetKey
  label: string
  M: Mat2
  hint: string
}

const PRESETS: Preset[] = [
  {
    key: 'diagonal',
    label: '对角',
    M: [3, 0, 0, 1],
    hint: 'λ = 3, λ = 1 · 两个坐标轴就是 eigenvector 方向',
  },
  {
    key: 'symmetric',
    label: '对称',
    M: [2, 1, 1, 2],
    hint: 'λ = 3, λ = 1 · 特征方向斜 45°',
  },
  {
    key: 'rotation',
    label: '旋转 90°',
    M: [0, -1, 1, 0],
    hint: '纯旋转 · discriminant < 0 · 无实 eigenvector',
  },
  {
    key: 'shear',
    label: '剪切',
    M: [1, 1, 0, 1],
    hint: 'λ = 1（重复）· 唯一特征方向是 x 轴',
  },
]

// ── Code snippet ──────────────────────────────────────────────────────────────
const SNIPPET = `# 2×2 特征值 · trace / det 闭合公式（Python / NumPy）
def eigen_2x2(M):
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    t = a + d            # trace  = λ₁ + λ₂
    D = a * d - b * c    # det    = λ₁ · λ₂
    disc = (t / 2) ** 2 - D
    if disc < 0:
        return "复数特征值 —— 纯旋转，无实 eigenvector"
    lam1 = t / 2 + disc ** 0.5
    lam2 = t / 2 - disc ** 0.5
    # eigenvector: (M − λI)v = 0 → v = [b, λ − a]
    import numpy as np
    e1 = np.array([b, lam1 - a], dtype=float)
    e2 = np.array([b, lam2 - a], dtype=float)
    e1 /= np.linalg.norm(e1) or 1
    e2 /= np.linalg.norm(e2) or 1
    return lam1, lam2, e1, e2

# 验证 M @ e ≈ λ * e：
M = np.array([[3., 0.], [0., 1.]])
l1, l2, e1, e2 = eigen_2x2(M)
assert np.allclose(M @ e1, l1 * e1)   # True ✓
assert np.allclose(M @ e2, l2 * e2)   # True ✓`

// ── Main page component ───────────────────────────────────────────────────────
export function Eigen() {
  const [activePreset, setActivePreset] = useState<PresetKey>('diagonal')
  const [probe, setProbe] = useState<Vec2>([1.5, 0.5])

  const preset = PRESETS.find(p => p.key === activePreset)!
  const M = preset.M
  const eigen = computeEigen(M)

  // For the verdict section (same math as canvas, computed here for JSX readout)
  const Mv = apply(M, probe)
  const probeLen = Math.hypot(probe[0], probe[1])
  const MvLen = Math.hypot(Mv[0], Mv[1])
  const angle = angleDeg(probe, Mv)
  const isParallel = probeLen > 0.15 && MvLen > 0.05 && angle < PARALLEL_DEG
  const λAtProbe = eigenScalar(probe, Mv)

  const fmtλ = (n: number) => {
    const r = Math.round(n * 100) / 100
    return (Object.is(r, -0) ? 0 : r).toFixed(2)
  }

  return (
      <ChapterShell
        slug="eigen"
        part="第三部分 · 方阵的秘密"
        sub="矩阵里「岿然不动」的方向"
        lede={
          <>
        对大多数向量，矩阵 M 既<strong>旋转</strong>又<strong>缩放</strong>它——<code>Mv</code>{' '}
        指向一个全新的方向。但有几个特殊方向，M 对它们<strong>只缩放，不旋转</strong>：
        {' '}<code>Mv&nbsp;=&nbsp;λv</code>。
        那些方向叫做 eigenvector（特征向量），那个比例叫做 eigenvalue（特征值 λ）。
        它们是变换的「纹理」——矩阵沿这些轴行事最简单。
        纯旋转矩阵没有实 eigenvector：它把每个方向都转走了，discriminant 为负，
        特征值落入复数域。
          </>
        }
      >

      {/* Preset selector */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">矩阵预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>选一个探索</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map(p => (
              <button
                key={p.key}
                onClick={() => {
                  setActivePreset(p.key)
                  setProbe([1.5, 0.5])
                }}
                style={{
                  padding: '0.35rem 0.9rem',
                  border: `1.5px solid ${activePreset === p.key ? IKB : '#ccd0d5'}`,
                  borderRadius: 4,
                  background: activePreset === p.key ? IKB : 'white',
                  color: activePreset === p.key ? 'white' : '#333',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                  fontWeight: activePreset === p.key ? 700 : 400,
                  transition: 'background 0.15s, color 0.15s, border-color 0.15s',
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
          <p style={{ margin: '0.4rem 0 0', color: '#666', fontSize: '0.82rem' }}>
            {preset.hint}
          </p>
        </div>
      </section>

      {/* Canvas stage */}
      <section
        className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}
      >
        <EigenCanvas M={M} probe={probe} onProbe={setProbe} />
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0, textAlign: 'center' }}>
          点击或拖动画布，让{' '}
          <span style={{ color: RUST, fontWeight: 700 }}>v</span>
          {' '}扫过各个方向——观察{' '}
          <span style={{ color: IKB, fontWeight: 700 }}>Mv</span>
          {' '}何时与 v 平行。
          虚线（
          <span style={{ color: EIGEN1_COLOR, fontWeight: 700 }}>金</span>
          {' '}／{' '}
          <span style={{ color: EIGEN2_COLOR, fontWeight: 700 }}>绿</span>
          ）是真实特征方向的提示
        </p>
      </section>

      {/* Eigenvalue readout + matrix */}
      <section
        className="readouts"
        style={{ justifyContent: 'center', alignItems: 'flex-start', gap: '2rem', flexWrap: 'wrap' }}
      >
        <MatrixDisplay M={M} />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem', minWidth: 200 }}>
          {eigen.kind === 'complex' ? (
            <div style={{ color: '#c62828', fontWeight: 600, fontSize: '0.9rem', lineHeight: 1.5 }}>
              特征值为复数<br />
              <span style={{ fontWeight: 400, fontSize: '0.82rem', color: '#888' }}>
                discriminant &lt; 0 · 纯旋转 · 无实 eigenvector
              </span>
            </div>
          ) : (
            <>
              <div>
                <span style={{ color: EIGEN1_COLOR, fontWeight: 700 }}>
                  λ₁ = {fmtλ(eigen.λ1)}
                </span>
                <span style={{ color: '#888', fontSize: '0.8rem', marginLeft: '0.6rem' }}>
                  e₁ = ({fmtλ(eigen.e1[0])}, {fmtλ(eigen.e1[1])})
                </span>
              </div>
              <div>
                <span style={{ color: EIGEN2_COLOR, fontWeight: 700 }}>
                  λ₂ = {fmtλ(eigen.λ2)}
                </span>
                <span style={{ color: '#888', fontSize: '0.8rem', marginLeft: '0.6rem' }}>
                  e₂ = ({fmtλ(eigen.e2[0])}, {fmtλ(eigen.e2[1])})
                </span>
              </div>
              {eigen.repeated && (
                <div style={{ color: '#888', fontSize: '0.8rem' }}>
                  重复 eigenvalue · 特征空间可能退化（defective）
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {/* Verdict */}
      <section className={`verdict ${isParallel ? 'verdict--eq' : 'verdict--neq'}`}>
        {isParallel ? (
          <p>
            <strong>找到特征向量！</strong>{' '}
            此时 <code>Mv ≈ {λAtProbe.toFixed(2)} · v</code>——
            v 被缩放了 {Math.abs(λAtProbe).toFixed(2)} 倍
            {λAtProbe < 0 ? '，方向翻转（λ &lt; 0）' : '，方向不变'}。
            <strong> 矩阵对这个方向只拉伸或压缩，绝不旋转——这就是 eigenvalue 的几何含义。</strong>
          </p>
        ) : (
          <p>
            v 与 Mv 夹角 {angle.toFixed(1)}°——矩阵旋转了 v，它还不是 eigenvector。
            {eigen.kind === 'real'
              ? ' 把探针移向金色或绿色虚线方向，当 Mv 完全平行于 v 时就找到了。'
              : ' 这是旋转矩阵——它对每个实数方向都旋转，真正的特征方向在复数域里。'}
          </p>
        )}
      </section>

      {/* Bridge to LLM */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            Eigenvector 是变换里「岿然不动」的主轴。把同一个矩阵反复作用在一个向量上，
            向量会被拉向 <strong>|λ| 最大</strong> 的特征方向——这是
            <strong>幂迭代（power iteration）</strong>的直觉，
            也是 PageRank 式传播之所以收敛的根本原因。
          </p>
          <p>
            在 Transformer 里，残差连接 + LayerNorm 的叠加在隐式地控制各层变换的 eigenvalue 谱不爆炸、不消亡，
            这关系到梯度流的稳定性。
            往后看：<ChRef slug="spectral" /><strong>谱分解</strong>把对称矩阵完全拆成特征方向的直和；
            <ChRef slug="svd" /> <strong>SVD</strong> 把任意矩阵写成旋转-拉伸-旋转——
            奇异值正是 MᵀM 特征值的平方根，eigenvalue 是那里的地基。
          </p>
        </div>
      </section>

      {/* Code block */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：2×2 eigenvalue 的闭合公式</h2>
        <CodeBlock code={SNIPPET} language="python" title="eigen_2x2.py" />
      </section>

      {/* Pager */}
      </ChapterShell>
  )
}
