import { useState, useRef, useCallback } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { ChapterShell } from '../components/ChapterShell'

// ─── Colors ───────────────────────────────────────────────────────────────────
const IKB = '#002fa7'
const RUST = '#c75b39'
const INK = '#5b6168'
const GRID_C = '#e8eaed'

// ─── Loss surface ─────────────────────────────────────────────────────────────
// f(x,y) = 2u² + 0.8v² + 0.6uv,  u = x − CX, v = y − CY
// Q = [[2, 0.3],[0.3, 0.8]] is positive-definite (det = 1.51 > 0, trace > 0).
// Minimum at (CX, CY); non-circular elliptic level sets.
const CX = 0.5   // x of minimum
const CY = -0.3  // y of minimum

function f(x: number, y: number): number {
  const u = x - CX
  const v = y - CY
  return 2 * u * u + 0.8 * v * v + 0.6 * u * v
}

// Analytic gradient (chain rule on affine substitution u=x−CX, v=y−CY):
//   ∂f/∂x = ∂f/∂u = 4u + 0.6v
//   ∂f/∂y = ∂f/∂v = 1.6v + 0.6u
// At the minimum u=v=0, so ∇f = (0,0). ✓
// ∇f ⊥ level sets by definition of the gradient. ✓
function gradF(x: number, y: number): [number, number] {
  const u = x - CX
  const v = y - CY
  return [4 * u + 0.6 * v, 1.6 * v + 0.6 * u]
}

// ─── SVG viewport ─────────────────────────────────────────────────────────────
const W = 460
const H = 460
const XMIN = -3
const XMAX = 3
const YMIN = -3
const YMAX = 3

function sx(x: number): number { return (x - XMIN) / (XMAX - XMIN) * W }
function sy(y: number): number { return (YMAX - y) / (YMAX - YMIN) * H }

// ─── Contour rings (module-level; constants only) ──────────────────────────────
// For level c, the ellipse satisfies 2u² + 0.8v² + 0.6uv = c.
// Parametrize: u = r cosθ, v = r sinθ → r² D(θ) = c → r = √(c / D(θ))
//   D(θ) = 2cos²θ + 0.8sin²θ + 0.6cosθsinθ > 0 (Q positive-definite, always).
const LEVELS = [0.3, 0.7, 1.4, 2.5, 4.0, 6.2, 9.0, 12.5]
const N_THETA = 240

function buildContour(c: number): string {
  const pts: string[] = []
  for (let i = 0; i <= N_THETA; i++) {
    const theta = (i / N_THETA) * 2 * Math.PI
    const cosT = Math.cos(theta)
    const sinT = Math.sin(theta)
    const denom = 2 * cosT * cosT + 0.8 * sinT * sinT + 0.6 * cosT * sinT
    if (denom < 1e-9) continue
    const r = Math.sqrt(c / denom)
    pts.push(`${sx(CX + r * cosT).toFixed(1)},${sy(CY + r * sinT).toFixed(1)}`)
  }
  return pts.join(' ')
}

// Light gray → IKB as level index increases
function levelColor(i: number, n: number): string {
  const t = i / (n - 1)
  const r = Math.round(0xcc + t * (0x00 - 0xcc))
  const g = Math.round(0xd0 + t * (0x2f - 0xd0))
  const b = Math.round(0xda + t * (0xa7 - 0xda))
  return `rgb(${r},${g},${b})`
}

const CONTOUR_DATA = LEVELS.map((c, i) => ({
  key: c,
  pts: buildContour(c),
  color: levelColor(i, LEVELS.length),
}))

// ─── SVG arrow glyph ──────────────────────────────────────────────────────────
interface ArrowProps {
  ox: number; oy: number   // origin, screen px
  dx: number; dy: number   // direction + magnitude, screen px
  color: string
  label: string
}

function SvgArrow({ ox, oy, dx, dy, color, label }: ArrowProps) {
  const len = Math.sqrt(dx * dx + dy * dy)
  if (len < 5) return null
  const tx = ox + dx
  const ty = oy + dy
  const angle = Math.atan2(dy, dx)
  const hLen = Math.min(12, len * 0.36)
  const hAng = Math.PI / 7
  const hx1 = tx - hLen * Math.cos(angle - hAng)
  const hy1 = ty - hLen * Math.sin(angle - hAng)
  const hx2 = tx - hLen * Math.cos(angle + hAng)
  const hy2 = ty - hLen * Math.sin(angle + hAng)
  const lx = tx + 17 * Math.cos(angle)
  const ly = ty + 17 * Math.sin(angle)
  return (
    <g>
      <line x1={ox} y1={oy} x2={tx} y2={ty}
        stroke={color} strokeWidth={2.5} strokeLinecap="round" />
      <polygon points={`${tx},${ty} ${hx1},${hy1} ${hx2},${hy2}`} fill={color} />
      <text x={lx} y={ly} fill={color} fontSize={12} fontWeight={700}
        textAnchor="middle" dominantBaseline="middle">
        {label}
      </text>
    </g>
  )
}

// ─── Code snippet ─────────────────────────────────────────────────────────────
const SNIPPET = `\
// 损失函数：非圆形椭球碗，最低点在 (0.5, −0.3)
function f(x: number, y: number): number {
  const u = x - 0.5, v = y + 0.3
  return 2*u*u + 0.8*v*v + 0.6*u*v
}

// 解析梯度：对 x、y 分别求 partial derivative
function grad(x: number, y: number): [number, number] {
  const u = x - 0.5, v = y + 0.3
  return [4*u + 0.6*v, 1.6*v + 0.6*u]    // ∂f/∂x, ∂f/∂y
}

// 当前参数位置；−∇f 指向 loss 下降最快的方向
const x = 1.2, y = 0.8
const [gx, gy] = grad(x, y)
const descent: [number, number] = [-gx, -gy]

// 数值验证（中心差分 finite differences）
const h = 1e-5
const numGx = (f(x+h, y) - f(x-h, y)) / (2*h)   // ≈ gx, 误差 < 1e-9
const numGy = (f(x, y+h) - f(x, y-h)) / (2*h)   // ≈ gy, 误差 < 1e-9`

// ─── Page component ───────────────────────────────────────────────────────────
export function Gradient() {
  const [pt, setPt] = useState<[number, number]>([1.8, 1.5])
  const [grabbing, setGrabbing] = useState(false)
  const isDragging = useRef(false)
  const svgRef = useRef<SVGSVGElement>(null)

  // Convert client coords → math coords; uses ref (stable) + module constants.
  const toMath = useCallback((clientX: number, clientY: number): [number, number] => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return [0, 0]
    const rx = (clientX - rect.left) / rect.width
    const ry = (clientY - rect.top) / rect.height
    return [
      Math.max(XMIN + 0.05, Math.min(XMAX - 0.05, rx * (XMAX - XMIN) + XMIN)),
      Math.max(YMIN + 0.05, Math.min(YMAX - 0.05, (1 - ry) * (YMAX - YMIN) + YMIN)),
    ]
  }, [])

  const [mx, my] = pt
  const fVal = f(mx, my)
  const [gx, gy] = gradF(mx, my)
  const mag = Math.sqrt(gx * gx + gy * gy)

  const d3 = (n: number) => n.toFixed(3)
  const d2 = (n: number) => n.toFixed(2)

  // Scale arrow to screen: proportional to |∇f|, capped at 80 px.
  // screen dx: +gx → rightward; screen dy: +gy → upward (−gy in SVG coords).
  const arrowPx = Math.min(80, mag * 22)
  const arrowScale = mag > 1e-9 ? arrowPx / mag : 0
  const asx = gx * arrowScale        // ∇f x component in screen px
  const asy = -gy * arrowScale       // ∇f y component in screen px (flipped)

  const pox = sx(mx)
  const poy = sy(my)

  const verdictClass = mag < 0.15 ? 'verdict--eq' : 'verdict--neq'
  const verdictMsg = mag < 0.15
    ? '|∇f| ≈ 0：接近最小值，gradient 趋于零，已无可用的「下坡方向」。'
    : mag < 2.0
    ? `坡度适中。沿 −∇f（IKB蓝）走一步，loss 下降最快。|∇f| = ${d3(mag)}`
    : `坡度很陡：|∇f| = ${d3(mag)}，训练信号强，参数更新幅度大。`

  return (
      <ChapterShell
        slug="gradient"
        part="第六部分 · 学习：模型怎么变聪明"
        sub="往哪走，loss 降得最快？"
        lede={
          <>
        对多变量函数 <code>f(x,y)</code>，<strong>梯度 ∇f</strong> 是一个<strong>向量</strong>——
        指向 <em>f</em> 上升最快的方向，长度等于那个方向的坡度，且总是
        <strong>垂直于 contour（等高线/level set）</strong>。
        朝 <code>+∇f</code> 走，<em>f</em> 升得最快；朝 <strong>−∇f</strong> 走，
        <em>f</em> 降得最快——这就是训练时每个参数收到的全部信号。
        <strong>在图上点击并拖动圆点</strong>，看两个箭头如何随地形变化。
          </>
        }
      >

      {/* ── Interactive contour plot ── */}
      <section className="stage" style={{ flexDirection: 'column', alignItems: 'center', gap: 20 }}>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-start' }}>

          {/* SVG canvas */}
          <svg
            ref={svgRef}
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            style={{
              display: 'block',
              border: '1px solid #dde1e8',
              borderRadius: 8,
              background: '#fafbfc',
              cursor: grabbing ? 'grabbing' : 'crosshair',
              touchAction: 'none',
              userSelect: 'none',
            }}
            onMouseDown={(e) => {
              isDragging.current = true
              setGrabbing(true)
              setPt(toMath(e.clientX, e.clientY))
            }}
            onMouseMove={(e) => {
              if (!isDragging.current) return
              setPt(toMath(e.clientX, e.clientY))
            }}
            onMouseUp={() => { isDragging.current = false; setGrabbing(false) }}
            onMouseLeave={() => { isDragging.current = false; setGrabbing(false) }}
            onTouchStart={(e) => {
              const t = e.touches[0]
              isDragging.current = true
              setPt(toMath(t.clientX, t.clientY))
            }}
            onTouchMove={(e) => {
              if (!isDragging.current) return
              const t = e.touches[0]
              setPt(toMath(t.clientX, t.clientY))
            }}
            onTouchEnd={() => { isDragging.current = false }}
          >
            {/* Grid lines */}
            {([-2, -1, 0, 1, 2] as const).map((v) => (
              <g key={v}>
                <line x1={sx(v)} y1={0} x2={sx(v)} y2={H} stroke={GRID_C} strokeWidth={1} />
                <line x1={0} y1={sy(v)} x2={W} y2={sy(v)} stroke={GRID_C} strokeWidth={1} />
              </g>
            ))}

            {/* Axes */}
            <line x1={sx(0)} y1={0} x2={sx(0)} y2={H} stroke="#c2c7d0" strokeWidth={1.5} />
            <line x1={0} y1={sy(0)} x2={W} y2={sy(0)} stroke="#c2c7d0" strokeWidth={1.5} />
            <text x={W - 10} y={sy(0) - 8} fill={INK} fontSize={12} textAnchor="end">x</text>
            <text x={sx(0) + 6} y={14} fill={INK} fontSize={12}>y</text>

            {/* Contour rings: light gray → IKB by level; SVG clips overflow */}
            {CONTOUR_DATA.map(({ key, pts, color }) => (
              <polygon key={key} points={pts} fill="none" stroke={color} strokeWidth={1.8} />
            ))}

            {/* Minimum marker */}
            <circle
              cx={sx(CX)} cy={sy(CY)} r={5}
              fill="none" stroke={IKB} strokeWidth={1.5} strokeDasharray="3 2"
            />
            <text x={sx(CX) + 9} y={sy(CY) - 8} fill={IKB} fontSize={11} fontWeight={600}>
              最小值
            </text>

            {/* −∇f arrow (IKB): descent, loss drops fastest */}
            <SvgArrow ox={pox} oy={poy} dx={-asx} dy={-asy} color={IKB} label="−∇f" />

            {/* ∇f arrow (rust): ascent, f rises fastest */}
            <SvgArrow ox={pox} oy={poy} dx={asx} dy={asy} color={RUST} label="∇f" />

            {/* Draggable point */}
            <circle cx={pox} cy={poy} r={9} fill="white" stroke={INK} strokeWidth={2.5} />
            <circle cx={pox} cy={poy} r={3.5} fill={INK} />
          </svg>

          {/* Readout panel */}
          <div style={{ minWidth: 228, maxWidth: 290, display: 'flex', flexDirection: 'column', gap: 0 }}>
            <div style={{
              fontWeight: 700, fontSize: 12, color: INK, marginBottom: 10,
              letterSpacing: 0.6, textTransform: 'uppercase',
            }}>
              实时读数
            </div>

            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13.5 }}>
              <tbody>
                {([
                  ['位置 (x, y)', `(${d2(mx)}, ${d2(my)})`],
                  ['f (x, y)', d3(fVal)],
                  ['∂f / ∂x', d3(gx)],
                  ['∂f / ∂y', d3(gy)],
                  ['|∇f| 坡度', d3(mag)],
                  ['−∇f 下坡方向', `(${d3(-gx)}, ${d3(-gy)})`],
                ] as [string, string][]).map(([label, val]) => (
                  <tr key={label} style={{ borderBottom: '1px solid #eaecf0' }}>
                    <td style={{ padding: '6px 8px 6px 0', color: INK, whiteSpace: 'nowrap' }}>
                      {label}
                    </td>
                    <td style={{
                      padding: '6px 0 6px 4px',
                      fontFamily: 'monospace', fontSize: 13,
                      color: '#1b1f24', fontWeight: 600,
                      wordBreak: 'break-all',
                    }}>
                      {val}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div style={{
              marginTop: 14, padding: '10px 12px', borderRadius: 8,
              background: '#eaf0ff', borderLeft: `3px solid ${IKB}`,
              fontSize: 13, lineHeight: 1.65,
            }}>
              <strong style={{ color: IKB }}>IKB蓝 −∇f</strong>：loss 下降最快方向（训练信号）
            </div>
            <div style={{
              marginTop: 8, padding: '10px 12px', borderRadius: 8,
              background: '#fdf1ed', borderLeft: `3px solid ${RUST}`,
              fontSize: 13, lineHeight: 1.65,
            }}>
              <strong style={{ color: RUST }}>锈色 ∇f</strong>：f 上升最快方向（反向即下坡）
            </div>
            <div style={{
              marginTop: 8, padding: '10px 12px', borderRadius: 8,
              background: '#f4f5f7', borderLeft: '3px solid #c2c7d0',
              fontSize: 13, lineHeight: 1.65, color: INK,
            }}>
              两箭头总是<strong>垂直</strong>于当前位置的 contour ring。
            </div>
            <p style={{ marginTop: 12, fontSize: 12, color: '#9aa1a9', lineHeight: 1.55 }}>
              靠近虚线圆（最小值）时箭头趋于零；越靠近边缘，坡度越陡，箭头越长。
            </p>
          </div>
        </div>
      </section>

      {/* ── Verdict ── */}
      <section className={`verdict ${verdictClass}`}>
        <p>{verdictMsg}</p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            训练 = 把所有参数看成超高维空间里的一个点，loss 是它的「高度」。
            反向传播算出每个权重对应的 <strong>partial derivative</strong>，拼成
            <strong> −∇<sub>θ</sub>loss</strong>，优化器就沿这个方向挪一小步（第 26 节：梯度下降）。
            垂直于 level set = 最高效的下坡方向——所以 gradient 是训练的全部信号，不多也不少。
          </p>
        </div>
      </section>

      {/* ── Code ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：梯度的定义与数值验证</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="gradient.ts" />
      </section>

      {/* ── Pager ── */}
      </ChapterShell>
  )
}
