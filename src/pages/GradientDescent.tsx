import React, { useState, useEffect, useCallback, type CSSProperties } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── SVG layout ─────────────────────────────────────────────────────────────
const SVG_W = 420
const SVG_H = 320
const CX = SVG_W / 2   // domain origin in SVG pixels
const CY = SVG_H / 2
const SCALE_X = 55     // px per domain unit (x)
const SCALE_Y = 45     // px per domain unit (y)

// ── descent config ─────────────────────────────────────────────────────────
const MAX_STEPS = 60
const DIVERGE_BOUND = 8   // clamp trajectory beyond this many domain units

// Loss f(θ₀,θ₁) = θ₀² + 5θ₁².
// Gradient: ∇f = [2θ₀, 10θ₁].
// Hessian eigenvalues: 2 and 10 → stable iff η < 2/10 = 0.200.
// At η = 0.22: y-factor = 1 − 10×0.22 = −1.2, |−1.2|>1 → diverges.

const ARROW_LEN = 26           // pixels for the descent-direction arrow
const IKB  = '#002fa7'         // Internationale Klein Blue — trajectory
const RUST = '#c75b39'         // current point / divergence

const CONTOUR_LEVELS = [0.2, 0.5, 1, 2, 3.5, 5.5, 8] as const

// ── math ───────────────────────────────────────────────────────────────────
function lossF(x: number, y: number): number { return x * x + 5 * y * y }
function gradFx(x: number): number { return 2 * x }
function gradFy(y: number): number { return 10 * y }

// ── coordinate transforms ──────────────────────────────────────────────────
function toSvgX(x: number): number { return CX + x * SCALE_X }
function toSvgY(y: number): number { return CY - y * SCALE_Y }
function fromSvgX(px: number): number { return (px - CX) / SCALE_X }
function fromSvgY(py: number): number { return -(py - CY) / SCALE_Y }

// ── trajectory ────────────────────────────────────────────────────────────
interface Pt { x: number; y: number; diverged: boolean }

function computeTrajectory(sx: number, sy: number, lr: number): Pt[] {
  const pts: Pt[] = [{ x: sx, y: sy, diverged: false }]
  let cx = sx
  let cy = sy
  for (let i = 0; i < MAX_STEPS; i++) {
    const gx = gradFx(cx)
    const gy = gradFy(cy)
    const nx = cx - lr * gx
    const ny = cy - lr * gy
    const bad =
      !isFinite(nx) || !isFinite(ny) ||
      Math.abs(nx) > DIVERGE_BOUND || Math.abs(ny) > DIVERGE_BOUND
    if (bad) {
      const cx2 = isFinite(nx) ? Math.max(-DIVERGE_BOUND, Math.min(DIVERGE_BOUND, nx)) : cx
      const cy2 = isFinite(ny) ? Math.max(-DIVERGE_BOUND, Math.min(DIVERGE_BOUND, ny)) : cy
      pts.push({ x: cx2, y: cy2, diverged: true })
      break
    }
    cx = nx
    cy = ny
    pts.push({ x: cx, y: cy, diverged: false })
    if (Math.abs(gx) < 1e-5 && Math.abs(gy) < 1e-5) break
  }
  return pts
}

// ── presets ────────────────────────────────────────────────────────────────
const PRESETS = [
  { label: '太小（慢）',    lr: 0.01 },
  { label: '刚好',          lr: 0.09 },
  { label: '太大（发散）',  lr: 0.22 },
] as const

// ── shared styles ──────────────────────────────────────────────────────────
const btnBase: CSSProperties = {
  padding: '4px 12px', fontSize: 13, cursor: 'pointer',
  borderRadius: 4, border: '1.5px solid #ccc', background: '#fff', color: '#333',
}
const cardBase: CSSProperties = {
  padding: '8px 14px', borderRadius: 6, border: '1px solid #e0e0e0',
  background: '#fff', minWidth: 120,
}
const cardLabel: CSSProperties = { fontSize: 11, color: '#888', marginBottom: 2, fontFamily: 'monospace' }
const cardVal:   CSSProperties = { fontSize: 15, fontWeight: 600, fontFamily: 'monospace', color: '#1b1f24' }

// ── code snippet ───────────────────────────────────────────────────────────
const CODE_SNIPPET = `\
def loss(theta):        # f(θ) = θ₀² + 5·θ₁²
    return theta[0]**2 + 5 * theta[1]**2

def grad(theta):        # ∇f = [2θ₀, 10θ₁]
    return [2 * theta[0], 10 * theta[1]]

theta = [2.5, 1.5]     # 任意起始点
lr    = 0.09           # learning rate η

for step in range(60):
    g = grad(theta)
    theta = [theta[0] - lr * g[0],   # θ ← θ − η·∇f
             theta[1] - lr * g[1]]
    if max(abs(g[0]), abs(g[1])) < 1e-5:
        print(f"收敛 @ step {step}"); break

# lr = 0.22：y 更新因子 = 1 − 10×0.22 = −1.2
# |−1.2| > 1 → 振幅逐步放大 → diverge
# 真实训练（SGD）：theta -= lr * grad_minibatch(theta, batch)
# Adam = SGD + 一阶矩（momentum）+ 二阶矩（自适应 η）`

// ── component ──────────────────────────────────────────────────────────────
export function GradientDescent() {
  const [start, setStart]           = useState<{ x: number; y: number } | null>(null)
  const [lr, setLr]                 = useState(0.09)
  const [currentStep, setCurrentStep] = useState(0)
  const [playing, setPlaying]       = useState(false)

  const trajectory  = start !== null ? computeTrajectory(start.x, start.y, lr) : []
  const maxStep     = Math.max(0, trajectory.length - 1)
  const safeStep    = Math.min(currentStep, maxStep)
  const visiblePts  = trajectory.slice(0, safeStep + 1)
  const curPt: Pt | undefined = trajectory[safeStep]
  const isDiverged  = visiblePts.some((p) => p.diverged)

  // Reset animation whenever start point or lr changes
  useEffect(() => {
    setCurrentStep(0)
    setPlaying(false)
  }, [start, lr])

  // Animation loop — advances one step every 180 ms
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setCurrentStep((s) => (s < maxStep ? s + 1 : s))
    }, 180)
    return () => clearInterval(id)
  }, [playing, maxStep])

  const handleSvgClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const dx = Math.max(-3.5, Math.min(3.5, fromSvgX(e.clientX - rect.left)))
    const dy = Math.max(-3.0, Math.min(3.0, fromSvgY(e.clientY - rect.top)))
    setStart({ x: dx, y: dy })
    setPlaying(false)
  }, [])

  const me = findChapter('gradient-descent')!
  const { prev, next } = neighbors('gradient-descent')

  // Per-step math
  const gx   = curPt !== undefined ? gradFx(curPt.x) : 0
  const gy   = curPt !== undefined ? gradFy(curPt.y) : 0
  const gMag = Math.sqrt(gx * gx + gy * gy)
  const fVal = curPt !== undefined ? lossF(curPt.x, curPt.y) : 0

  // Descent arrow in SVG pixel-space.
  // Descent direction in domain: (−gx, −gy).
  // SVG x is same direction; SVG y is flipped → domain −gy maps to SVG +gy·SCALE_Y.
  const descSvgX = -gx * SCALE_X
  const descSvgY =  gy * SCALE_Y
  const descMag  = Math.sqrt(descSvgX * descSvgX + descSvgY * descSvgY)
  const arrowDx  = descMag > 0.01 ? (descSvgX / descMag) * ARROW_LEN : 0
  const arrowDy  = descMag > 0.01 ? (descSvgY / descMag) * ARROW_LEN : 0

  const isConverged = !isDiverged && gMag < 1e-3 && safeStep > 0

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第六部分 · 学习：模型怎么变聪明
        </div>
        <div className="kicker">第 {me.num} 节 · 前置：DSCI 6200 · 监督学习</div>
        <h1>梯度下降<span className="zh-sub">模型怎么一步步逼近答案？</span></h1>
        <p className="lede">
          训练一个模型，本质上是在 loss 曲面上找最低点。
          <strong>梯度下降（gradient descent）</strong>只有一条规则：
          <code>θ ← θ − η·∇f(θ)</code>——沿最陡下坡方向迈一步。
          <strong>Learning rate η</strong> 控制步长：η 太小走得慢；
          η 刚好则稳稳 converge；η 太大就会 overshoot，甚至 diverge。
          在下面的等高线图上点一个起点，看轨迹怎么走。
        </p>
      </header>

      {/* ── learning-rate slider + presets ── */}
      <section className="controls">
        <div className="control">
          <label className="slider-row" style={{ gap: 12, alignItems: 'center' }}>
            <span style={{ fontWeight: 700, fontFamily: 'monospace', minWidth: 96, fontSize: 15 }}>
              η = {lr.toFixed(3)}
            </span>
            <input
              type="range" min={0.005} max={0.28} step={0.005}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              style={{ flex: 1, accentColor: IKB }}
            />
          </label>
          <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            {PRESETS.map((p) => {
              const active = Math.abs(lr - p.lr) < 0.001
              return (
                <button
                  key={p.label}
                  onClick={() => setLr(p.lr)}
                  style={{
                    ...btnBase,
                    border:     `1.5px solid ${active ? IKB : '#ccc'}`,
                    background: active ? IKB : '#fff',
                    color:      active ? '#fff' : '#333',
                  }}
                >
                  {p.label}
                </button>
              )
            })}
          </div>
          <p style={{ fontSize: 12, color: '#888', margin: '6px 0 0', fontFamily: 'monospace' }}>
            稳定阈值：η &lt; 0.200（最大 Hessian 特征值 = 10，来自 5θ₁² 项）
          </p>
        </div>
      </section>

      {/* ── SVG contour plot + trajectory ── */}
      <section className="stage" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
        <p style={{ color: '#888', fontSize: 13, margin: 0 }}>
          点击等高线图设置起点 · f(θ₀, θ₁) = θ₀² + 5θ₁²（横宽竖窄的椭圆碗）
        </p>

        <svg
          width={SVG_W} height={SVG_H}
          onClick={handleSvgClick}
          style={{ cursor: 'crosshair', border: '1px solid #e8e8e8', borderRadius: 6, background: '#fafafa' }}
        >
          <defs>
            {/* arrowhead for descent direction */}
            <marker id="gd-arr" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill={RUST} />
            </marker>
          </defs>

          {/* grid */}
          {([-3, -2, -1, 0, 1, 2, 3] as number[]).map((v) => (
            <g key={v}>
              <line x1={toSvgX(v)} y1={0}     x2={toSvgX(v)} y2={SVG_H}
                stroke="#e0e0e0" strokeWidth={v === 0 ? 1 : 0.4} />
              <line x1={0}     y1={toSvgY(v)} x2={SVG_W}    y2={toSvgY(v)}
                stroke="#e0e0e0" strokeWidth={v === 0 ? 1 : 0.4} />
            </g>
          ))}

          {/* contour ellipses: f = c → rx = √c · SCALE_X, ry = √(c/5) · SCALE_Y */}
          {CONTOUR_LEVELS.map((c, idx) => {
            const t  = idx / (CONTOUR_LEVELS.length - 1)   // 0 (outer, light) → 1 (inner, IKB)
            // interpolate light gray #d0d0d0 → IKB #002fa7
            const cr = Math.round(208 * (1 - t))
            const cg = Math.round(208 * (1 - t) + 47 * t)
            const cb = Math.round(208 * (1 - t) + 167 * t)
            return (
              <ellipse
                key={c}
                cx={CX} cy={CY}
                rx={Math.sqrt(c) * SCALE_X}
                ry={Math.sqrt(c / 5) * SCALE_Y}
                fill="none"
                stroke={`rgb(${cr},${cg},${cb})`}
                strokeWidth={1}
              />
            )
          })}

          {/* trajectory segments */}
          {visiblePts.map((pt, i) => {
            if (i === 0) return null
            const prevPt = visiblePts[i - 1]
            if (prevPt === undefined) return null
            return (
              <line
                key={`seg-${i}`}
                x1={toSvgX(prevPt.x)} y1={toSvgY(prevPt.y)}
                x2={toSvgX(pt.x)}     y2={toSvgY(pt.y)}
                stroke={pt.diverged ? RUST : IKB}
                strokeWidth={2}
                strokeDasharray={pt.diverged ? '5 3' : undefined}
              />
            )
          })}

          {/* trajectory dots */}
          {visiblePts.map((pt, i) => {
            const isLast = i === visiblePts.length - 1
            return (
              <circle
                key={`dot-${i}`}
                cx={toSvgX(pt.x)} cy={toSvgY(pt.y)}
                r={isLast ? 6 : 3}
                fill={isLast ? RUST : IKB}
                opacity={isLast ? 1 : 0.5}
                stroke={isLast ? '#fff' : 'none'}
                strokeWidth={2}
              />
            )
          })}

          {/* descent-direction arrow at current point */}
          {curPt !== undefined && !isDiverged && gMag > 0.01 && (
            <line
              x1={toSvgX(curPt.x)}
              y1={toSvgY(curPt.y)}
              x2={toSvgX(curPt.x) + arrowDx}
              y2={toSvgY(curPt.y) + arrowDy}
              stroke={RUST} strokeWidth={2}
              markerEnd="url(#gd-arr)"
            />
          )}

          {/* axis labels */}
          <text x={toSvgX(3.3)} y={CY - 6}     fontSize={11} fill="#bbb" textAnchor="end">θ₀</text>
          <text x={CX + 6}      y={toSvgY(2.8)} fontSize={11} fill="#bbb">θ₁</text>

          {start === null && (
            <text x={CX} y={CY + 22} textAnchor="middle" fontSize={13} fill="#ccc">
              点击任意位置设置起点
            </text>
          )}
        </svg>

        {/* playback controls */}
        {start !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <button
              style={btnBase}
              onClick={() => { setCurrentStep(0); setPlaying(false) }}
            >
              重置
            </button>
            <button
              style={btnBase}
              disabled={safeStep === 0}
              onClick={() => setCurrentStep((s) => Math.max(0, s - 1))}
            >
              ← 退一步
            </button>
            <button
              style={{ ...btnBase, background: IKB, color: '#fff', borderColor: IKB }}
              onClick={() => {
                if (playing) {
                  setPlaying(false)
                } else if (safeStep >= maxStep) {
                  setCurrentStep(0)
                  setPlaying(true)
                } else {
                  setPlaying(true)
                }
              }}
            >
              {playing ? '暂停' : safeStep >= maxStep ? '重播' : '播放'}
            </button>
            <button
              style={btnBase}
              disabled={safeStep >= maxStep}
              onClick={() => setCurrentStep((s) => Math.min(maxStep, s + 1))}
            >
              前进一步 →
            </button>
            <span style={{ fontSize: 13, color: '#666', fontFamily: 'monospace' }}>
              第 {safeStep} / {maxStep} 步
            </span>
          </div>
        )}
      </section>

      {/* ── readouts ── */}
      {start !== null && curPt !== undefined && (
        <section className="readouts" style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <div style={cardBase}>
            <div style={cardLabel}>θ = (θ₀, θ₁)</div>
            <div style={cardVal}>({curPt.x.toFixed(3)}, {curPt.y.toFixed(3)})</div>
          </div>
          <div style={cardBase}>
            <div style={cardLabel}>f(θ)</div>
            <div style={cardVal}>{fVal.toFixed(4)}</div>
          </div>
          <div style={cardBase}>
            <div style={cardLabel}>|∇f(θ)|</div>
            <div style={cardVal}>{gMag.toFixed(4)}</div>
          </div>
          <div style={cardBase}>
            <div style={cardLabel}>步数</div>
            <div style={cardVal}>{safeStep}</div>
          </div>
          {isDiverged && (
            <div style={{ ...cardBase, borderColor: RUST, background: '#fff5f2' }}>
              <div style={{ ...cardLabel, color: RUST }}>状态</div>
              <div style={{ ...cardVal,   color: RUST }}>发散 ✕</div>
            </div>
          )}
          {isConverged && (
            <div style={{ ...cardBase, borderColor: '#0a7d52', background: '#f0faf5' }}>
              <div style={{ ...cardLabel, color: '#0a7d52' }}>状态</div>
              <div style={{ ...cardVal,   color: '#0a7d52' }}>收敛 ✓</div>
            </div>
          )}
        </section>
      )}

      {/* ── verdict ── */}
      {start !== null && curPt !== undefined && (
        <section className={`verdict ${isDiverged ? 'verdict--neq' : 'verdict--eq'}`}>
          {isDiverged ? (
            <p>
              <strong>Diverge 了。</strong> η = {lr.toFixed(3)} 超过了稳定阈值 0.200。
              θ₁ 方向的更新因子 = 1 − 10 × {lr.toFixed(3)} ={' '}
              <strong style={{ color: RUST }}>{(1 - 10 * lr).toFixed(3)}</strong>，
              绝对值大于 1，每步都在放大误差，轨迹向外爆炸。把 η 调小到 0.2 以下再试。
            </p>
          ) : isConverged ? (
            <p>
              <strong>Converge 了。</strong> 经过 {safeStep} 步，
              |∇f| = {gMag.toExponential(2)}，loss = {fVal.toFixed(5)}。
              η = {lr.toFixed(3)} 是安全步长；y 方向因子 = {(1 - 10 * lr).toFixed(2)}（绝对值 &lt; 1）。
            </p>
          ) : (
            <p>
              第 {safeStep} 步：θ = ({curPt.x.toFixed(3)}, {curPt.y.toFixed(3)})，
              f(θ) = {fVal.toFixed(4)}。继续播放，或拖动 η 滑块对比不同步长的效果。
            </p>
          )}
        </section>
      )}

      {/* ── bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            这就是训练循环的核心：<code>for batch: θ −= lr × grad</code>。
            真实 LLM 用的是它的变体——<strong>SGD（小批量）+ momentum（动量）+
            Adam（自适应步长）</strong>——但骨架就是这一行：沿负梯度方向迈一步。
          </p>
          <p>
            Learning rate 是最重要的超参之一。调度策略（warmup → cosine decay）和梯度裁剪
            （gradient clipping）都是在管「步子大小」。Loss 曲线震荡、训练不稳、diverge——
            你在这页都能亲手复现。Adam 的本质就是对每个参数自适应地调整这个 η——
            理解了这一页，Adam 就自然了。
          </p>
        </div>
      </section>

      {/* ── code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：梯度下降循环</h2>
        <CodeBlock code={CODE_SNIPPET} language="python" title="gradient_descent.py" />
      </section>

      {/* ── pager ── */}
      <nav className="pager">
        {prev !== undefined
          ? <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
              <span className="pager-dir">← 上一章</span>
              <span className="pager-title">{prev.num} {prev.title}</span>
            </Link>
          : <span />}
        {next !== undefined
          ? <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
              <span className="pager-dir">下一章 →</span>
              <span className="pager-title">{next.num} {next.title}{next.status !== 'live' && ' · 规划中'}</span>
            </Link>
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
