import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Types ─────────────────────────────────────────────────────────────────
type Point = { id: number; x: number; y: number }
type View = 'scatter' | 'vector'

// ── Initial data points (data-space coordinates: x, y ∈ [0, 9]) ────────────
const INIT_POINTS: Point[] = [
  { id: 0, x: 1.0, y: 2.1 },
  { id: 1, x: 2.1, y: 3.5 },
  { id: 2, x: 3.2, y: 2.8 },
  { id: 3, x: 4.5, y: 5.3 },
  { id: 4, x: 5.5, y: 4.7 },
  { id: 5, x: 6.3, y: 6.1 },
  { id: 6, x: 7.2, y: 5.6 },
  { id: 7, x: 8.1, y: 7.4 },
]

// ── OLS closed-form ─────────────────────────────────────────────────────────
// slope   = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
// intercept = ȳ − slope·x̄
function computeOLS(pts: readonly Point[]): { slope: number; intercept: number } {
  const n = pts.length
  const xBar = pts.reduce((s, p) => s + p.x, 0) / n
  const yBar = pts.reduce((s, p) => s + p.y, 0) / n
  const ssxy = pts.reduce((s, p) => s + (p.x - xBar) * (p.y - yBar), 0)
  const ssxx = pts.reduce((s, p) => s + (p.x - xBar) ** 2, 0)
  const slope = ssxx < 1e-12 ? 0 : ssxy / ssxx
  return { slope, intercept: yBar - slope * xBar }
}

// ── Scatter-plot SVG layout constants ───────────────────────────────────────
const SVG_W = 480
const SVG_H = 320
const PAD_L = 50, PAD_R = 20, PAD_T = 16, PAD_B = 44
const PLOT_W = SVG_W - PAD_L - PAD_R   // 410
const PLOT_H = SVG_H - PAD_T - PAD_B   // 260
const DX_MIN = 0, DX_MAX = 9
const DY_MIN = 0, DY_MAX = 9

const toSX = (x: number): number =>
  PAD_L + ((x - DX_MIN) / (DX_MAX - DX_MIN)) * PLOT_W
const toSY = (y: number): number =>
  PAD_T + PLOT_H - ((y - DY_MIN) / (DY_MAX - DY_MIN)) * PLOT_H
const fromSY = (sy: number): number =>
  DY_MIN + ((PAD_T + PLOT_H - sy) / PLOT_H) * (DY_MAX - DY_MIN)

const SCATTER_TICKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
const SCATTER_LBL_TICKS = [0, 2, 4, 6, 8]

// ── Arrowhead helper (used in VectorDiagram) ────────────────────────────────
// Returns polygon `pts` string for the arrowhead tip at (x2,y2),
// and (lx2, ly2): the line endpoint (arrowhead base) so the shaft
// doesn't overdraw the arrowhead.
function ap(
  x1: number, y1: number,
  x2: number, y2: number,
): { pts: string; lx2: number; ly2: number } {
  const dx = x2 - x1
  const dy = y2 - y1
  const len = Math.hypot(dx, dy)
  if (len < 1e-6) return { pts: '', lx2: x2, ly2: y2 }
  const ux = dx / len
  const uy = dy / len
  const ah = 10, aw = 5
  const bx = x2 - ah * ux
  const by = y2 - ah * uy
  return {
    pts: `${x2},${y2} ${bx - aw * uy},${by + aw * ux} ${bx + aw * uy},${by - aw * ux}`,
    lx2: bx,
    ly2: by,
  }
}

// ── Python / NumPy code snippet ──────────────────────────────────────────────
const SNIPPET = `import numpy as np

x = np.array([1.0, 2.1, 3.2, 4.5, 5.5, 6.3, 7.2, 8.1])
y = np.array([2.1, 3.5, 2.8, 5.3, 4.7, 6.1, 5.6, 7.4])

# 闭合解：slope / intercept
x_bar, y_bar = x.mean(), y.mean()
slope     = np.dot(x - x_bar, y - y_bar) / np.dot(x - x_bar, x - x_bar)
intercept = y_bar - slope * x_bar
y_hat     = slope * x + intercept
residuals = y - y_hat
rss       = np.dot(residuals, residuals)   # == sum(residuals**2)

# 矩阵视角：β̂ = (XᵀX)⁻¹ Xᵀy
X    = np.column_stack([np.ones_like(x), x])    # design matrix n×2
beta = np.linalg.inv(X.T @ X) @ X.T @ y         # [intercept, slope]

# Hat matrix H = X(XᵀX)⁻¹Xᵀ  →  ŷ = Hy，幂等投影矩阵
H = X @ np.linalg.inv(X.T @ X) @ X.T
print("H² ≈ H :", np.allclose(H @ H, H))            # True  ← 投影矩阵定义
print("Xᵀe ≈ 0:", np.allclose(X.T @ residuals, 0))  # True  ← 正规方程`

// ── ScatterView (interactive, data-space) ────────────────────────────────────
function ScatterView() {
  const [points, setPoints] = useState<Point[]>(INIT_POINTS)
  const [dragging, setDragging] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const { slope, intercept } = computeOLS(points)
  const lineAt = (x: number) => slope * x + intercept

  // Per-point fitted value and residual
  const augmented = points.map(p => ({
    id: p.id,
    x: p.x,
    yObs: p.y,
    yFit: lineAt(p.x),
    e: p.y - lineAt(p.x),
  }))
  const rss = augmented.reduce((s, r) => s + r.e * r.e, 0)

  function dataYFromClientY(clientY: number): number {
    const svg = svgRef.current
    if (!svg) return 0
    const rect = svg.getBoundingClientRect()
    const sy = ((clientY - rect.top) / rect.height) * SVG_H
    return Math.max(DY_MIN, Math.min(DY_MAX, fromSY(sy)))
  }

  // Fit line from x=0 to x=9; SVG viewport clips naturally
  const ly0 = lineAt(DX_MIN)
  const ly9 = lineAt(DX_MAX)

  return (
    <div>
      <svg
        ref={svgRef}
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{
          display: 'block',
          border: '1px solid #e4e6e9',
          borderRadius: 4,
          background: '#fff',
          touchAction: 'none',
          userSelect: 'none',
          cursor: dragging !== null ? 'grabbing' : 'default',
          maxWidth: '100%',
        }}
        onPointerMove={(e) => {
          if (dragging === null) return
          const newY = dataYFromClientY(e.clientY)
          setPoints(prev =>
            prev.map(p =>
              p.id === dragging
                ? { ...p, y: Math.round(newY * 10) / 10 }
                : p
            )
          )
        }}
        onPointerUp={() => setDragging(null)}
        onPointerLeave={() => setDragging(null)}
      >
        {/* Grid lines */}
        {SCATTER_TICKS.map(i => (
          <g key={`grid-${i}`}>
            <line
              x1={toSX(i)} y1={PAD_T}
              x2={toSX(i)} y2={PAD_T + PLOT_H}
              stroke="#e4e6e9" strokeWidth={0.8}
            />
            <line
              x1={PAD_L} y1={toSY(i)}
              x2={PAD_L + PLOT_W} y2={toSY(i)}
              stroke="#e4e6e9" strokeWidth={0.8}
            />
          </g>
        ))}

        {/* Axis borders */}
        <line
          x1={PAD_L} y1={PAD_T}
          x2={PAD_L} y2={PAD_T + PLOT_H}
          stroke="#c7cbd0" strokeWidth={1.2}
        />
        <line
          x1={PAD_L} y1={PAD_T + PLOT_H}
          x2={PAD_L + PLOT_W} y2={PAD_T + PLOT_H}
          stroke="#c7cbd0" strokeWidth={1.2}
        />

        {/* Tick labels */}
        {SCATTER_LBL_TICKS.map(i => (
          <g key={`lbl-${i}`}>
            <text
              x={toSX(i)} y={PAD_T + PLOT_H + 16}
              textAnchor="middle" fontSize={11} fill="#9aa1a9"
            >{i}</text>
            {i > 0 && (
              <text
                x={PAD_L - 7} y={toSY(i) + 4}
                textAnchor="end" fontSize={11} fill="#9aa1a9"
              >{i}</text>
            )}
          </g>
        ))}
        <text
          x={PAD_L + PLOT_W + 6} y={PAD_T + PLOT_H + 16}
          fontSize={12} fill="#5b6168" fontStyle="italic"
        >x</text>
        <text
          x={PAD_L - 9} y={PAD_T + 9}
          fontSize={12} fill="#5b6168" fontStyle="italic"
        >y</text>

        {/* Residual vertical segments (red dashed) */}
        {augmented.map(r => (
          <line
            key={`e-${r.id}`}
            x1={toSX(r.x)} y1={toSY(r.yObs)}
            x2={toSX(r.x)} y2={toSY(r.yFit)}
            stroke="#c0392b" strokeWidth={1.8} strokeDasharray="4 2"
          />
        ))}

        {/* Least-squares fit line (IKB blue) */}
        <line
          x1={toSX(DX_MIN)} y1={toSY(ly0)}
          x2={toSX(DX_MAX)} y2={toSY(ly9)}
          stroke="#002fa7" strokeWidth={2.2} strokeLinecap="round"
        />

        {/* ŷ label near the right end of the fit line */}
        <text
          x={toSX(8.4)} y={toSY(lineAt(8.4)) - 9}
          fontSize={13} fill="#002fa7" fontStyle="italic"
        >ŷ</text>

        {/* Data points — draggable (y-axis only) */}
        {points.map(p => (
          <circle
            key={`pt-${p.id}`}
            cx={toSX(p.x)} cy={toSY(p.y)} r={7}
            fill="#002fa7" fillOpacity={0.82}
            style={{ cursor: 'grab' }}
            onPointerDown={(e) => {
              ;(e.target as Element).setPointerCapture(e.pointerId)
              setDragging(p.id)
            }}
          />
        ))}
      </svg>

      {/* Live readout */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: '10px 28px',
        marginTop: 12, fontFamily: 'var(--mono)', fontSize: 13,
      }}>
        <span>
          <span style={{ color: '#5b6168' }}>slope </span>
          <strong style={{ color: '#002fa7' }}>{slope.toFixed(4)}</strong>
        </span>
        <span>
          <span style={{ color: '#5b6168' }}>intercept </span>
          <strong style={{ color: '#002fa7' }}>{intercept.toFixed(4)}</strong>
        </span>
        <span>
          <span style={{ color: '#5b6168' }}>RSS = </span>
          <strong style={{ color: '#c0392b' }}>{rss.toFixed(3)}</strong>
        </span>
      </div>
      <p style={{ fontSize: 13, color: '#5b6168', margin: '8px 0 0' }}>
        红色虚线 = residual（垂直距离）；上下拖动各点，OLS 直线和 RSS 实时更新。
      </p>
    </div>
  )
}

// ── VectorDiagram (static, vector-space view) ────────────────────────────────
function VectorDiagram() {
  // SVG canvas
  const VW = 420, VH = 280

  // Key positions (SVG pixel coordinates)
  const OX = 88,  OY = 218   // origin O  (on the col(X) plane)
  const YHX = 278, YHY = 200  // ŷ  (projection foot, on the plane)
  const YX  = 294, YY  = 82   // y  (observed vector, above the plane)

  // Arrows: O→ŷ, O→y, ŷ→y
  const ayhat = ap(OX,  OY,  YHX, YHY)
  const ay    = ap(OX,  OY,  YX,  YY)
  const ae    = ap(YHX, YHY, YX,  YY)

  // Right-angle mark at ŷ  (shows e ⊥ col(X))
  const RA = 12
  // plane direction unit vector (from O toward ŷ)
  const pDx = YHX - OX, pDy = YHY - OY
  const pLen = Math.hypot(pDx, pDy)
  const pux = pDx / pLen, puy = pDy / pLen
  // e direction unit vector (from ŷ toward y)
  const eDx = YX - YHX, eDy = YY - YHY
  const eLen = Math.hypot(eDx, eDy)
  const eux = eDx / eLen, euy = eDy / eLen
  // Three corners of the right-angle symbol
  const raAx = YHX + RA * pux,       raAy = YHY + RA * puy
  const raBx = raAx + RA * eux,      raBy = raAy + RA * euy
  const raCx = YHX  + RA * eux,      raCy = YHY  + RA * euy

  // Label positions
  const eMidX = (YHX + YX) / 2
  const eMidY = (YHY + YY) / 2

  return (
    <svg
      width={VW} height={VH}
      viewBox={`0 0 ${VW} ${VH}`}
      style={{
        display: 'block',
        border: '1px solid #e4e6e9',
        borderRadius: 4,
        background: '#fff',
        maxWidth: '100%',
      }}
    >
      {/* col(X) subspace — parallelogram, perspective-style */}
      <polygon
        points="24,242 384,242 354,192 54,192"
        fill="#e8edff" stroke="#002fa7"
        strokeWidth={1} strokeDasharray="6 3" opacity={0.85}
      />
      <text x={300} y={232} fontSize={12} fill="#002fa7" fontStyle="italic">col(X)</text>

      {/* Origin O */}
      <circle cx={OX} cy={OY} r={3.5} fill="#5b6168" />
      <text x={OX - 4} y={OY + 17} fontSize={12} fill="#5b6168">O</text>

      {/* Arrow O → ŷ (lies in the plane) */}
      <line
        x1={OX} y1={OY} x2={ayhat.lx2} y2={ayhat.ly2}
        stroke="#002fa7" strokeWidth={2} strokeLinecap="round"
      />
      <polygon points={ayhat.pts} fill="#002fa7" />

      {/* Arrow O → y (points above the plane) */}
      <line
        x1={OX} y1={OY} x2={ay.lx2} y2={ay.ly2}
        stroke="#002fa7" strokeWidth={2} strokeLinecap="round"
      />
      <polygon points={ay.pts} fill="#002fa7" />

      {/* Residual e = y − ŷ  (dashed red, perpendicular to col(X)) */}
      <line
        x1={YHX} y1={YHY} x2={ae.lx2} y2={ae.ly2}
        stroke="#c0392b" strokeWidth={2}
        strokeDasharray="5 3" strokeLinecap="round"
      />
      <polygon points={ae.pts} fill="#c0392b" />

      {/* Right-angle mark */}
      <polyline
        points={`${raAx},${raAy} ${raBx},${raBy} ${raCx},${raCy}`}
        fill="none" stroke="#5b6168" strokeWidth={1.2}
      />

      {/* Labels */}
      <text x={YHX + 10} y={YHY + 17}
        fontSize={13} fill="#002fa7" fontStyle="italic">ŷ = Hy</text>
      <text x={YX + 10} y={YY - 6}
        fontSize={13} fill="#002fa7" fontStyle="italic">y</text>
      <text x={eMidX + 16} y={eMidY + 4}
        fontSize={12} fill="#c0392b">e = y − ŷ</text>
      <text x={eMidX + 16} y={eMidY + 18}
        fontSize={11} fill="#5b6168">⊥ col(X)</text>
    </svg>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────
export function RegressionProjection() {
  const [view, setView] = useState<View>('scatter')

  const me = findChapter('regression-projection')!
  const { prev, next } = neighbors('regression-projection')

  function tabBtn(active: boolean) {
    return {
      padding: '8px 18px',
      fontFamily: 'var(--sans)' as const,
      fontSize: 14,
      fontWeight: active ? 600 : 400,
      background: active ? '#002fa7' : '#fff',
      color: active ? '#fff' : '#5b6168',
      border: `1px solid ${active ? '#002fa7' : '#c7cbd0'}`,
      borderRadius: 3,
      cursor: 'pointer' as const,
    }
  }

  return (
    <article className="page">

      {/* ── Masthead ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第四部分 · 正交、回归与投影
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          回归 = 投影
          <span className="zh-sub">最小二乘的几何真相</span>
        </h1>
        {me.prereq && (
          <div style={{
            display: 'inline-block', fontSize: 12, color: '#002fa7',
            background: '#e8edff', padding: '2px 10px', borderRadius: 3,
            marginBottom: 16,
          }}>
            前置：{me.prereq}
          </div>
        )}
        <p className="lede">
          你在硕士课里反复跑的 OLS，几何上就是一次<strong>正交投影</strong>——
          把响应向量 <code>y ∈ ℝⁿ</code> 投影到设计矩阵 <code>X</code> 的列空间
          <code>col(X)</code> 上。「最小化 RSS」和「把 y 投到 col(X) 上」
          是<strong>同一件事的两种说法</strong>。
        </p>
      </header>

      {/* ── Setup ── */}
      <section style={{ marginBottom: 32 }}>
        <p style={{ fontSize: 16, lineHeight: 1.75, maxWidth: '68ch', margin: '0 0 14px' }}>
          有 <em>n</em> 个观测值。把所有响应值堆成向量 <code>y ∈ ℝⁿ</code>，
          设计矩阵（含截距列）记为 <code>X</code>（<em>n</em>×<em>p</em>）。
          你能拟合的预测值 <code>ŷ = Xβ</code> 只能落在
          <code>col(X)</code>（一个 <em>p</em> 维子空间）里——你无法到达 col(X) 外的任何地方。
        </p>
        <p style={{ fontSize: 16, lineHeight: 1.75, maxWidth: '68ch', margin: '0 0 16px' }}>
          问题于是变成：<em>col(X) 里哪个点离 y 最近？</em>
          答案是 y 在 col(X) 上的正交投影。记 hat matrix
          <code>H = X(XᵀX)⁻¹Xᵀ</code>，则：
        </p>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 15,
          background: '#fafbfc', border: '1px solid #e4e6e9',
          padding: '14px 20px', borderRadius: 4,
          display: 'inline-block', marginBottom: 8,
        }}>
          ŷ = Hy &nbsp;·&nbsp; β̂ = (XᵀX)⁻¹Xᵀy &nbsp;·&nbsp; H² = H
        </div>
        <p style={{ fontSize: 14, color: '#5b6168', margin: '6px 0 0', maxWidth: '64ch' }}>
          <code>H² = H</code>（幂等性）是投影矩阵的代数定义：
          把一个已经在 col(X) 里的向量再投影一次，它不会移动。
        </p>
      </section>

      {/* ── View toggle ── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button style={tabBtn(view === 'scatter')} onClick={() => setView('scatter')}>
          数据空间 · 散点图（可拖动）
        </button>
        <button style={tabBtn(view === 'vector')} onClick={() => setView('vector')}>
          向量空间 · 投影图
        </button>
      </div>
      <p style={{ fontSize: 13, color: '#5b6168', margin: '0 0 14px' }}>
        {view === 'scatter'
          ? '数据空间：n 个观测点，OLS 拟合线（IKB 蓝），红色虚线 = residual（垂直距离）。'
          : '向量空间：y ∈ ℝⁿ 是目标向量，col(X) 是子空间，ŷ = Hy 是投影落脚点，e = y−ŷ ⊥ col(X)。'}
      </p>

      {/* ── Interactive section ── */}
      <section style={{ marginBottom: 16 }}>
        {view === 'scatter' ? <ScatterView /> : <VectorDiagram />}
      </section>

      {/* ── Normal equations callout ── */}
      <div className="verdict verdict--eq">
        <p>
          <strong>正规方程（normal equations）= 垂直条件。</strong>{' '}
          OLS 的一阶条件是 <code>Xᵀ(y − ŷ) = 0</code>——
          residual <code>e = y − ŷ</code> 必须与 <code>X</code> 的每一列正交。
          这在几何上正是「e 垂直于 col(X)」，也正是「ŷ 是正交投影」的定义。
          「统计里的 normal equations」和「几何里的投影垂直」是<strong>同一个等式</strong>。
        </p>
      </div>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            你硕士里反复做的最小二乘，几何上就是 18 节那张「投影 + 垂直残差」图——
            <strong>统计与几何在这里合流</strong>。
            「线性预测 + 正交投影」这套直觉一路延伸下去：
            线性层是线性预测器，把输入映射到输出子空间；
            注意力的 value 加权是一种投影读出（从 value 子空间里提取信息）；
            岭回归（第 27 节）是带正则化的投影，<code>H_λ = X(XᵀX + λI)⁻¹Xᵀ</code>，
            加了弹簧、缩了投影。
          </p>
          <p>
            Hat matrix <code>H² = H</code> 的幂等性在 Transformer 里也反复出现：
            投影头的权重矩阵组合、残差连接之后再归一化，本质上都是「投影到某个子空间再读出」。
            看懂这一页，你就同时看懂了 OLS、projection、normal equations 和 hat matrix——
            <strong>四个概念，同一张图。</strong>
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：标量视角 vs. 矩阵视角</h2>
        <CodeBlock code={SNIPPET} language="python" title="ols_projection.py" />
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
