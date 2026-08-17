import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── SVG layout (square plot area → equal px/unit on both axes) ──────────────
const SVG_W = 480
const PAD_L = 52, PAD_R = 24, PAD_T = 24, PAD_B = 52
const PLOT_W = SVG_W - PAD_L - PAD_R   // 404 px
const PLOT_H = PLOT_W                  // square → same scale on x and y
const SVG_H = PLOT_H + PAD_T + PAD_B  // 480 px
const D_MIN = -4, D_MAX = 4
const SCALE = PLOT_W / (D_MAX - D_MIN) // 50.5 px per unit

// ── Fixed OLS (unconstrained) solution ──────────────────────────────────────
const BHAT_1 = 2.5
const BHAT_2 = 1.8
const BHAT_L2 = Math.sqrt(BHAT_1 ** 2 + BHAT_2 ** 2) // ‖β̂‖₂ ≈ 3.08（L2 圆包住 β̂ 的半径）
const BHAT_L1 = Math.abs(BHAT_1) + Math.abs(BHAT_2)  // ‖β̂‖₁ = 4.3（L1 菱形包住 β̂ 的预算）

// 约束失效（退回无约束 OLS 解）的阈值——按范数类型取，不能混用：
// L2 用 ‖β̂‖₂，L1 用 ‖β̂‖₁。两者不等，混用会让 L1 在 t∈[3.08,4.3) 时
// 误判为「已到无约束」，而 β̂ 此刻仍在菱形之外，画面自相矛盾。
function unconstrainedT(mode: Mode): number {
  return mode === 'L2' ? BHAT_L2 : BHAT_L1
}

// ── Anisotropic quadratic loss centered at β̂ ────────────────────────────────
// loss(β) = LA·(β₁−β̂₁)² + LB·(β₂−β̂₂)²
// LA > LB → ellipses narrower in β₁, taller in β₂
const LA = 1.8
const LB = 0.6

const SPARSE_EPS = 0.08  // |β| below this counts as zero

type Mode = 'L1' | 'L2'

// ── Coordinate transforms ────────────────────────────────────────────────────
function sx(x: number): number {
  return PAD_L + ((x - D_MIN) / (D_MAX - D_MIN)) * PLOT_W
}
function sy(y: number): number {
  return PAD_T + PLOT_H - ((y - D_MIN) / (D_MAX - D_MIN)) * PLOT_H
}

// ── Loss value at a point ────────────────────────────────────────────────────
function lossAt(b1: number, b2: number): number {
  return LA * (b1 - BHAT_1) ** 2 + LB * (b2 - BHAT_2) ** 2
}

// ── Find constrained solution by scanning the boundary ───────────────────────
function findSolution(t: number, mode: Mode): [number, number] {
  if (t >= unconstrainedT(mode)) return [BHAT_1, BHAT_2]

  if (mode === 'L2') {
    // Scan circle (t·cosθ, t·sinθ)
    const N = 3000
    let bestL = Infinity
    let best: [number, number] = [t, 0]
    for (let i = 0; i < N; i++) {
      const theta = (2 * Math.PI * i) / N
      const b1 = t * Math.cos(theta)
      const b2 = t * Math.sin(theta)
      const l = lossAt(b1, b2)
      if (l < bestL) { bestL = l; best = [b1, b2] }
    }
    return best
  }

  // L1 diamond: scan all 4 edges
  // E1: (t,0)→(0,t)  E2: (0,t)→(-t,0)  E3: (-t,0)→(0,-t)  E4: (0,-t)→(t,0)
  const edges: Array<(s: number) => [number, number]> = [
    (s) => [t * (1 - s), t * s],
    (s) => [-t * s, t * (1 - s)],
    (s) => [t * (s - 1), -t * s],
    (s) => [t * s, t * (s - 1)],
  ]
  const M = 1200
  let bestL = Infinity
  let best: [number, number] = [t, 0]
  for (const edge of edges) {
    for (let i = 0; i <= M; i++) {
      const [b1, b2] = edge(i / M)
      const l = lossAt(b1, b2)
      if (l < bestL) { bestL = l; best = [b1, b2] }
    }
  }
  return best
}

// ── Code snippet ─────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

# Ridge 闭合解: β_ridge = (XᵀX + λI)⁻¹ Xᵀy
# λI 让矩阵满秩、数值稳定；λ→0 退化为 OLS
beta_ridge = np.linalg.inv(X.T @ X + lam * np.eye(p)) @ X.T @ y

# Lasso: 无闭合解 —— 坐标下降 + 软阈值 (orthogonal X 特例)
# β_lasso_j = sign(β̂_j) · max(|β̂_j| − λ/2, 0)
def soft_threshold(beta_ols: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(beta_ols) * np.maximum(np.abs(beta_ols) - lam / 2, 0)

# 约束式 ↔ 惩罚式等价 (t = g(λ), 单调递减):
# min ‖y−Xβ‖² + λ‖β‖²   ⟺   min ‖y−Xβ‖² s.t. ‖β‖₂ ≤ t   ← 页面滑块的 t 就是这个半径
# min ‖y−Xβ‖² + λ‖β‖₁   ⟺   min ‖y−Xβ‖² s.t. ‖β‖₁ ≤ t

# Bayesian MAP 解读:
# Ridge ≡ MAP,  β ~ N(0, σ²/λ · I)      → Gaussian prior  → L2 惩罚
# Lasso ≡ MAP,  β ~ Laplace(0, 1/λ)     → Laplace prior   → L1 惩罚
# "正则化" 与 "贝叶斯先验" 是同一枚硬币的两面`

// ── Grid tick values ─────────────────────────────────────────────────────────
const TICKS     = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
const LBL_TICKS = [-4, -2, 0, 2, 4]

// ── Page component ───────────────────────────────────────────────────────────
export function Regularization() {
  const [mode, setMode] = useState<Mode>('L1')
  const [t, setT] = useState(1.2)

  const me = findChapter('regularization')!
  const { prev, next } = neighbors('regularization')

  const [s1, s2] = findSolution(t, mode)
  const isSparse1 = Math.abs(s1) < SPARSE_EPS
  const isSparse2 = Math.abs(s2) < SPARSE_EPS
  const isConstrained = t < unconstrainedT(mode)

  // Loss contour levels: smallest passes through solution, rest expand out
  const lSol = lossAt(s1, s2)
  const contourLevels: number[] = lSol > 0.05
    ? [lSol, lSol * 2.5, lSol * 6, lSol * 15, lSol * 35]
    : [0.08, 0.4, 1.2, 3.5, 9]

  // Ellipse semi-axes in SVG pixels for loss level L
  function ellipseAxes(L: number): { rx: number; ry: number } {
    return { rx: Math.sqrt(L / LA) * SCALE, ry: Math.sqrt(L / LB) * SCALE }
  }

  // Key SVG pixel positions
  const bSX = sx(BHAT_1), bSY = sy(BHAT_2)
  const solSX = sx(s1),   solSY = sy(s2)
  const origSX = sx(0),   origSY = sy(0)

  function modeBtn(m: Mode) {
    const active = mode === m
    return {
      padding: '7px 18px',
      fontFamily: 'var(--mono)' as const,
      fontSize: 13,
      fontWeight: active ? 600 : 400,
      background: active ? '#002fa7' : '#fff',
      color: active ? '#fff' : '#5b6168',
      border: `1px solid ${active ? '#002fa7' : '#c7cbd0'}`,
      borderRadius: 3 as const,
      cursor: 'pointer' as const,
    }
  }

  const badgeText = isSparse1 ? '稀疏! β₁=0' : '稀疏! β₂=0'

  return (
    <article className="page">

      {/* ── Masthead ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第六部分 · 学习：模型怎么变聪明
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          正则化：Ridge 与 Lasso
          <span className="zh-sub">约束几何、sparsity 与贝叶斯先验</span>
        </h1>
        <p className="lede">
          Ridge 和 Lasso 是两种最常见的正则化——给损失加一项惩罚，压住参数别乱长。
          这一节换一个视角：在<strong>参数空间</strong>（β₁, β₂）里画出来，
          看清为什么 L1 能让系数<strong>精确为零</strong>（sparsity），
          而 L2 只是均匀收缩。再往深一步：这两种正则化，
          恰好是两种<strong>贝叶斯先验</strong>下的 MAP 估计。
        </p>
      </header>

      {/* ── Core concept ── */}
      <section style={{ marginBottom: 28 }}>
        <p style={{ fontSize: 16, lineHeight: 1.75, maxWidth: '68ch', margin: '0 0 12px' }}>
          在 β 空间里，OLS 的目标是最小化 <code>‖y−Xβ‖²</code>——
          等高线是以 <strong>β̂</strong>（无约束最优解）为中心的<strong>椭圆</strong>。
          正则化加了一个以原点为中心的<strong>约束区域</strong>：
        </p>
        <ul style={{ fontSize: 15, lineHeight: 1.9, maxWidth: '64ch', paddingLeft: 22, margin: '0 0 12px' }}>
          <li><strong>Ridge (L2)</strong>：约束 β₁²+β₂² ≤ t — 一个<strong>圆</strong>，光滑无角。</li>
          <li><strong>Lasso (L1)</strong>：约束 |β₁|+|β₂| ≤ t — 一个<strong>菱形</strong>，角点在坐标轴上。</li>
        </ul>
        <p style={{ fontSize: 16, lineHeight: 1.75, maxWidth: '68ch', margin: 0 }}>
          解 = 不断扩张的 loss 椭圆<strong>第一次碰到约束边界</strong>的点。
          菱形的角点正好在坐标轴上，椭圆极容易「卡进」角点 → 某个 β = 0 → <strong>sparsity</strong>。
          圆没有角，接触点在圆周任意位置 → 系数只是收缩，不会精确为零。
        </p>
      </section>

      {/* ── Controls ── */}
      <section className="controls" style={{ flexWrap: 'wrap', gap: 20, alignItems: 'flex-end' }}>
        <div className="control">
          <div style={{ fontSize: 12, color: '#5b6168', marginBottom: 8 }}>约束类型</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button style={modeBtn('L1')} onClick={() => setMode('L1')}>
              L1 菱形 (Lasso)
            </button>
            <button style={modeBtn('L2')} onClick={() => setMode('L2')}>
              L2 圆 (Ridge)
            </button>
          </div>
        </div>
        <div className="control" style={{ minWidth: 260 }}>
          <label style={{ display: 'block', fontSize: 12, color: '#5b6168', marginBottom: 6 }}>
            约束半径 t =&nbsp;<strong style={{ fontFamily: 'var(--mono)' }}>{t.toFixed(2)}</strong>
            <span style={{ marginLeft: 10, color: '#9aa1a9', fontWeight: 400 }}>
              (λ 越大 → t 越小 → 约束越紧)
            </span>
          </label>
          <input
            type="range" min={0.3} max={3.4} step={0.05}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            style={{ width: '100%' }}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#9aa1a9', marginTop: 2 }}>
            <span>强正则 (λ 大)</span><span>无约束 (λ≈0)</span>
          </div>
        </div>
      </section>

      {/* ── Weight-space plot ── */}
      <section className="stage" style={{ display: 'block', marginBottom: 16 }}>
        <svg
          width={SVG_W} height={SVG_H}
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          style={{
            display: 'block',
            border: '1px solid #e4e6e9',
            borderRadius: 4,
            background: '#fff',
            maxWidth: '100%',
          }}
        >
          {/* Grid */}
          {TICKS.map(v => (
            <g key={`g${v}`}>
              <line x1={sx(v)} y1={PAD_T} x2={sx(v)} y2={PAD_T + PLOT_H}
                stroke="#e4e6e9" strokeWidth={0.7} />
              <line x1={PAD_L} y1={sy(v)} x2={PAD_L + PLOT_W} y2={sy(v)}
                stroke="#e4e6e9" strokeWidth={0.7} />
            </g>
          ))}

          {/* Axes — heavier, important: sparsity = landing exactly on these */}
          <line x1={PAD_L} y1={origSY} x2={PAD_L + PLOT_W} y2={origSY}
            stroke="#aab3be" strokeWidth={1.8} />
          <line x1={origSX} y1={PAD_T} x2={origSX} y2={PAD_T + PLOT_H}
            stroke="#aab3be" strokeWidth={1.8} />

          {/* Axis labels */}
          <text x={PAD_L + PLOT_W + 10} y={origSY + 4}
            fontSize={13} fill="#5b6168" fontStyle="italic">β₁</text>
          <text x={origSX - 3} y={PAD_T - 7}
            fontSize={13} fill="#5b6168" fontStyle="italic">β₂</text>

          {/* Tick labels */}
          {LBL_TICKS.map(v => (
            <g key={`l${v}`}>
              <text x={sx(v)} y={PAD_T + PLOT_H + 18}
                textAnchor="middle" fontSize={11} fill="#9aa1a9">{v}</text>
              {v !== 0 && (
                <text x={PAD_L - 7} y={sy(v) + 4}
                  textAnchor="end" fontSize={11} fill="#9aa1a9">{v}</text>
              )}
            </g>
          ))}

          {/* Loss contours — rust ellipses centered at β̂ */}
          {contourLevels.map((L, idx) => {
            const { rx, ry } = ellipseAxes(L)
            return (
              <ellipse
                key={`c${idx}`}
                cx={bSX} cy={bSY} rx={rx} ry={ry}
                fill="none"
                stroke="#c75b39"
                strokeWidth={idx === 0 ? 2.2 : 1}
                strokeOpacity={idx === 0 ? 0.9 : Math.max(0.18, 0.55 - idx * 0.09)}
                strokeDasharray={idx === 0 ? undefined : '5 3'}
              />
            )
          })}

          {/* Constraint region (IKB blue) */}
          {mode === 'L2' ? (
            <circle
              cx={origSX} cy={origSY}
              r={t * SCALE}
              fill="#002fa7" fillOpacity={0.07}
              stroke="#002fa7" strokeWidth={2.2}
            />
          ) : (
            <polygon
              points={`${sx(t)},${sy(0)} ${sx(0)},${sy(t)} ${sx(-t)},${sy(0)} ${sx(0)},${sy(-t)}`}
              fill="#002fa7" fillOpacity={0.07}
              stroke="#002fa7" strokeWidth={2.2}
            />
          )}

          {/* Dashed line: solution → β̂ (shows ellipse "reaching" toward β̂) */}
          {isConstrained && (
            <line
              x1={solSX} y1={solSY} x2={bSX} y2={bSY}
              stroke="#c75b39" strokeWidth={1.2}
              strokeDasharray="3 2" strokeOpacity={0.45}
            />
          )}

          {/* OLS β̂ */}
          <circle cx={bSX} cy={bSY} r={5}
            fill="#c75b39" fillOpacity={0.8} />
          <text x={bSX + 9} y={bSY - 7}
            fontSize={12} fill="#c75b39" fontStyle="italic">β̂ (OLS)</text>

          {/* Origin */}
          <circle cx={origSX} cy={origSY} r={3} fill="#5b6168" />
          <text x={origSX + 6} y={origSY - 5}
            fontSize={10} fill="#9aa1a9">O</text>

          {/* Constrained solution point */}
          {isConstrained && (
            <circle cx={solSX} cy={solSY} r={7}
              fill="#c75b39" fillOpacity={0.92} />
          )}

          {/* Sparsity badge — appears when solution lands on an axis */}
          {isConstrained && (isSparse1 || isSparse2) && (
            <g>
              <rect
                x={solSX - 38} y={solSY - 26}
                width={76} height={18} rx={3}
                fill="#c75b39"
              />
              <text
                x={solSX} y={solSY - 13}
                textAnchor="middle" fontSize={11}
                fill="#fff" fontWeight="bold"
              >{badgeText}</text>
            </g>
          )}
        </svg>

        <p style={{ fontSize: 12, color: '#5b6168', margin: '8px 0 0' }}>
          <span style={{ color: '#002fa7', fontWeight: 600 }}>蓝色</span> = 约束区域；
          <span style={{ color: '#c75b39', fontWeight: 600 }}>锈红实线</span> = loss 最小等高线（切约束边界处）；
          锈红虚线 = 较大 loss 等高线；<strong>实心圆</strong> = 正则化后的解。
          坐标轴（加粗）是 sparsity 的发生地。
        </p>
      </section>

      {/* ── Live readout ── */}
      <section className="readouts" style={{ flexWrap: 'wrap', gap: '10px 28px' }}>
        <span>
          <span style={{ color: '#5b6168' }}>β₁ = </span>
          <strong style={{ color: isSparse1 ? '#c75b39' : '#002fa7', fontFamily: 'var(--mono)' }}>
            {s1.toFixed(3)}
          </strong>
          {isSparse1 && (
            <span style={{ color: '#c75b39', marginLeft: 6, fontSize: 12 }}>← 稀疏</span>
          )}
        </span>
        <span>
          <span style={{ color: '#5b6168' }}>β₂ = </span>
          <strong style={{ color: isSparse2 ? '#c75b39' : '#002fa7', fontFamily: 'var(--mono)' }}>
            {s2.toFixed(3)}
          </strong>
          {isSparse2 && (
            <span style={{ color: '#c75b39', marginLeft: 6, fontSize: 12 }}>← 稀疏</span>
          )}
        </span>
        <span>
          <span style={{ color: '#5b6168' }}>
            {mode === 'L2' ? '‖β‖₂ = ' : '‖β‖₁ = '}
          </span>
          <strong style={{ color: '#002fa7', fontFamily: 'var(--mono)' }}>
            {mode === 'L2'
              ? Math.sqrt(s1 ** 2 + s2 ** 2).toFixed(3)
              : (Math.abs(s1) + Math.abs(s2)).toFixed(3)}
          </strong>
          <span style={{ color: '#9aa1a9', marginLeft: 4, fontSize: 12 }}>/ t = {t.toFixed(2)}</span>
        </span>
        <span style={{ color: '#5b6168', fontSize: 13 }}>
          {mode === 'L1'
            ? (isSparse1 || isSparse2
              ? 'Lasso 逼出稀疏'
              : 'Lasso — 解在边上，尚未稀疏')
            : 'Ridge 平滑收缩'}
        </span>
      </section>

      {/* ── Verdict ── */}
      <section className={`verdict ${mode === 'L1' && (isSparse1 || isSparse2) ? 'verdict--eq' : 'verdict--neq'}`}>
        {mode === 'L1' ? (
          isSparse1 || isSparse2 ? (
            <p>
              <strong>Lasso 逼出了 sparsity。</strong>{' '}
              解落在菱形的角点——正好在坐标轴上——
              {isSparse1 ? 'β₁' : 'β₂'}<strong>精确为零</strong>。
              这不是巧合：菱形的四个角点在坐标轴上，
              loss 椭圆从任何方向向角点收缩，都比切弧面更容易「卡住」。
              拖大 t（减小 λ）可看到解逐渐从角点滑向边上，sparsity 消失。
            </p>
          ) : (
            <p>
              <strong>Lasso (L1)。</strong>{' '}
              当前 t 较大，解落在菱形某条边上（非角点）——两个系数均非零。
              拖小 t（增大 λ）可观察解滑向角点，直到
              {Math.abs(s2) < Math.abs(s1) ? ' β₂' : ' β₁'} 精确归零。
            </p>
          )
        ) : (
          <p>
            <strong>Ridge 平滑收缩。</strong>{' '}
            解落在圆周上。圆是光滑曲面、无角点——
            loss 椭圆切圆时，接触点<strong>几乎不会</strong>落在坐标轴上，
            两个系数被同步向零收缩但均不为零。
            Ridge 不产生 sparsity，只是压缩 ‖β‖。
          </p>
        )}
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            <strong>Weight decay = L2 正则。</strong>{' '}
            AdamW 的那个「W」就是 weight decay：每步更新后把权重乘以略小于 1 的系数，
            等价于对 loss 加了 <code>λ‖W‖²</code> 惩罚。
            几乎所有现代 LLM 训练默认开启，让权重别长太大、提升泛化。
            L1 的 sparsity 用于剪枝（pruning）与模型压缩：
            把接近零的权重直接截断，可大幅缩减模型体积。
          </p>
          <p>
            还有更深的一层——从贝叶斯视角看，正则化其实就是给权重加了个先验：
            <strong>Ridge = 权重 Gaussian 先验下的 MAP 估计</strong>，
            <strong>Lasso = 权重 Laplace 先验下的 MAP 估计</strong>。
            Gaussian 先验把概率质量均匀地堆在零附近，对应圆形约束、L2 惩罚；
            Laplace 先验的尖峰形状把更多质量堆在零点，
            自然地把若干系数推到<em>精确</em>为零。
            「正则化」与「贝叶斯先验」是同一枚硬币的两面——
            统计与深度学习在这页又一次合流。
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：闭合解、软阈值、MAP 等价</h2>
        <CodeBlock code={SNIPPET} language="python" title="regularization.py" />
      </section>

      {/* ── Pager ── */}
      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一节</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : <span />}
        {next ? (
          <Link
            className="pager-link next"
            to={next.status === 'live' ? `/ch/${next.slug}` : '/'}
          >
            <span className="pager-dir">下一节 →</span>
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
