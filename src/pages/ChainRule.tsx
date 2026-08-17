import { useState } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { ChRef } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'

// ─── colour tokens ────────────────────────────────────────────────
const IKB        = '#002fa7'  // Swiss-blue: forward-pass values
const RUST       = '#c75b39'  // rust: backward-pass / gradients
const GRAY       = '#b0b7c0'  // neutral edges
const LIGHT_IKB  = '#e8edf8'
const LIGHT_RUST = '#faeae5'

// ─── helpers ──────────────────────────────────────────────────────
function fmt(n: number, d = 3): string { return n.toFixed(d) }

type Phase = 'forward' | 'backward'

// ─── SVG node ─────────────────────────────────────────────────────
interface GraphNodeProps {
  cx: number; cy: number; r: number
  name: string; sub: string
  phase: Phase
}

function GraphNode({ cx, cy, r, name, sub, phase }: GraphNodeProps) {
  const fg = phase === 'forward' ? IKB : RUST
  const bg = phase === 'forward' ? LIGHT_IKB : LIGHT_RUST
  return (
    <g>
      <circle cx={cx} cy={cy} r={r} fill={bg} stroke={fg} strokeWidth={2.5} />
      <text
        x={cx} y={cy - 7}
        textAnchor="middle" fontSize={12} fontWeight="700"
        fill={fg} fontFamily="system-ui, sans-serif"
      >
        {name}
      </text>
      <text
        x={cx} y={cy + 9}
        textAnchor="middle" fontSize={10}
        fill={fg} fontFamily="monospace, monospace"
      >
        {sub}
      </text>
    </g>
  )
}

// ─── code snippet ─────────────────────────────────────────────────
const SNIPPET = `import numpy as np

# Scalar chain: loss = ½·(v·tanh(w·x))²
x, w, v = 1.0, 1.5, 2.0

# ── Forward pass ──────────────────────────────────────
a    = w * x                  # linear gate:   1.5
h    = np.tanh(a)             # nonlinearity:  0.905
y    = v * h                  # linear gate:   1.810
loss = 0.5 * y**2             # scalar loss:   1.639

# ── Backward pass (chain rule, right → left) ──────────
dL_dy = y                     # ∂L/∂y  (seed: d of ½y²)
dL_dh = dL_dy * v             # × ∂y/∂h = v
dL_da = dL_dh * (1 - h**2)   # × ∂h/∂a = 1 − tanh²  (tanh')
dL_dx = dL_da * w             # × ∂a/∂x = w  →  ≈ 0.981

# For vector layers: replace each scalar deriv with a Jacobian
# grad_in = J.T @ grad_out    (J[i,j] = ∂output_i/∂input_j)`

// ─── page ─────────────────────────────────────────────────────────
export function ChainRule() {
  const [xVal, setXVal] = useState(1.0)
  const [wVal, setWVal] = useState(1.5)
  const [phase, setPhase] = useState<Phase>('forward')

  const V = 2.0  // fixed weight for y = V·h

  // Forward pass
  const a    = wVal * xVal
  const h    = Math.tanh(a)
  const y    = V * h
  const loss = 0.5 * y * y

  // Local derivatives (one per edge in the graph)
  const da_dx = wVal               // ∂a/∂x = w
  const dh_da = 1 - h * h         // ∂h/∂a = tanh'(a) = 1 − tanh²(a)
  const dy_dh = V                  // ∂y/∂h = v
  const dl_dy = y                  // ∂L/∂y = d(½y²)/dy = y

  // Accumulated gradients (right → left)
  const dl_dh = dl_dy * dy_dh
  const dl_da = dl_dh * dh_da
  const dl_dx = dl_da * da_dx

  const isBack = phase === 'backward'

  // SVG layout constants
  const CY = 95   // node centre Y
  const R  = 34   // node radius

  // Node data for forward and backward display
  const nodeData = [
    { cx: 65,  name: 'x', fwd: fmt(xVal), bwd: fmt(dl_dx) },
    { cx: 195, name: 'a', fwd: fmt(a),    bwd: fmt(dl_da) },
    { cx: 335, name: 'h', fwd: fmt(h),    bwd: fmt(dl_dh) },
    { cx: 475, name: 'y', fwd: fmt(y),    bwd: fmt(dl_dy) },
    { cx: 610, name: 'L', fwd: fmt(loss), bwd: '1.000'    },
  ]

  // Edge metadata: midpoint X, operation label, local-derivative label
  const edgeMeta = [
    { mx: 130, op: '×w',   ld: `∂a/∂x=${fmt(da_dx, 2)}` },
    { mx: 265, op: 'tanh', ld: `∂h/∂a=${fmt(dh_da, 2)}` },
    { mx: 405, op: '×v',   ld: `∂y/∂h=${fmt(dy_dh, 2)}` },
    { mx: 542, op: '½y²',  ld: `∂L/∂y=${fmt(dl_dy, 2)}` },
  ]

  return (
      <ChapterShell
        slug="chain-rule"
        part="第六部分 · 学习：模型怎么变聪明"
        sub="backpropagation 就是链式法则走一遍计算图"
        lede={
          <>
        训练神经网络需要知道每个参数对 loss 的贡献。给定
        {' '}<code>loss = f₃(f₂(f₁(x)))</code>，
        链式法则 (chain rule) 说：
        {' '}<code>d loss/dx = f₃' · f₂' · f₁'</code>。
        Forward pass 从左到右算每个节点的值；
        backward pass 从右到左把每条边的 local derivative 乘回来——
        这就是 backpropagation (反向传播) 的全部。
        当节点是向量时，local derivative 变成 Jacobian 矩阵，
        backprop 变成一串 Jacobian-vector product。
          </>
        }
      >


      {/* ── controls ── */}
      <section className="controls">
        <div className="control">
          <label className="slider-row">
            <span>输入 <em>x</em></span>
            <input
              type="range" min={-2} max={2} step={0.05}
              value={xVal}
              onChange={(e) => setXVal(Number(e.target.value))}
            />
            <span className="param-val">{fmt(xVal, 2)}</span>
          </label>
        </div>
        <div className="control">
          <label className="slider-row">
            <span>权重 <em>w</em></span>
            <input
              type="range" min={0.2} max={3} step={0.05}
              value={wVal}
              onChange={(e) => setWVal(Number(e.target.value))}
            />
            <span className="param-val">{fmt(wVal, 2)}</span>
          </label>
        </div>
        <div
          className="control"
          style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}
        >
          <button
            onClick={() => setPhase('forward')}
            style={{
              padding: '0.4rem 1rem',
              border: `2px solid ${phase === 'forward' ? IKB : GRAY}`,
              background: phase === 'forward' ? LIGHT_IKB : '#fff',
              color: phase === 'forward' ? IKB : '#6b7280',
              fontWeight: 700,
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontFamily: 'system-ui, sans-serif',
            }}
          >
            ▶ 前向传播
          </button>
          <button
            onClick={() => setPhase('backward')}
            style={{
              padding: '0.4rem 1rem',
              border: `2px solid ${phase === 'backward' ? RUST : GRAY}`,
              background: phase === 'backward' ? LIGHT_RUST : '#fff',
              color: phase === 'backward' ? RUST : '#6b7280',
              fontWeight: 700,
              borderRadius: 6,
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontFamily: 'system-ui, sans-serif',
            }}
          >
            ◀ 反向传播
          </button>
        </div>
      </section>

      {/* ── computation graph ── */}
      <section className="stage" style={{ padding: '1.25rem 0 0.5rem' }}>
        <svg
          viewBox="0 0 660 200"
          style={{ width: '100%', maxWidth: 660, display: 'block', margin: '0 auto' }}
          aria-label="计算图：x → a（×w）→ h（tanh）→ y（×v）→ L（½y²）"
        >
          <defs>
            {/* unique id avoids collisions with any other SVG on the same page */}
            <marker id="arr-cr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill={GRAY} />
            </marker>
          </defs>

          {/* Edges (lines between nodes) */}
          {nodeData.slice(0, -1).map((n, i) => {
            const nxt = nodeData[i + 1]!
            return (
              <line
                key={i}
                x1={n.cx + R} y1={CY}
                x2={nxt.cx - R} y2={CY}
                stroke={GRAY} strokeWidth={1.8}
                markerEnd="url(#arr-cr)"
              />
            )
          })}

          {/* Operation labels: below each edge, always visible */}
          {edgeMeta.map(({ mx, op }) => (
            <text
              key={mx}
              x={mx} y={CY + R + 16}
              textAnchor="middle" fontSize={10}
              fill="#9aa1a9" fontFamily="monospace, monospace"
            >
              {op}
            </text>
          ))}

          {/* Local-derivative labels: above each edge, backward pass only */}
          {isBack && edgeMeta.map(({ mx, ld }) => (
            <text
              key={mx}
              x={mx} y={CY - R - 10}
              textAnchor="middle" fontSize={10}
              fill={RUST} fontFamily="monospace, monospace"
              fontStyle="italic"
            >
              {ld}
            </text>
          ))}

          {/* Nodes */}
          {nodeData.map((n) => (
            <GraphNode
              key={n.name}
              cx={n.cx} cy={CY} r={R}
              name={n.name}
              sub={isBack ? n.bwd : n.fwd}
              phase={phase}
            />
          ))}

          {/* Gradient-direction hint (backward only) */}
          {isBack && (
            <text
              x={330} y={185}
              textAnchor="middle" fontSize={11}
              fill={RUST} fontFamily="system-ui, sans-serif"
            >
              ← gradient flows right to left
            </text>
          )}
        </svg>
      </section>

      {/* ── live chain readout ── */}
      <div style={{
        background: isBack ? LIGHT_RUST : LIGHT_IKB,
        borderLeft: `4px solid ${isBack ? RUST : IKB}`,
        borderRadius: '0 6px 6px 0',
        padding: '0.9rem 1.25rem',
        margin: '0.25rem 0 1.5rem',
      }}>
        <pre style={{
          margin: 0,
          fontFamily: 'monospace, monospace',
          fontSize: '0.85rem',
          lineHeight: 1.9,
          color: isBack ? RUST : IKB,
          whiteSpace: 'pre-wrap',
          background: 'transparent',
          border: 'none',
          padding: 0,
        }}>
          {isBack
            ? `链式法则 (backward pass)              v = ${fmt(V, 1)} 固定

∂L/∂y = y                       = ${fmt(dl_dy)}
∂L/∂h = ∂L/∂y · v              = ${fmt(dl_dy)} × ${fmt(V, 1)}  = ${fmt(dl_dh)}
∂L/∂a = ∂L/∂h · (1 − tanh²a)  = ${fmt(dl_dh)} × ${fmt(dh_da)} = ${fmt(dl_da)}
∂L/∂x = ∂L/∂a · w              = ${fmt(dl_da)} × ${fmt(da_dx)} = ${fmt(dl_dx)}`
            : `前向传播 (forward pass)               v = ${fmt(V, 1)} 固定

a    = w · x          = ${fmt(wVal, 2)} × ${fmt(xVal, 2)} = ${fmt(a)}
h    = tanh(a)        = tanh(${fmt(a)}) = ${fmt(h)}
y    = v · h          = ${fmt(V, 1)} × ${fmt(h)}  = ${fmt(y)}
loss = ½ · y²         = ½ × ${fmt(y)}²  = ${fmt(loss)}`}
        </pre>
      </div>

      {/* ── Jacobian aside ── */}
      <section style={{
        margin: '0 0 1.5rem',
        padding: '1rem 1.25rem',
        background: '#f8f9fc',
        borderRadius: 8,
        borderLeft: `3px solid ${IKB}40`,
      }}>
        <h3 style={{ margin: '0 0 0.5rem', fontSize: '1rem', color: IKB }}>
          Jacobian：节点是向量时
        </h3>
        <p style={{ margin: '0 0 0.75rem', fontSize: '0.9rem', lineHeight: 1.65, color: '#1b1f24' }}>
          上面每个节点是标量，local derivative 是一个数。
          若节点是 <em>n</em> 维向量，local derivative 变成
          {' '}<strong>Jacobian 矩阵 J</strong>，
          其中 <code>J[i,j] = ∂output<sub>i</sub>/∂input<sub>j</sub></code>。
          Backprop 变成链式 Jacobian-vector product：
        </p>
        <pre style={{
          margin: '0 0 0.75rem',
          padding: '0.5rem 0.75rem',
          background: '#fff',
          border: `1px solid ${GRAY}`,
          borderRadius: 4,
          fontFamily: 'monospace, monospace',
          fontSize: '0.85rem',
          color: '#1b1f24',
          whiteSpace: 'pre-wrap',
        }}>
{`grad_in = Jᵀ @ grad_out    # Jacobian-vector product`}
        </pre>
        <p style={{ margin: '0 0 0.75rem', fontSize: '0.85rem', color: '#6b7280', lineHeight: 1.5 }}>
          例：线性层 <code>y = Wx</code>（m 维 → n 维），J = W（n×m 矩阵），
          grad_in = Wᵀ @ grad_out。下面的 2×2 Jacobian 示意了「每个输出对每个输入的偏导」：
        </p>
        {/* tiny 2×2 Jacobian display */}
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          fontFamily: 'monospace, monospace',
          fontSize: '0.85rem',
          color: IKB,
        }}>
          <span>J&nbsp;=</span>
          <span style={{
            borderLeft: `2.5px solid ${IKB}`,
            borderRight: `2.5px solid ${IKB}`,
            padding: '4px 10px',
          }}>
            <table style={{ borderSpacing: '10px 3px', margin: 0 }}>
              <tbody>
                <tr>
                  <td style={{ padding: 0 }}>W₁₁</td>
                  <td style={{ padding: 0 }}>W₁₂</td>
                </tr>
                <tr>
                  <td style={{ padding: 0 }}>W₂₁</td>
                  <td style={{ padding: 0 }}>W₂₂</td>
                </tr>
              </tbody>
            </table>
          </span>
          <span style={{ color: '#6b7280', fontSize: '0.8rem' }}>
            (∂output<sub>i</sub>/∂input<sub>j</sub>)
          </span>
        </div>
      </section>

      {/* ── verdict ── */}
      <section className="verdict">
        <p>
          把 <em>x</em> 拖到 ±2（或把 <em>w</em> 调大），
          切换到 backward 观察 <code>∂L/∂x</code>：
          当 <code>|a| ≫ 1</code>，tanh'(a) = 1 − tanh²(a) 趋近于 0，
          整条链路乘积被压扁——这就是
          <strong>梯度消失 (vanishing gradient)</strong> 的根源。
          反之，若某个 local derivative 远大于 1，乘积爆炸，即
          <strong>梯度爆炸 (exploding gradient)</strong>。
        </p>
      </section>

      {/* ── bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            一个 Transformer 几百层算子，训练时就是这张计算图的超大版：
            forward pass 算出 loss，backward pass 用链式法则把
            {' '}<code>∂loss/∂每个参数</code> 一层层乘回去——
            这就是 <code>loss.backward()</code> 干的事。
          </p>
          <p>
            梯度消失 / 爆炸 = 这串乘积太小 / 太大；
            残差连接 (residual connection) 和归一化 (LayerNorm，<ChRef slug="normalization" />)
            正是为了让这串乘积保持稳定，让梯度能从最后一层安全流回最前一层。
          </p>
        </div>
      </section>

      {/* ── code ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：标量 autograd 风格的 forward + backward</h2>
        <CodeBlock code={SNIPPET} language="python" title="chain_rule.py" />
      </section>

      {/* ── pager ── */}
      </ChapterShell>
  )
}
