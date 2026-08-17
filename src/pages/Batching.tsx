import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Visual constants ──────────────────────────────────────────────
const D = 4        // input dim (fixed visual)
const H = 3        // output dim (fixed visual)
const T_MIN = 1
const T_MAX = 6

const IKB        = '#002fa7'
const IKB_SOFT   = '#eaf0ff'
const IKB_MID    = '#b3c6f7'
const IKB_HOT    = '#c7d9ff'
const ACCENT     = '#c75b39'
const ACCENT_SOFT = '#fdf0ec'
const ACCENT_MID  = '#e8a88e'
const ACCENT_HOT  = '#f9c4b0'
const GREEN      = '#2d7d32'
const GREEN_SOFT  = '#edf7ed'
const GREEN_MID   = '#9ecf9e'
const GREEN_HOT   = '#b8d4b8'

const CELL_W   = 28
const CELL_H   = 24
const LABEL_W  = 44

const TOKENS = ['我', '爱', '向量', '数学', '编程', '学习']

// ── Code snippet ──────────────────────────────────────────────────
const SNIPPET = `import torch

B = 4   # batch size：同时处理 4 句话
T = 6   # seq_len：每句话 6 个 token
d = 512 # 输入维度 (d_model)
h = 64  # 输出维度（比如单头的 d_k）

# 一批句子 = 一个三维张量
X = torch.randn(B, T, d)   # shape: (B, T, d)
W = torch.randn(d, h)      # 同一组权重，所有 token、所有句子共用

# 一行代码，处理全部 token × 全部句子
Y = X @ W                  # shape: (B, T, h)
# @ 自动在 B 和 T 两根轴上"骑行"——同一个 W 对每一行都做相同的事

# 对比：慢写法三层 for-loop，数值相同，GPU 不高兴
Y_slow = torch.zeros(B, T, h)
for b in range(B):
    for t in range(T):
        Y_slow[b, t] = X[b, t] @ W   # 每次只算一行 (d,) @ (d, h)

assert torch.allclose(Y, Y_slow, atol=1e-5)
# 输出: X: (4, 6, 512) -> Y: (4, 6, 64)`

// ── MatCell ───────────────────────────────────────────────────────
function MatCell({ value, bg, borderCol, color }: {
  value: string
  bg: string
  borderCol: string
  color: string
}) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: CELL_W,
      height: CELL_H,
      background: bg,
      border: `1.5px solid ${borderCol}`,
      borderRadius: 3,
      fontSize: 10,
      color,
      fontFamily: 'monospace',
      fontWeight: 500,
      margin: 2,
      flexShrink: 0,
    }}>
      {value}
    </span>
  )
}

// ── XBlock: T×D matrix with token row labels ──────────────────────
function XBlock({ T, activeRow, onEnter, onLeave, onToggle }: {
  T: number
  activeRow: number | null
  onEnter: (r: number) => void
  onLeave: () => void
  onToggle: (r: number) => void
}) {
  return (
    <div style={{ userSelect: 'none' }}>
      <div style={{
        fontSize: 11, color: '#9aa1a9', textAlign: 'center',
        marginBottom: 6, fontFamily: 'monospace',
      }}>
        X&nbsp;({T}×{D})
      </div>
      {Array.from({ length: T }, (_, r) => {
        const hot = activeRow === r
        return (
          // 用 button 而不是 div：桌面悬停、触屏点选、键盘 Tab+Enter 三种方式都能触发。
          <button
            key={r}
            type="button"
            aria-pressed={hot}
            aria-label={`第 ${r + 1} 行 token「${TOKENS[r]}」`}
            style={{
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              font: 'inherit',
              textAlign: 'left',
              border: 'none',
              background: hot ? 'rgba(0,47,167,0.08)' : 'transparent',
              borderRadius: 4,
              padding: '1px 2px',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
            onPointerEnter={(e) => { if (e.pointerType === 'mouse') onEnter(r) }}
            onPointerLeave={(e) => { if (e.pointerType === 'mouse') onLeave() }}
            onClick={() => onToggle(r)}
            onFocus={() => onEnter(r)}
            onBlur={onLeave}
          >
            <span style={{
              width: LABEL_W,
              fontSize: 14,
              textAlign: 'right',
              color: IKB,
              fontWeight: 700,
              paddingRight: 8,
              flexShrink: 0,
            }}>
              {TOKENS[r]}
            </span>
            {Array.from({ length: D }, (_, c) => (
              <MatCell
                key={c}
                value={`x${r + 1}${c + 1}`}
                bg={hot ? IKB_HOT : IKB_SOFT}
                borderCol={hot ? IKB : IKB_MID}
                color={IKB}
              />
            ))}
          </button>
        )
      })}
    </div>
  )
}

// ── WBlock: D×H weight matrix ─────────────────────────────────────
function WBlock({ active }: { active: boolean }) {
  return (
    <div style={{ userSelect: 'none' }}>
      <div style={{
        fontSize: 11, color: '#9aa1a9', textAlign: 'center',
        marginBottom: 6, fontFamily: 'monospace',
      }}>
        W&nbsp;({D}×{H})
      </div>
      {Array.from({ length: D }, (_, r) => (
        <div
          key={r}
          style={{
            display: 'flex',
            alignItems: 'center',
            padding: '1px 2px',
            background: active ? 'rgba(199,91,57,0.1)' : 'transparent',
            borderRadius: 4,
            transition: 'background 0.15s',
          }}
        >
          {Array.from({ length: H }, (_, c) => (
            <MatCell
              key={c}
              value={`w${r + 1}${c + 1}`}
              bg={active ? ACCENT_HOT : ACCENT_SOFT}
              borderCol={active ? ACCENT : ACCENT_MID}
              color={ACCENT}
            />
          ))}
        </div>
      ))}
      <div style={{
        fontSize: 10,
        color: ACCENT,
        textAlign: 'center',
        marginTop: 5,
        fontWeight: 600,
        opacity: active ? 1 : 0.45,
        transition: 'opacity 0.2s',
      }}>
        每行 token 都用同一个 W
      </div>
    </div>
  )
}

// ── YBlock: T×H output matrix ─────────────────────────────────────
function YBlock({ T, activeRow }: {
  T: number
  activeRow: number | null
}) {
  return (
    <div style={{ userSelect: 'none' }}>
      <div style={{
        fontSize: 11, color: '#9aa1a9', textAlign: 'center',
        marginBottom: 6, fontFamily: 'monospace',
      }}>
        Y&nbsp;({T}×{H})
      </div>
      {Array.from({ length: T }, (_, r) => {
        const hot = activeRow === r
        return (
          <div
            key={r}
            style={{
              display: 'flex',
              alignItems: 'center',
              padding: '1px 2px',
              background: hot ? 'rgba(45,125,50,0.08)' : 'transparent',
              borderRadius: 4,
              transition: 'background 0.15s',
            }}
          >
            {Array.from({ length: H }, (_, c) => (
              <MatCell
                key={c}
                value={`y${r + 1}${c + 1}`}
                bg={hot ? GREEN_HOT : GREEN_SOFT}
                borderCol={hot ? GREEN : GREEN_MID}
                color={GREEN}
              />
            ))}
          </div>
        )
      })}
    </div>
  )
}

// ── BatchDiagram: 3 offset sheets representing batch axis ─────────
function BatchDiagram({ T }: { T: number }) {
  const shW = 156
  const shH = Math.max(64, 18 + T * 30)
  const offset = 14
  const svgW = shW + 2 * offset
  const svgH = shH + 2 * offset

  // Drawn back-to-front: batch 2 first, batch 0 on top
  const sheets: { dx: number; dy: number; label: string; fillAlpha: number; strokeAlpha: number }[] = [
    { dx: 2 * offset, dy: 0,          label: 'batch 2', fillAlpha: 0.04, strokeAlpha: 0.35 },
    { dx: offset,     dy: offset,     label: 'batch 1', fillAlpha: 0.08, strokeAlpha: 0.60 },
    { dx: 0,          dy: 2 * offset, label: 'batch 0', fillAlpha: 0.14, strokeAlpha: 1.00 },
  ]

  return (
    <div>
      <div style={{
        fontSize: 11, color: '#9aa1a9', marginBottom: 8,
        fontFamily: 'monospace', textAlign: 'center',
      }}>
        X&nbsp;(B=3,&nbsp;T={T},&nbsp;d={D}) — 同一句 × 3 个句子叠在一起
      </div>
      <svg
        width={svgW}
        height={svgH}
        style={{ overflow: 'visible', display: 'block' }}
        aria-label="三张叠放的矩形，代表 batch 维度"
      >
        {sheets.map(({ dx, dy, label, fillAlpha, strokeAlpha }) => (
          <g key={label}>
            <rect
              x={dx} y={dy}
              width={shW} height={shH}
              fill={`rgba(0,47,167,${fillAlpha})`}
              stroke={IKB}
              strokeOpacity={strokeAlpha}
              strokeWidth={1.5}
              rx={4}
            />
            <text
              x={dx + 8} y={dy + 16}
              fontSize={10}
              fill={IKB}
              fillOpacity={Math.min(1, strokeAlpha + 0.15)}
              fontFamily="monospace"
            >
              {label}
            </text>
            <text
              x={dx + shW / 2} y={dy + shH / 2}
              fontSize={11}
              fill={IKB}
              fillOpacity={strokeAlpha * 0.65}
              fontFamily="monospace"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              ({T}×{D})
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

// ── ShapeReadout ──────────────────────────────────────────────────
function ShapeReadout({ T, showBatch }: { T: number; showBatch: boolean }) {
  const xShape = showBatch ? `(3, ${T}, ${D})` : `(${T}, ${D})`
  const yShape = showBatch ? `(3, ${T}, ${H})` : `(${T}, ${H})`

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      flexWrap: 'wrap',
      padding: '12px 16px',
      background: '#f7f8fa',
      borderRadius: 6,
      border: '1px solid #e0e4ea',
      marginTop: 20,
    }}>
      <span style={{ fontSize: 13, color: '#5b6168', flexShrink: 0 }}>形状：</span>
      <span style={{
        background: IKB, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
      }}>
        X: {xShape}
      </span>
      <span style={{ color: '#9aa1a9', fontSize: 14 }}>@</span>
      <span style={{
        background: ACCENT, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
      }}>
        W: ({D}, {H})
      </span>
      <span style={{ color: '#9aa1a9', fontSize: 14 }}>→</span>
      <span style={{
        background: GREEN, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 13, fontWeight: 700, fontFamily: 'monospace',
      }}>
        Y: {yShape}
      </span>
      {showBatch && (
        <span style={{ fontSize: 12, color: '#9aa1a9', fontFamily: 'monospace' }}>
          B=3 固定演示
        </span>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────
export function Batching() {
  const [T, setT] = useState(3)
  const [activeRow, setActiveRow] = useState<number | null>(null)
  const [showBatch, setShowBatch] = useState(false)

  // Clamp activeRow when T shrinks so W/Y don't stay highlighted for a phantom row
  const effectiveRow = activeRow !== null && activeRow < T ? activeRow : null

  const me = findChapter('batching')!
  const { prev, next } = neighbors('batching')

  return (
    <article className="page">

      {/* ── Masthead ─────────────────────────────────────────────── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第二部分 · 矩阵：一个动作
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>批量与张量<span className="zh-sub">一次处理一整句话怎么算？</span></h1>
        <p className="lede">
          神经网络不是逐词处理的——它把一整句话的所有 token <strong>堆成一个矩阵</strong>，
          然后用<strong>一次</strong> matmul 把所有行一起算完。
          同一组权重 W 乘以每一行 token，输出也是同样多的行。
          再加一个 batch 维，三维张量 <code>(B, T, d)</code> 就诞生了：
          B 句话、每句 T 个 token、每个 token 是 d 维向量。
          这就是 GPU 高效工作的核心逻辑。
        </p>
      </header>

      {/* ── Controls ─────────────────────────────────────────────── */}
      <section className="controls">

        {/* T stepper */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">token 数 T</span>
            <span style={{ fontSize: 13, color: '#5b6168' }}>
              句子里有多少个 token — 决定 X 的行数
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 10 }}>
            <button
              onClick={() => { setT((v) => Math.max(T_MIN, v - 1)); setActiveRow(null) }}
              disabled={T <= T_MIN}
              style={{
                width: 36, height: 36, borderRadius: 6,
                border: `1.5px solid ${T <= T_MIN ? '#ddd' : IKB}`,
                background: T <= T_MIN ? '#f4f4f4' : IKB_SOFT,
                color: T <= T_MIN ? '#bbb' : IKB,
                fontSize: 20, fontWeight: 700,
                cursor: T <= T_MIN ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}
              aria-label="减少一个 token"
            >
              −
            </button>
            <span style={{
              minWidth: 32, textAlign: 'center',
              fontSize: 24, fontWeight: 700, color: IKB, fontFamily: 'monospace',
            }}>
              {T}
            </span>
            <button
              onClick={() => { setT((v) => Math.min(T_MAX, v + 1)); setActiveRow(null) }}
              disabled={T >= T_MAX}
              style={{
                width: 36, height: 36, borderRadius: 6,
                border: `1.5px solid ${T >= T_MAX ? '#ddd' : IKB}`,
                background: T >= T_MAX ? '#f4f4f4' : IKB_SOFT,
                color: T >= T_MAX ? '#bbb' : IKB,
                fontSize: 20, fontWeight: 700,
                cursor: T >= T_MAX ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0,
              }}
              aria-label="增加一个 token"
            >
              +
            </button>
            <span style={{ fontSize: 13, color: '#9aa1a9' }}>
              {TOKENS.slice(0, T).join(' · ')}
            </span>
          </div>
        </div>

        {/* Batch toggle */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">batch 维</span>
            <span style={{ fontSize: 13, color: '#5b6168' }}>
              同时处理多句话，形状升为 3D
            </span>
          </div>
          <label style={{
            display: 'flex', alignItems: 'center', gap: 10,
            marginTop: 10, cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={showBatch}
              onChange={(e) => setShowBatch(e.target.checked)}
              style={{ width: 18, height: 18, accentColor: IKB, cursor: 'pointer' }}
            />
            <span style={{
              fontSize: 14,
              color: showBatch ? IKB : '#5b6168',
              fontWeight: showBatch ? 600 : 400,
              fontFamily: 'monospace',
              transition: 'color 0.15s',
            }}>
              {showBatch
                ? `X: (B=3, ${T}, ${D})  →  Y: (B=3, ${T}, ${H})`
                : `X: (${T}, ${D})  →  Y: (${T}, ${H})`}
            </span>
          </label>
        </div>
      </section>

      {/* ── Matrix visual ─────────────────────────────────────────── */}
      <section style={{ padding: '4px 0 16px' }}>
        <div style={{ overflowX: 'auto' }}>
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 18,
            padding: '12px 24px',
            minWidth: 'max-content',
          }}>
            <XBlock
              T={T}
              activeRow={effectiveRow}
              onEnter={(r) => setActiveRow(r)}
              onLeave={() => setActiveRow(null)}
              onToggle={(r) => setActiveRow((cur) => (cur === r ? null : r))}
            />
            <div style={{
              fontSize: 22, color: '#aab5c8', fontWeight: 300,
              alignSelf: 'center', paddingTop: 22, flexShrink: 0,
            }}>
              @
            </div>
            <WBlock active={effectiveRow !== null} />
            <div style={{
              fontSize: 22, color: '#aab5c8', fontWeight: 300,
              alignSelf: 'center', paddingTop: 22, flexShrink: 0,
            }}>
              =
            </div>
            <YBlock T={T} activeRow={effectiveRow} />
          </div>
        </div>

        <p style={{
          fontSize: 12, color: '#9aa1a9', margin: '0 24px 0',
          fontStyle: 'italic',
        }}>
          点一行 token（用鼠标的话悬停也行）——看 W 如何整体亮起，说明同一个 W 在处理这一行。
        </p>

        {/* Batch diagram */}
        {showBatch && (
          <div style={{
            margin: '20px 24px 0',
            padding: '18px 20px',
            background: IKB_SOFT,
            borderRadius: 8,
            border: `1px solid ${IKB_MID}`,
          }}>
            <p style={{
              margin: '0 0 14px',
              fontSize: 13, color: IKB, fontWeight: 600, lineHeight: 1.5,
            }}>
              沿 batch 轴叠加：B 句话各自是一个 ({T}×{D}) 矩阵，
              堆叠后变成 (3, {T}, {D}) 的三维张量。
              做 <code>X @ W</code> 时，matmul 自动在 B 轴上骑行——同一个 W 被所有句子共用。
            </p>
            <BatchDiagram T={T} />
          </div>
        )}

        {/* Shape readout */}
        <div style={{ margin: '0 24px' }}>
          <ShapeReadout T={T} showBatch={showBatch} />
        </div>
      </section>

      {/* ── Bridge ───────────────────────────────────────────────── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            真实的 forward pass 张量就是 <code>(batch, seq_len, d_model)</code>。
            一句话的所有 token 同时过同一组权重——<code>W_Q</code>、<code>W_K</code>、<code>W_V</code>
            都是对整个 <code>(T, d_model)</code> 矩阵做一次 matmul，不是逐词计算。
            注意力也是在 seq 维上一次性两两打分：<code>Q @ Kᵀ</code>
            输出 <code>(T, T)</code>，一步拿到所有 token 之间的相似度，没有任何循环。
          </p>
          <p>
            理解「加一根 batch / seq 轴、matmul 自动并行」，你就懂了为什么训练和推理
            始终在跟 <code>(B, T, d)</code> 这种形状打交道。也懂了 GPU 为什么关键：
            它天生擅长同时执行<strong>几千个独立的小矩阵乘法</strong>，
            而 batch matmul 的本质就是把这些并行机会全部打包给了硬件。
          </p>
        </div>
      </section>

      {/* ── Code block ───────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：一行 matmul vs 三层 for-loop</h2>
        <CodeBlock code={SNIPPET} language="python" title="batching.py" />
      </section>

      {/* ── Pager ────────────────────────────────────────────────── */}
      <nav className="pager">
        {prev
          ? (
            <Link
              className="pager-link prev"
              to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
            >
              <span className="pager-dir">← 上一节</span>
              <span className="pager-title">{prev.num} {prev.title}</span>
            </Link>
          )
          : <span />}
        {next
          ? (
            <Link
              className="pager-link next"
              to={next.status === 'live' ? `/ch/${next.slug}` : '/'}
            >
              <span className="pager-dir">下一节 →</span>
              <span className="pager-title">
                {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
              </span>
            </Link>
          )
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
