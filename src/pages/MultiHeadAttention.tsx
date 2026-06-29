import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Colors ────────────────────────────────────────────────────────────────────
const IKB  = '#002fa7'
const RUST = '#c75b39'

// Per-head accent: IKB-dominant, subtle hue shift per head
function headAccent(h: number): string {
  if (h === 1) return '#005a8b'   // teal-blue
  if (h === 2) return '#3d0e9c'   // violet-blue
  return IKB                       // head 0: standard IKB
}

// Attention cell background: rgba of the per-head accent
function cellBg(h: number, w: number): string {
  const a = (w * 0.9).toFixed(3)
  if (h === 1) return `rgba(0,90,139,${a})`
  if (h === 2) return `rgba(61,14,156,${a})`
  return `rgba(0,47,167,${a})`
}

// ── Sequence ──────────────────────────────────────────────────────────────────
const TOKENS = ['猫', '坐', '在', '垫子', '上'] as const
const N = TOKENS.length   // 5

// ── Dimensions ────────────────────────────────────────────────────────────────
const H      = 3          // number of heads
const D      = 6          // model dimension d
const D_HEAD = D / H      // 2  — per-head dimension d_head = d / h

// ── Preset raw scores (before softmax) ───────────────────────────────────────
//  Head 0 — local / diagonal   (each token ↔ self + immediate neighbors)
//  Head 1 — subject tracking   (all tokens → 猫, the sentence subject)
//  Head 2 — long-range/anti-diagonal (each token ↔ its distant counterpart)
const RAW_SCORES: number[][][] = [
  [
    [3.0, 1.5, 0.0, 0.0, 0.0],
    [1.5, 3.0, 1.5, 0.0, 0.0],
    [0.0, 1.5, 3.0, 1.5, 0.0],
    [0.0, 0.0, 1.5, 3.0, 1.5],
    [0.0, 0.0, 0.0, 1.5, 3.0],
  ],
  [
    [3.0, 0.5, 0.5, 0.5, 0.5],
    [3.0, 0.5, 0.5, 0.0, 0.0],
    [3.0, 0.0, 0.5, 0.0, 0.5],
    [3.0, 0.0, 0.0, 0.5, 0.0],
    [3.0, 0.0, 0.0, 0.5, 0.5],
  ],
  [
    [0.5, 0.5, 0.5, 0.5, 2.5],
    [0.5, 0.5, 0.5, 2.5, 0.5],
    [0.5, 0.5, 2.0, 0.5, 0.5],
    [0.5, 2.5, 0.5, 0.5, 0.5],
    [2.5, 0.5, 0.5, 0.5, 0.5],
  ],
]

// ── Row-wise softmax (numerically stable) ─────────────────────────────────────
function softmaxRow(row: number[]): number[] {
  const mx   = Math.max(...row)
  const exps = row.map((v) => Math.exp(v - mx))
  const sum  = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

// Pre-compute H × N × N attention weight matrices (all rows sum to 1)
const ATTN: number[][][] = RAW_SCORES.map((hs) => hs.map((r) => softmaxRow(r)))

// ── Head labels & notes ───────────────────────────────────────────────────────
const HEAD_LABEL: string[] = [
  'Head 0  ·  局部 / 相邻',
  'Head 1  ·  主语追踪',
  'Head 2  ·  长程 / 反向',
]
const HEAD_NOTE: string[] = [
  '对角线主导——每个 token 主要关注自身及相邻 token',
  '第一列主导——所有 token 的注意力集中在「猫」',
  '反对角线——token 关注序列中远端对应位置',
]

// ── HeadHeatmap ───────────────────────────────────────────────────────────────
function HeadHeatmap({
  headIdx,
  attn,
  selectedQuery,
}: {
  headIdx: number
  attn: number[][]
  selectedQuery: number
}) {
  const accent  = headAccent(headIdx)
  const CELL    = 38   // px per cell
  const LABEL_W = 38   // px for the query-label column

  return (
    <div style={{ flexShrink: 0 }}>
      {/* Head label */}
      <div style={{
        fontFamily: 'var(--mono)',
        fontWeight: 700,
        fontSize: '12px',
        color: accent,
        borderBottom: `2px solid ${accent}`,
        paddingBottom: '5px',
        marginBottom: '10px',
        whiteSpace: 'nowrap',
        letterSpacing: '0.01em',
      }}>
        {HEAD_LABEL[headIdx]}
      </div>

      {/* Column headers — key tokens */}
      <div style={{ display: 'flex', marginLeft: LABEL_W }}>
        {TOKENS.map((t, j) => (
          <div key={j} style={{
            width: CELL,
            textAlign: 'center',
            fontSize: '11px',
            color: '#999',
            marginBottom: '3px',
          }}>
            {t}
          </div>
        ))}
      </div>

      {/* Rows — one per query token */}
      {attn.map((row, i) => {
        const sel = i === selectedQuery
        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
            {/* Row label */}
            <div style={{
              width: LABEL_W,
              fontSize: '11px',
              textAlign: 'right',
              paddingRight: '5px',
              color: sel ? RUST : '#999',
              fontWeight: sel ? 700 : 400,
              flexShrink: 0,
            }}>
              {TOKENS[i]}
            </div>
            {/* Attention cells */}
            {row.map((w, j) => (
              <div key={j} style={{
                width: CELL,
                height: CELL,
                background: cellBg(headIdx, w),
                outline: sel ? `2px solid ${RUST}` : '1px solid transparent',
                outlineOffset: sel ? '-1px' : '0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '10px',
                fontFamily: 'var(--mono)',
                color: w > 0.52 ? '#fff' : '#2c3036',
                flexShrink: 0,
              }}>
                {w.toFixed(2)}
              </div>
            ))}
          </div>
        )
      })}

      {/* Short axis label */}
      <div style={{
        display: 'flex',
        marginTop: '6px',
        fontSize: '9.5px',
        color: '#bbb',
        fontFamily: 'var(--mono)',
      }}>
        <div style={{ width: LABEL_W, textAlign: 'right', paddingRight: '5px' }}>q↓</div>
        <div>k→</div>
      </div>

      {/* Pattern note */}
      <p style={{
        fontSize: '11.5px',
        color: 'var(--ink-soft)',
        margin: '6px 0 0',
        lineHeight: 1.45,
        maxWidth: `${LABEL_W + CELL * N}px`,
      }}>
        {HEAD_NOTE[headIdx]}
      </p>
    </div>
  )
}

// ── ShapePipeline — visual concat + W_O diagram ───────────────────────────────
function ShapePipeline() {
  const S   = 10            // px per dimension unit
  const mH  = N * S         // matrix height = n rows = 50 px
  const hW  = D_HEAD * S    // head output width = d_head cols = 20 px
  const fW  = D * S         // full output width = d cols = 60 px

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '18px',
      padding: '22px 24px',
      background: 'var(--ikb-soft)',
      border: '1px solid var(--line)',
      borderRadius: '4px',
      flexWrap: 'wrap',
      justifyContent: 'center',
      fontFamily: 'var(--mono)',
      fontSize: '11px',
    }}>

      {/* Three head output blocks */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
        <div style={{ display: 'flex', gap: '3px' }}>
          {[0, 1, 2].map((h) => (
            <div key={h} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px' }}>
              <div style={{ width: hW, height: mH, background: headAccent(h) }} />
              <span style={{ color: headAccent(h), fontSize: '9px' }}>h{h}</span>
            </div>
          ))}
        </div>
        <span style={{ color: '#666' }}>{H}×({N}×{D_HEAD})</span>
      </div>

      {/* concat arrow */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
        <span style={{ color: '#888', fontSize: '10px' }}>concat</span>
        <span style={{ color: IKB, fontSize: '22px', lineHeight: '1.2' }}>→</span>
      </div>

      {/* Concatenated matrix — colored stripe per head */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
        <div style={{
          width: fW,
          height: mH,
          background: `linear-gradient(to right,
            ${headAccent(0)} 33.33%,
            ${headAccent(1)} 33.33% 66.66%,
            ${headAccent(2)} 66.66%)`,
          border: `1px solid ${IKB}`,
          flexShrink: 0,
        }} />
        <span style={{ color: IKB }}>({N}×{D})</span>
      </div>

      {/* W_O arrow */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
        <span style={{ color: '#888', fontSize: '10px' }}>× W_O ({D}×{D})</span>
        <span style={{ color: IKB, fontSize: '22px', lineHeight: '1.2' }}>→</span>
      </div>

      {/* Final (n × d) */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
        <div style={{ width: fW, height: mH, background: IKB, flexShrink: 0 }} />
        <span style={{ color: IKB, fontWeight: 700 }}>({N}×{D})</span>
      </div>
    </div>
  )
}

// ── Code snippet ───────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def multi_head_attention(X, WQs, WKs, WVs, WO):
    # X   : (n, d)         输入序列
    # WQs : (h, d, d_head) h 个 Query 投影（WKs / WVs 同形）
    # WO  : (d, d)         输出投影，融合所有 head
    h      = len(WQs)
    d_head = WQs[0].shape[-1]   # = d // h

    head_outputs = []
    for i in range(h):
        Q      = X @ WQs[i]                      # (n, d_head)
        K      = X @ WKs[i]                      # (n, d_head)
        V      = X @ WVs[i]                      # (n, d_head)
        scores = Q @ K.T / np.sqrt(d_head)       # (n, n)  scaled dot-product
        attn   = softmax(scores)                  # (n, n)  每行 sum = 1 ✓
        head_outputs.append(attn @ V)             # (n, d_head)

    concat = np.concatenate(head_outputs, axis=-1)   # (n, d)
    return concat @ WO                               # (n, d) ✓

# PyTorch 等价：nn.MultiheadAttention(embed_dim=d, num_heads=h)`

// ── Main component ─────────────────────────────────────────────────────────────
export function MultiHeadAttention() {
  const [selectedQuery, setSelectedQuery] = useState(0)

  const me             = findChapter('multi-head')!
  const { prev, next } = neighbors('multi-head')

  return (
    <article className="page">

      {/* ── 页头 ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第八部分 · 合成：亲手拼出注意力
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          多头注意力
          <span className="zh-sub">为什么要「分头」看，每个 head 关注什么？</span>
        </h1>
        <p className="lede">
          单头注意力每次只能从一个角度「看」序列——所有 token 的 Q/K/V 投影共享同一个子空间。
          <strong>多头注意力（multi-head attention）</strong>并行运行 <code>{H}</code> 个独立的
          attention head，每个 head 拥有自己的 <code>W_Q</code>/<code>W_K</code>/<code>W_V</code>，
          在各自的低维 <strong>subspace</strong>（第 17/18 节）里独立计算注意力。
          最后，各 head 的输出沿特征维 <strong>concat</strong>，再乘输出投影 <code>W_O</code> 融合。
          计算量与单头大致相同，能同时捕捉的关系种类却成倍增加。
        </p>
      </header>

      {/* ── Token query selector ── */}
      <section className="controls" style={{ gridTemplateColumns: '1fr' }}>
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">Query</span>
            <span style={{ fontSize: '14px', color: 'var(--ink-soft)' }}>
              点击选择 query token（高亮{' '}
              <span style={{ color: RUST, fontWeight: 700 }}>锈色行</span>
              ），对比三个 head 各自如何分配注意力
            </span>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {TOKENS.map((t, i) => (
              <button
                key={i}
                onClick={() => setSelectedQuery(i)}
                style={{
                  padding: '9px 20px',
                  border: `2px solid ${selectedQuery === i ? RUST : 'var(--line-strong)'}`,
                  background: selectedQuery === i ? 'rgba(199,91,57,0.07)' : '#fff',
                  color: selectedQuery === i ? RUST : 'var(--ink)',
                  cursor: 'pointer',
                  fontFamily: 'var(--sans)',
                  fontSize: '16px',
                  fontWeight: selectedQuery === i ? 700 : 400,
                  borderRadius: '3px',
                  transition: 'border-color 0.1s, color 0.1s, background 0.1s',
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Three heatmaps side by side ── */}
      <section
        className="stage"
        style={{ alignItems: 'flex-start', gap: '28px' }}
      >
        {[0, 1, 2].map((h) => (
          <HeadHeatmap
            key={h}
            headIdx={h}
            attn={ATTN[h]}
            selectedQuery={selectedQuery}
          />
        ))}
      </section>

      {/* ── Shape readout: concat + W_O ── */}
      <section className="readouts" style={{ display: 'block', padding: '28px 0 20px' }}>
        <h2 className="sec-h">
          concat + W_O：把 {H} 个 head 的输出拼合再投影
        </h2>
        <ShapePipeline />
        <p style={{
          textAlign: 'center',
          fontSize: '13px',
          fontFamily: 'var(--mono)',
          color: 'var(--ink-soft)',
          marginTop: '12px',
        }}>
          {H} heads × ({N}×{D_HEAD})
          {'  ─concat─▶  '}({N}×{D})
          {`  ─×W_O(${D}×${D})─▶  `}({N}×{D})
        </p>
      </section>

      {/* ── Key insight ── */}
      <section className="verdict" style={{ borderColor: IKB, background: 'var(--ikb-soft)' }}>
        <p>
          <strong>同样的计算量，更丰富的表达力。</strong>
          设模型维度 <code>d = {D}</code>，单头注意力用 (<code>{D}×{D}</code>) 的
          大矩阵投影 Q/K/V。多头把它切成 <code>h = {H}</code> 个
          <code> d_head = d/h = {D_HEAD}</code> 维的小投影——每个 head 独立探索一个
          <strong> subspace</strong>，捕捉不同类型的 token 关系：
          上方 Head 0 的对角热图（局部相邻）、Head 1 的第一列热图（主语追踪）、
          Head 2 的反对角热图（长程依赖），三张图的模式截然不同。
          <code>W_O</code> 最终把这些子空间视角融合成一个统一的表征。
        </p>
        <p style={{ marginTop: '10px' }}>
          点击不同的 query token 对比三张热图——同一个 query，三个 head 做出的选择
          <em>完全不同</em>。这正是「分头」的全部意义。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            真实模型动辄 12–128 个 head（GPT-2 small 用 12）。
            多头 = 让模型在多个 <strong>subspace</strong>（第 17/18 节）里<em>同时</em>看不同关系，
            再用 <code>W_O</code> 融合。可解释性研究（interpretability）就是在给这些 head 找「它在干嘛」：
            <em>induction head</em>（归纳/复制前文规律）、名词-动词一致性 head、直接翻译 head……
            每个 head 各有分工。
          </p>
          <p>
            在 PyTorch 里一行搞定：
            <code>nn.MultiheadAttention(embed_dim=d, num_heads=h)</code>。
            它内部做的正是上面演示的：投影到 <code>h</code> 个 <code>d_head = d/h</code> 维 subspace，
            各自算注意力，concat，过 <code>W_O</code>。
          </p>
        </div>
      </section>

      {/* ── Code ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：多头注意力怎么算的</h2>
        <CodeBlock code={SNIPPET} language="python" title="multi_head_attention.py" />
      </section>

      {/* ── Pager ── */}
      <nav className="pager">
        {prev ? (
          <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
            <span className="pager-dir">← 上一章</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
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
