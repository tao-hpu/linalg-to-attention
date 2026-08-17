import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'

// ── design tokens ─────────────────────────────────────────────────────────────
const IKB = '#002fa7'
const RUST = '#c75b39'
const IKB_RGB = '0,47,167'
const RUST_RGB = '199,91,57'

// ── toy scale ─────────────────────────────────────────────────────────────────
// 一个小到形状能一眼读完的玩具模型：3 个 token、4 维、单头、6 词词表。
const TOKENS: readonly string[] = ['猫', '坐', '在']
const VOCAB: readonly string[] = ['猫', '坐', '在', '垫子', '上', '睡']
const N = TOKENS.length      // 序列长 = 3
const D = 4                  // d_model = d_head = 4（单头）
const V = VOCAB.length       // 词表大小 = 6
const FF = 4 * D             // FFN 隐层 = 16

// ── 示意 logits（只有上游数值是示意，softmax 这步是真算的）──────────────────────
const LOGITS: readonly number[] = [0.8, 1.1, 0.9, 3.0, 2.1, 1.3]

function softmax(xs: readonly number[]): number[] {
  const m = Math.max(...xs)
  const exps = xs.map((x) => Math.exp(x - m))
  const s = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / s)
}
function argmax(xs: readonly number[]): number {
  let best = 0
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i
  return best
}

const PROBS = softmax(LOGITS)         // 真·概率分布，每项 0~1，和为 1
const SAMPLED = argmax(PROBS)         // 贪心采样选中的词（示意里取最大概率）

// ── 流水线步骤定义 ──────────────────────────────────────────────────────────────
type Zone = 'in' | 'block' | 'out'
interface Ch { slug: string; label: string }
interface Step {
  label: string       // 阶段名
  shape: string       // 形状表达式（mono）
  role: string        // 一句话作用
  links: Ch[]         // 回链到教过它的章节
  zone: Zone
}

const STEPS: readonly Step[] = [
  {
    label: 'Tokenize 分词',
    shape: `["猫","坐","在"]  →  ids : (${N},)`,
    role: '把文字切成词表里的整数 id——模型只认数字，不认汉字。',
    links: [{ slug: 'bow-to-embedding', label: '第 02 节 · 从词袋到词向量' }],
    zone: 'in',
  },
  {
    label: 'Embedding 查表',
    shape: `one-hot (${N}×${V}) · W_E (${V}×${D}) = X (${N}×${D})`,
    role: '每个 id 取出一行 embedding——本质就是 one-hot 向量乘嵌入矩阵 W_E。',
    links: [
      { slug: 'vectors', label: '第 01 节 · 向量与坐标系' },
      { slug: 'bow-to-embedding', label: '第 02 节 · 从词袋到词向量' },
    ],
    zone: 'in',
  },
  {
    label: 'RMSNorm（进 block）',
    shape: `(${N}×${D})  →  RMSNorm  →  (${N}×${D})`,
    role: '进子层前先把每个 token 的向量尺度拉平，深层训练才稳。',
    links: [{ slug: 'normalization', label: '第 28 节 · 归一化' }],
    zone: 'block',
  },
  {
    label: '投影 Q, K, V',
    shape: `X·W_Q, X·W_K, X·W_V : (${N}×${D})·(${D}×${D}) = (${N}×${D}) ×3`,
    role: '同一个 X 用三个可学习矩阵投到三个子空间：查询、键、值。',
    links: [
      { slug: 'matrix-as-transform', label: '第 07 节 · 矩阵是变换' },
      { slug: 'matrix-mult', label: '第 08 节 · 矩阵乘法的几何' },
    ],
    zone: 'block',
  },
  {
    label: 'RoPE 位置编码',
    shape: `Q, K (${N}×${D})  →  R(mθ)·Q, R(nθ)·K  →  (${N}×${D})`,
    role:
      '注意力本身分不清词序，靠按位置旋转注入「它在第几位」。注意两点：RoPE 只转 Q 和 K，不碰 V；' +
      '而且它在每一层的注意力里都做一次，不是在输入端一次性加完——那是正弦绝对位置编码的做法。',
    links: [{ slug: 'rope', label: '第 33 节 · 位置编码与 RoPE' }],
    zone: 'block',
  },
  {
    label: '注意力 softmax(QKᵀ/√d)·V',
    shape: `QKᵀ (${N}×${N})  →  softmax  →  ·V = (${N}×${D})`,
    role: '唯一跨 token 的操作：每个 token 按相关性把别人的 Value 加权汇入自己。',
    links: [
      { slug: 'self-attention', label: '第 31 节 · 自注意力' },
      { slug: 'transpose-shape', label: '第 10 节 · 转置与形状' },
    ],
    zone: 'block',
  },
  {
    label: '⊕ 残差相加',
    shape: `x + Attn(·) : (${N}×${D}) + (${N}×${D}) = (${N}×${D})`,
    role: '注意力只贡献一个增量，加回残差流，原信息不被覆盖、梯度直通。',
    links: [{ slug: 'transformer-block', label: '第 34 节 · 一个 Transformer Block' }],
    zone: 'block',
  },
  {
    label: 'RMSNorm（进 FFN）',
    shape: `(${N}×${D})  →  RMSNorm  →  (${N}×${D})`,
    role: '第二次归一化，保证 FFN 拿到尺度一致的输入。',
    links: [{ slug: 'normalization', label: '第 28 节 · 归一化' }],
    zone: 'block',
  },
  {
    label: 'FFN 前馈（升维→降维）',
    shape: `(${N}×${D})·(${D}×${FF}) → 激活 → ·(${FF}×${D}) = (${N}×${D})`,
    role: '逐 token 独立加工：先升到 4d 的「思考空间」做非线性，再降回 d。',
    links: [{ slug: 'transformer-block', label: '第 34 节 · 一个 Transformer Block' }],
    zone: 'block',
  },
  {
    label: '⊕ 残差相加',
    shape: `x + FFN(·) : (${N}×${D}) + (${N}×${D}) = (${N}×${D})`,
    role: '第二条残差汇入。一个 block 走完 = 在 x 上叠了两个增量，形状不变。',
    links: [{ slug: 'transformer-block', label: '第 34 节 · 一个 Transformer Block' }],
    zone: 'block',
  },
  {
    label: '末位归一化 + Unembedding',
    shape: `x[${N - 1}] (1×${D}) · W_Eᵀ (${D}×${V}) = logits (1×${V})`,
    role: '取最后一个 token 的向量，乘回 W_E 的转置（权重绑定），得到整个词表的分数。',
    links: [
      { slug: 'bow-to-embedding', label: '第 02 节 · 从词袋到词向量' },
      { slug: 'transpose-shape', label: '第 10 节 · 转置与形状' },
    ],
    zone: 'out',
  },
  {
    label: 'Softmax → 下一词分布',
    shape: `logits (1×${V})  →  softmax  →  P(next) (1×${V})`,
    role: '把一排分数压成一组概率，和为 1——这就是模型对「下一个词」的预测分布。',
    links: [{ slug: 'softmax', label: '第 29 节 · Softmax 与概率分布' }],
    zone: 'out',
  },
  {
    label: '采样取词',
    shape: `从 P(next) 采样  →  「${VOCAB[SAMPLED]}」`,
    role: '按概率从分布里挑出一个词——温度 / top-k / top-p 在这里控制随机性。',
    links: [{ slug: 'sampling-decoding', label: '第 36 节 · 采样与解码' }],
    zone: 'out',
  },
]

const IN_IDX = [0, 1]
const BLOCK_IDX = [2, 3, 4, 5, 6, 7, 8, 9]
const OUT_IDX = [10, 11, 12]
const LAST = STEPS.length - 1

// ── 代码片段 ────────────────────────────────────────────────────────────────────
const SNIPPET = `\
import numpy as np

def forward(ids, W_E, blocks):
    # ids: (n,) — 一句话的 token id
    n = len(ids)

    # 1) embedding 查表：one-hot @ W_E，等价于按行取
    X = W_E[ids]                       # (n, d)

    # 2) 堆叠 N 个 Transformer block
    for blk in blocks:                 # 每个 block 形状不变：(n, d) → (n, d)
        h = rmsnorm(X)                 # (n, d)            第 28 节
        Q, K, Vv = h @ blk.Wq, h @ blk.Wk, h @ blk.Wv   # (n, d) 第 07/08 节
        Q, K = rope(Q), rope(K)        # 每层都转一次，只转 Q/K   第 33 节
        A = softmax(Q @ K.T / d**0.5)  # (n, n)            第 31/10/29 节
        X = X + A @ Vv                 # ⊕ 残差            第 34 节
        h = rmsnorm(X)                 # (n, d)            第 28 节
        h = gelu(h @ blk.W1) @ blk.W2  # (n, d)→(n,4d)→(n,d)
        X = X + h                      # ⊕ 残差

    # 3) 末位归一化 + unembedding（权重绑定 W_Eᵀ）
    logits = rmsnorm(X)[-1] @ W_E.T    # (1, vocab)        第 02 节

    # 4) softmax → 下一词分布
    probs = softmax(logits)            # (1, vocab)        第 29 节

    # 5) 采样取词
    next_id = sample(probs)            # temperature/top-k 第 36 节
    return next_id

# shapes:  ids:(n,) → X:(n,d) → … → logits:(vocab,) → probs:(vocab,) → 下一个词`

// ── 单个阶段方块 ────────────────────────────────────────────────────────────────
function StepChip({
  idx, active, onClick,
}: {
  idx: number
  active: boolean
  onClick: () => void
}) {
  const s = STEPS[idx]
  return (
    <button
      onClick={onClick}
      aria-current={active ? 'step' : undefined}
      style={{
        display: 'block', width: '100%', textAlign: 'left',
        padding: '8px 12px', cursor: 'pointer',
        border: `2px solid ${active ? RUST : `rgba(${IKB_RGB},0.2)`}`,
        borderRadius: 6,
        background: active ? `rgba(${RUST_RGB},0.08)` : '#fff',
        boxShadow: active ? `0 0 0 3px rgba(${RUST_RGB},0.16)` : 'none',
        transition: 'all 0.12s ease',
        fontFamily: 'inherit',
      }}
    >
      <span style={{
        fontSize: 13.5, fontWeight: 700,
        color: active ? RUST : IKB,
      }}>
        {s.label}
      </span>
      <span style={{
        display: 'block', marginTop: 2,
        fontSize: 10.5, fontFamily: 'monospace', color: '#888',
        overflowX: 'auto', whiteSpace: 'nowrap',
      }}>
        {s.shape}
      </span>
    </button>
  )
}

function Down() {
  return (
    <div style={{ textAlign: 'center', color: '#bbb', fontSize: 15, lineHeight: 1.2, userSelect: 'none' }}>
      ↓
    </div>
  )
}

// ── 主组件 ──────────────────────────────────────────────────────────────────────
export function ForwardPass() {
  const [active, setActive] = useState(0)
  const step = STEPS[active]

  const renderChips = (indices: number[]) =>
    indices.map((idx, k) => (
      <div key={idx}>
        {k > 0 && <Down />}
        <StepChip idx={idx} active={idx === active} onClick={() => setActive(idx)} />
      </div>
    ))

  const lede = (
    <>
      零件全都见过了：embedding、位置编码、Q/K/V 投影、注意力、残差、归一化、FFN、softmax、采样。
      这一节<strong>把它们首尾相接</strong>，跑通一次完整的前向传播——
      把「<strong>猫 坐 在</strong>」喂进一个玩具模型，看数据怎么从 token id
      流到「下一个词」的概率分布，<strong>每一步都标着矩阵形状</strong>，也标着是哪一节教的。
      点「下一步」或直接点某个阶段，跟着数据走一遍。
    </>
  )

  return (
    <ChapterShell
      slug="forward-pass"
      part="第十部分 · 终点：完整的前向传播"
      lede={lede}
    >
      {/* ── 控制区：stepper ── */}
      <section className="controls" style={{ gridTemplateColumns: '1fr' }}>
        <div className="control" style={{ borderTopColor: RUST }}>
          <div className="control-head">
            <span className="slot-tag" style={{ background: RUST }}>
              {active + 1}/{STEPS.length}
            </span>
            <span style={{ fontWeight: 700, fontSize: '0.95rem', color: '#1b1f24' }}>
              {step.label}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button
              onClick={() => setActive((i) => Math.max(0, i - 1))}
              disabled={active === 0}
              style={btnStyle(active === 0, false)}
            >
              ← 上一步
            </button>
            <button
              onClick={() => setActive((i) => Math.min(LAST, i + 1))}
              disabled={active === LAST}
              style={btnStyle(active === LAST, true)}
            >
              下一步 →
            </button>
            <button onClick={() => setActive(0)} style={btnStyle(false, false)}>
              ↺ 重置
            </button>
          </div>
          <p style={{ margin: '10px 0 0', fontSize: '0.82rem', color: '#5b6168', lineHeight: 1.5 }}>
            形状/数值用预先算好的<strong>示意</strong>值（softmax 那步是真算的），
            重点是看清零件如何首尾相接、形状如何对齐。
          </p>
        </div>
      </section>

      {/* ── 流水线全景 + 详情 ── */}
      <section className="stage" style={{ alignItems: 'flex-start', gap: '2rem' }}>
        {/* 左：竖向流水线 */}
        <div style={{ flex: '1 1 280px', minWidth: 260, maxWidth: 360 }}>
          <p style={{ fontSize: '0.82rem', color: '#5b6168', margin: '0 0 10px', textAlign: 'center' }}>
            embedding → 位置 → 注意力 → FFN → logits → softmax → 采样
          </p>

          <div style={{ fontSize: 11, fontWeight: 700, color: IKB, letterSpacing: '0.04em', margin: '0 0 6px' }}>
            ① 输入：文字 → 向量
          </div>
          {renderChips(IN_IDX)}

          <Down />

          {/* Transformer Block 包裹框：标明这一段都在一个 block 里 */}
          <div style={{
            border: `1.5px dashed rgba(${RUST_RGB},0.55)`,
            borderRadius: 8, padding: '8px 8px 10px', marginTop: 2,
            background: `rgba(${RUST_RGB},0.03)`,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: RUST, letterSpacing: '0.04em', margin: '0 0 6px' }}>
              ② 一个 Transformer Block · 第 34 节 · 重复 N 层
            </div>
            {renderChips(BLOCK_IDX)}
          </div>

          <Down />

          <div style={{ fontSize: 11, fontWeight: 700, color: IKB, letterSpacing: '0.04em', margin: '2px 0 6px' }}>
            ③ 输出：向量 → 下一个词
          </div>
          {renderChips(OUT_IDX)}
        </div>

        {/* 右：当前阶段详情 */}
        <div style={{ flex: '1 1 320px', minWidth: 280 }}>
          <div style={{
            border: `1px solid rgba(${RUST_RGB},0.35)`,
            borderRadius: 8, padding: '16px 18px', background: '#fdf8f7',
          }}>
            <div style={{
              fontSize: 11, fontWeight: 700, color: '#888',
              letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6,
            }}>
              第 {active + 1} 步
            </div>
            <h3 style={{ margin: '0 0 10px', fontSize: 18, color: RUST }}>
              {step.label}
            </h3>
            <div style={{
              fontFamily: 'monospace', fontSize: 13, color: '#333',
              background: '#fff', border: `1px solid rgba(${IKB_RGB},0.18)`,
              borderRadius: 5, padding: '8px 12px', marginBottom: 12,
              overflowX: 'auto', whiteSpace: 'nowrap',
            }}>
              {step.shape}
            </div>
            <p style={{ margin: '0 0 14px', fontSize: 14.5, color: '#2c3036', lineHeight: 1.65 }}>
              {step.role}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {step.links.map((c) => (
                <Link
                  key={c.slug}
                  to={`/ch/${c.slug}`}
                  style={{
                    fontSize: 12.5, fontWeight: 600, textDecoration: 'none',
                    color: IKB, background: `rgba(${IKB_RGB},0.08)`,
                    border: `1px solid rgba(${IKB_RGB},0.25)`,
                    borderRadius: 999, padding: '4px 12px',
                  }}
                >
                  {c.label} →
                </Link>
              ))}
            </div>
          </div>

          {/* 进入输出阶段时，提示去看下方分布 */}
          {step.zone === 'out' && (
            <p style={{ margin: '12px 2px 0', fontSize: 13, color: RUST, fontWeight: 600 }}>
              ↓ 看页面下方的「下一词概率分布」——模型选出了「{VOCAB[SAMPLED]}」。
            </p>
          )}
        </div>
      </section>

      {/* ── 形状总账 ── */}
      <section className="readouts" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
        <h2 className="sec-h">形状总账：一句话流过整个模型</h2>
        <div style={{
          fontFamily: 'monospace', fontSize: 12.5, lineHeight: 2.1,
          color: '#333', background: '#f6f8ff',
          border: `1px solid rgba(${IKB_RGB},0.18)`,
          borderRadius: 6, padding: '14px 20px', overflowX: 'auto',
        }}>
          {[
            [`ids        : (${N},)`, '   — 3 个 token 的整数 id'],
            [`X          : ${N}×${D}`, '   — embedding 查表（第 02 节）'],
            [`Q,K,V      : ${N}×${D}`, '   — 三路投影（第 07/08 节）'],
            [`QKᵀ        : ${N}×${N}`, '   — 注意力分数，token×token（第 10 节）'],
            [`attn·V     : ${N}×${D}`, '   — 加权 Value（第 31 节），形状回到 X'],
            [`FFN        : ${N}×${D}`, `   — 升到 ${FF} 再降回 ${D}（第 34 节）`],
            [`block out  : ${N}×${D}`, '   — 一层走完，形状不变 · ×N 层'],
            [`logits     : 1×${V}`, '   — 末位 · W_Eᵀ（权重绑定，第 02 节）'],
            [`P(next)    : 1×${V}`, '   — softmax 成概率分布（第 29 节）'],
            [`next token : 「${VOCAB[SAMPLED]}」`, '   — 采样取词（第 36 节）'],
          ].map(([shape, note], i) => (
            <div key={i}>
              <span style={{ color: IKB, fontWeight: 700 }}>{shape}</span>
              <span style={{ color: '#888' }}>{note}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── 下一词概率分布（高潮收口） ── */}
      <section className="stage" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
        <h2 className="sec-h" style={{ marginBottom: 4 }}>下一词概率分布（示意）</h2>
        <p style={{ fontSize: '0.9rem', color: '#444', margin: '0 0 16px' }}>
          输入「<strong>猫 坐 在</strong>」，模型在整个词表上给出一组概率。
          采样选中<strong style={{ color: RUST }}>「{VOCAB[SAMPLED]}」</strong>——
          于是下一个词就是它。把它接到句尾，再喂回去，就是「自回归」一个词一个词地生成。
        </p>

        <div style={{ width: '100%', maxWidth: 520 }}>
          {VOCAB.map((w, i) => {
            const p = PROBS[i]
            const isPick = i === SAMPLED
            return (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '5px 0' }}>
                <span style={{
                  width: 44, textAlign: 'right', flexShrink: 0,
                  fontSize: 14, fontWeight: isPick ? 700 : 500,
                  color: isPick ? RUST : '#333',
                }}>
                  {w}
                </span>
                <div style={{ flex: 1, background: '#f0f2f5', borderRadius: 4, overflow: 'hidden', height: 22 }}>
                  <div style={{
                    width: `${(p * 100).toFixed(1)}%`, height: '100%',
                    background: isPick ? RUST : `rgba(${IKB_RGB},0.55)`,
                    borderRadius: 4, transition: 'width 0.3s ease',
                  }} />
                </div>
                <span style={{
                  width: 52, flexShrink: 0, fontFamily: 'monospace', fontSize: 12.5,
                  color: isPick ? RUST : '#666', fontWeight: isPick ? 700 : 400,
                }}>
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            )
          })}
        </div>

        <p style={{ fontSize: 12, color: '#888', margin: '12px 0 0' }}>
          概率由示意 logits 经 softmax 真算得到，和为 {PROBS.reduce((a, b) => a + b, 0).toFixed(3)}。
          采样选中 = 概率最高的「{VOCAB[SAMPLED]}」（贪心解码；调高温度则可能选到别的词）。
        </p>
      </section>

      {/* ── verdict ── */}
      <section className="verdict verdict--eq">
        <p>
          <strong>你刚把整本书连成了一条线。</strong>
          一句话进来变成 <code>{N}×{D}</code> 的矩阵，流过 N 个形状不变的 block，
          最后一行乘 <code>W_Eᵀ</code> 摊开成 <code>1×{V}</code> 的 logits，softmax 成概率，采样吐出下一个词。
          中间没有任何魔法——每一步都是前面某一节里你亲手算过的矩阵运算。
        </p>
      </section>

      {/* ── 代码：完整前向传播 ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：一次完整的 forward</h2>
        <CodeBlock code={SNIPPET} language="python" title="forward.py" />
      </section>

      {/* ── bridge ── */}
      <Bridge>
        <p>
          这就是大模型推理时每生成一个词所做的全部事情：
          <strong>embedding → 位置编码 → N 层 block → unembedding → softmax → 采样</strong>。
          把吐出的词接回句尾、再跑一遍，循环往复，就是你在 ChatGPT 里看到的逐字蹦出来的文本。
          GPT、LLaMA、Qwen 的前向传播，骨架和上面这张图一模一样，只是 N 更大、d 更宽、词表更长。
        </p>
        <p>
          注意力不再是黑箱：它只是 <code>softmax(QKᵀ/√d)·V</code>，夹在残差和归一化之间，
          被 embedding 喂进、被 unembedding 读出。整台机器，你已经能从头看到尾。
        </p>
      </Bridge>

      {/* ── 终点：交棒 + ViT 钩子 ── */}
      <section style={{
        border: `1px solid ${IKB}`, borderRadius: 8,
        background: `rgba(${IKB_RGB},0.04)`, padding: '22px 24px', marginBottom: 32,
      }}>
        <div style={{
          fontSize: 12, fontWeight: 700, color: IKB,
          letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 12,
        }}>
          终点 · 也是起点
        </div>
        <p style={{ margin: '0 0 12px', fontSize: 15.5, color: '#2c3036', lineHeight: 1.7 }}>
          你已经到达这门预科课的终点。从「一个向量」一路搭到「跑通一次前向传播」——
          线性代数和概率的零件，在注意力里合流，又在这一页接成了完整的一台模型。
        </p>
        <p style={{ margin: '0 0 12px', fontSize: 15.5, color: '#2c3036', lineHeight: 1.7 }}>
          再多一句钩子：把上面流水线里的「词」换成「图像 patch」，同样这套注意力就是
          <strong> ViT（Vision Transformer）</strong>——于是模型能看图，多模态、视觉语言模型由此展开。
        </p>
        <p style={{ margin: '0 0 16px', fontSize: 15.5, color: '#2c3036', lineHeight: 1.7 }}>
          这些进阶内容——多模态、VLM / MLLM、真正动手训练与部署——在正课里展开。地基已经打好，去那边拧螺丝吧。
        </p>
        <a
          href="https://learn-llm.fim.ai"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            background: IKB, color: '#fff', fontWeight: 650, fontSize: 15,
            padding: '11px 20px', borderRadius: 4, textDecoration: 'none',
          }}
        >
          去正课 · learn-llm.fim.ai →
        </a>
      </section>
    </ChapterShell>
  )
}

// ── 按钮样式 ────────────────────────────────────────────────────────────────────
function btnStyle(disabled: boolean, primary: boolean): React.CSSProperties {
  return {
    padding: '7px 16px', fontWeight: 700, fontSize: 13.5,
    cursor: disabled ? 'not-allowed' : 'pointer',
    borderRadius: 6, fontFamily: 'inherit',
    border: `2px solid ${disabled ? '#ddd' : primary ? RUST : `rgba(${IKB_RGB},0.3)`}`,
    background: disabled ? '#f4f4f4' : primary ? RUST : '#fff',
    color: disabled ? '#aaa' : primary ? '#fff' : IKB,
    opacity: disabled ? 0.7 : 1,
    transition: 'all 0.1s ease',
  }
}
