import { useState } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { chNum } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'

// ─── types ───────────────────────────────────────────────────────────────────

type BoxKey = 'ln1' | 'attn' | 'add1' | 'ln2' | 'mlp' | 'add2'

interface BoxMeta {
  label: string
  shape: string
  desc: string
  /** pre-norm 与 post-norm 下说明不同的方块，在这里给出 post-norm 版本。 */
  descPost?: string
  isAdd: boolean
}

// ─── constants ────────────────────────────────────────────────────────────────

const IKB = '#002fa7'
const RUST = '#c75b39'

const BOX_META: Record<BoxKey, BoxMeta> = {
  ln1: {
    label: 'LayerNorm₁',
    shape: '(n × d) → (n × d)',
    desc:
      '对每个 token 独立做归一化：减均值、除标准差，再乘可学习参数 γ 和 β，' +
      '让各维度尺度归一。pre-norm 把它放在子层之前——梯度从残差相加处直通底层，' +
      `不受归一化阻断，深层训练更稳定（连接第 ${chNum('normalization')} 节）。`,
    descPost:
      '对每个 token 独立做归一化：减均值、除标准差，再乘可学习参数 γ 和 β。' +
      'post-norm 把它放在残差相加「之后」——梯度回传时必须先穿过 LayerNorm 才能' +
      `回到底层，路径没有 pre-norm 干净，所以原始论文那套深层网络需要 warm-up 才稳（连接第 ${chNum('normalization')} 节）。`,
    isAdd: false,
  },
  attn: {
    label: 'Multi-Head Attention',
    shape: '(n × d) → (n × d)',
    desc:
      '唯一跨 token 的操作：每个 token 的 Query 查询序列中所有 Key，' +
      '用 softmax 得到注意力权重后加权聚合 Value，写回 n × d 的残差流。' +
      `多头 = 分 h 个子空间并行关注，再拼回来（连接第 ${chNum('multi-head')} 节）。` +
      'Attention 负责在 token 之间搬运信息；MLP 负责逐 token 独立加工。',
    isAdd: false,
  },
  add1: {
    label: '⊕ 残差相加',
    shape: '(n × d) + (n × d) → (n × d)',
    desc:
      '把注意力输出加回输入流：x ← x + Attn(LN₁(x))。⊕ 是残差流的汇入点——' +
      '子层只贡献一个增量，原始信息不会被覆盖。' +
      `反向传播时梯度直穿 ⊕，不经过注意力矩阵，有效防止梯度消失（连接第 ${chNum('chain-rule')} 节链式法则）。`,
    descPost:
      '把注意力输出加回输入流：x ← x + Attn(x)，随后才轮到 LN₁——' +
      '整层写作 x ← LN₁(x + Attn(x))。⊕ 依然是残差流的汇入点，' +
      `但梯度穿过 ⊕ 之后还要再过一层归一化，不像 pre-norm 那样一路直通（连接第 ${chNum('chain-rule')} 节链式法则）。`,
    isAdd: true,
  },
  ln2: {
    label: 'LayerNorm₂',
    shape: '(n × d) → (n × d)',
    desc:
      '第二个 LayerNorm，放在 MLP 之前。原理与 LN₁ 完全相同，' +
      '只是作用在第一次残差相加之后的流上，保证 MLP 拿到尺度归一的输入，' +
      `让深层训练不崩（连接第 ${chNum('normalization')} 节）。`,
    descPost:
      '第二个 LayerNorm，post-norm 里它排在 MLP 的残差相加之后：x ← LN₂(x + MLP(x))。' +
      `原理与 LN₁ 完全相同，只是归一化的是已经汇入增量的残差流本身（连接第 ${chNum('normalization')} 节）。`,
    isAdd: false,
  },
  mlp: {
    label: 'MLP (Feed-Forward)',
    shape: '(n × d) → (n × 4d) → (n × d)',
    desc:
      '逐 token 独立处理，完全不跨 token。结构：Linear(d→4d) + GELU + Linear(4d→d)。' +
      '4d 扩展层是"思考空间"——做非线性变换、隐式存储从预训练语料学到的知识。' +
      'GELU 是平滑激活函数，负值区也有小梯度，比 ReLU 更适合 Transformer。',
    isAdd: false,
  },
  add2: {
    label: '⊕ 残差相加',
    shape: '(n × d) + (n × d) → (n × d)',
    desc:
      '将 MLP 输出加回流：x ← x + MLP(LN₂(x))。两次残差相加后，' +
      '整个 block 只是在输入 x 上叠加了两个增量。' +
      'N 层堆叠 = N 次叠加，表征从词面逐渐演变为深层语义。' +
      '残差流是信息的高速公路，每一层悄悄修改它，从不覆盖。',
    descPost:
      '将 MLP 输出加回流：x ← x + MLP(x)，再交给 LN₂ 归一化。' +
      '两次残差相加后，整个 block 同样只是在输入 x 上叠加了两个增量；' +
      '差别只在于归一化夹在相加的哪一侧。',
    isAdd: true,
  },
}

const PRE_NORM_SEQ: BoxKey[] = ['ln1', 'attn', 'add1', 'ln2', 'mlp', 'add2']
const POST_NORM_SEQ: BoxKey[] = ['attn', 'add1', 'ln1', 'mlp', 'add2', 'ln2']

const SNIPPET = `# PyTorch 风格，pre-norm Transformer block
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, heads)  # 第 ${chNum('multi-head')} 节
        self.ln2  = nn.LayerNorm(d)
        self.mlp  = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):                    # x: (batch, n, d)
        x = x + self.attn(self.ln1(x))      # 跨 token：注意力
        x = x + self.mlp(self.ln2(x))       # 逐 token：MLP
        return x                             # (batch, n, d)

# 堆叠 N 个 block = GPT
gpt = nn.Sequential(
    *[TransformerBlock(d=768, heads=12) for _ in range(12)]
)                                            # GPT-2 架构`

// ─── main component ───────────────────────────────────────────────────────────

export function TransformerBlock() {
  const [selected, setSelected] = useState<BoxKey | null>(null)
  const [preNorm, setPreNorm] = useState(true)

  const order = preNorm ? PRE_NORM_SEQ : POST_NORM_SEQ
  return (
      <ChapterShell
        slug="transformer-block"
        part="第八部分 · 合成：亲手拼出注意力"
        sub="残差流、LayerNorm、MLP——完整的一层"
        lede={
          <>
        注意力知道了，MLP 知道了，LayerNorm 知道了——现在把它们焊在一起。
        一个 Transformer block 是两个子层的叠加，两条 residual connection
        把它们串成整体；重复 N 次，就是一台完整的 GPT。
          </>
        }
      >

      {/* ── pre/post-norm toggle ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1b1f24' }}>
              LayerNorm 位置
            </span>
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['pre-norm', 'post-norm'] as const).map((mode) => {
                const active = preNorm ? mode === 'pre-norm' : mode === 'post-norm'
                return (
                  <button
                    key={mode}
                    onClick={() => {
                      setPreNorm(mode === 'pre-norm')
                      setSelected(null)
                    }}
                    style={{
                      padding: '5px 14px',
                      background: active ? IKB : '#f0f2f5',
                      color: active ? '#fff' : '#5b6168',
                      border: `1.5px solid ${active ? IKB : '#d0d4da'}`,
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      transition: 'background 0.1s',
                    }}
                  >
                    {mode}
                  </button>
                )
              })}
            </div>
          </div>
          <p style={{ margin: '8px 0 0', fontSize: '0.84rem', color: '#5b6168', lineHeight: 1.5 }}>
            {preNorm
              ? 'pre-norm（现代标准）：LayerNorm 在子层之前，梯度路径最干净，训练更稳定。GPT-2 起普遍使用。'
              : 'post-norm（原始 2017 论文）：LayerNorm 在残差相加之后，梯度路径更曲折，深层需要 warm-up 才稳。'}
          </p>
        </div>
      </section>

      {/* ── stage: block diagram + N-layers visual ── */}
      <section className="stage" style={{ display: 'flex', gap: '32px', alignItems: 'flex-start', flexWrap: 'wrap' }}>

        {/* Block diagram */}
        <div style={{ flex: '1 1 300px', minWidth: '280px' }}>
          <p style={{ fontSize: '0.82rem', color: '#5b6168', margin: '0 0 10px', textAlign: 'center' }}>
            点击方块展开说明 &nbsp;·&nbsp;
            <span style={{ color: RUST, fontWeight: 700 }}>┃</span> 残差流（跳过子层，直接相加）
          </p>

          {/* Diagram container */}
          <div style={{ position: 'relative', maxWidth: '460px', margin: '0 auto' }}>
            {/* Residual stream: vertical rust bar on the right */}
            <div
              style={{
                position: 'absolute',
                right: '6px',
                top: '32px',
                bottom: '32px',
                width: '4px',
                background: RUST,
                borderRadius: '2px',
                opacity: 0.8,
              }}
            />
            {/* Residual stream label */}
            <div
              style={{
                position: 'absolute',
                right: '12px',
                top: '50%',
                transform: 'translateY(-50%)',
                fontSize: '0.58rem',
                color: RUST,
                fontWeight: 700,
                writingMode: 'vertical-rl',
                textOrientation: 'mixed',
                letterSpacing: '0.12em',
                userSelect: 'none',
                opacity: 0.85,
              }}
            >
              residual stream
            </div>

            {/* Box column, padded away from the residual line */}
            <div style={{ paddingRight: '36px' }}>
              {/* Input node */}
              <div style={{ textAlign: 'center', fontSize: '0.82rem', color: '#5b6168', fontFamily: 'monospace', padding: '4px 0' }}>
                x &nbsp;<span style={{ opacity: 0.55 }}>(n × d)</span>
              </div>

              {order.map((key) => {
                const meta = BOX_META[key]
                const isSel = selected === key
                return (
                  <div key={key}>
                    {/* Arrow connector */}
                    <div style={{ textAlign: 'center', color: '#9aa1a9', fontSize: '1.05rem', lineHeight: '1.4', userSelect: 'none' }}>
                      ↓
                    </div>

                    {/* Clickable box */}
                    <button
                      onClick={() => setSelected(isSel ? null : key)}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        width: '100%',
                        padding: '10px 14px',
                        background: isSel ? RUST : meta.isAdd ? '#fff7f5' : IKB,
                        color: isSel ? '#fff' : meta.isAdd ? RUST : '#fff',
                        border: `2px solid ${meta.isAdd ? RUST : isSel ? RUST : IKB}`,
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontFamily: 'inherit',
                        fontSize: '0.9rem',
                        fontWeight: 600,
                        textAlign: 'left',
                        boxShadow: isSel ? `0 0 0 3px ${RUST}28` : 'none',
                        transition: 'background 0.1s, box-shadow 0.1s',
                      }}
                    >
                      <span>{meta.label}</span>
                      <code style={{ fontSize: '0.7rem', fontWeight: 400, opacity: 0.72 }}>
                        n × d
                      </code>
                    </button>

                    {/* Horizontal dashed connector to residual stream for ⊕ boxes */}
                    {meta.isAdd && (
                      <div
                        style={{
                          height: '2px',
                          marginTop: '-2px',
                          marginRight: '-30px',
                          background: `repeating-linear-gradient(90deg, ${RUST} 0, ${RUST} 5px, transparent 5px, transparent 10px)`,
                          opacity: 0.65,
                        }}
                      />
                    )}

                    {/* Expanded info panel */}
                    {isSel && (
                      <div
                        style={{
                          marginTop: '6px',
                          padding: '11px 14px',
                          background: '#fdf8f7',
                          border: `1px solid ${RUST}38`,
                          borderRadius: '5px',
                          fontSize: '0.86rem',
                          color: '#1b1f24',
                          lineHeight: 1.65,
                        }}
                      >
                        <code style={{ display: 'block', marginBottom: '6px', color: RUST, fontWeight: 600, fontSize: '0.79rem' }}>
                          {meta.shape}
                        </code>
                        <p style={{ margin: 0 }}>
                          {preNorm ? meta.desc : (meta.descPost ?? meta.desc)}
                        </p>
                      </div>
                    )}
                  </div>
                )
              })}

              {/* Arrow + output node */}
              <div style={{ textAlign: 'center', color: '#9aa1a9', fontSize: '1.05rem', lineHeight: '1.4', userSelect: 'none' }}>
                ↓
              </div>
              <div style={{ textAlign: 'center', fontSize: '0.82rem', color: '#5b6168', fontFamily: 'monospace', padding: '4px 0' }}>
                x &nbsp;<span style={{ opacity: 0.55 }}>(n × d)</span>
              </div>
            </div>
          </div>
        </div>

        {/* N-layers stacked visual */}
        <div style={{ flex: '0 0 200px', minWidth: '180px' }}>
          <p style={{ fontSize: '0.82rem', color: '#5b6168', margin: '0 0 10px', textAlign: 'center' }}>
            × N 层堆叠
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '3px' }}>
            {['Block 1', 'Block 2', '···', 'Block N'].map((label) => (
              <div
                key={label}
                style={{
                  width: '140px',
                  padding: '8px 0',
                  textAlign: 'center',
                  background: label === 'Block N' ? IKB : '#e8eeff',
                  color: label === 'Block N' ? '#fff' : IKB,
                  border: `1.5px solid ${IKB}`,
                  borderRadius: '4px',
                  fontSize: '0.82rem',
                  fontWeight: label === '···' ? 400 : 600,
                  opacity: label === '···' ? 0.4 : 1,
                }}
              >
                {label}
              </div>
            ))}
          </div>
          <p style={{ margin: '10px 0 0', fontSize: '0.76rem', color: '#5b6168', textAlign: 'center', lineHeight: 1.6 }}>
            GPT-2: N=12<br />
            GPT-3: N=96<br />
            Llama-3-70B: N=80
          </p>

          {/* Roles callout */}
          <div
            style={{
              marginTop: '16px',
              padding: '10px 12px',
              background: '#f5f7ff',
              border: `1px solid ${IKB}28`,
              borderRadius: '5px',
              fontSize: '0.8rem',
              color: '#1b1f24',
              lineHeight: 1.6,
            }}
          >
            <div><strong style={{ color: IKB }}>Attention</strong>：跨 token 混合信息</div>
            <div><strong style={{ color: IKB }}>MLP</strong>：逐 token 独立加工</div>
            <div style={{ marginTop: '4px', color: '#5b6168' }}>两者分工，层层递进</div>
          </div>
        </div>
      </section>

      {/* ── bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            把这个 block 摞 N 次（GPT-2 是 12 层，大模型上百层），前面加 embedding + 位置编码，
            后面接 <code>Linear → softmax</code> 预测下一个词——<strong>这就是一个完整的 GPT</strong>。
          </p>
          <p>
            残差流是信息的高速公路：Attention 跨 token 搬运信息，MLP 逐 token 深度加工，
            LayerNorm 稳住尺度。三个角色，各司其职，嵌套成圈，叠出深度。
            你已经把整台机器从零件拼到整机了。
          </p>
          <p>
            下一部分接到真实工程：LoRA 微调、采样解码、量化推理——
            知道了机器怎么运转，才真正知道在哪里拧螺丝、怎么拧得准。
          </p>
        </div>
      </section>

      {/* ── code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：pre-norm block 的 forward</h2>
        <CodeBlock code={SNIPPET} language="python" title="transformer_block.py" />
      </section>

      {/* ── pager ── */}
      </ChapterShell>
  )
}
