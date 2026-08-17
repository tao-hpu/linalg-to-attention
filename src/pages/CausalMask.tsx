import { useState } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { ChRef, chNum } from '../components/ChRef'

// ── Colors ────────────────────────────────────────────────────────────────────
const IKB  = '#002fa7'
const RUST = '#c75b39'
// 品牌锈色当文字放白底上只有 4.16:1，差一点够不到 WCAG AA 的 4.5。
// 边框、柱子、色块用 RUST，凡是「字」一律用压深一档的 RUST_INK（5.16:1），肉眼几乎无差。
const RUST_INK = '#b34e2f'
const OK   = '#0a7d52'

// ── Sequence ──────────────────────────────────────────────────────────────────
const TOKENS = ['猫', '坐', '在', '垫子', '上'] as const
const N = TOKENS.length   // 5
const D = 4               // d_head

// 玩具 embedding：4 维，数值挑得让「猫/坐」「在/垫子」各自靠近，
// 这样 QKᵀ 出来的分数有结构可看，不是一团糊。
//
// SCALE 是必要的：坐标写在 ±1 区间读起来舒服，但 QKᵀ 是两次相乘，
// 分数会被压在 0～0.6 这么窄的区间里，softmax 出来几乎是均匀分布，
// 「掩码前后差别很大」这句话在图上就看不出来。放大 2.2 倍后分数拉到 0～3，
// 注意力才有明显的峰。
const SCALE = 2.2
const X_BASE: readonly (readonly number[])[] = [
  [ 1.0,  0.2, -0.3,  0.4],   // 猫
  [ 0.6,  0.9,  0.1, -0.2],   // 坐
  [-0.2,  0.7,  0.8,  0.1],   // 在
  [ 0.3, -0.1,  0.9,  0.7],   // 垫子
  [ 0.1,  0.4,  0.2,  1.0],   // 上
]
const X: number[][] = X_BASE.map((r) => r.map((v) => v * SCALE))

// 固定的 W_Q / W_K（真实模型里这是学出来的，这里写死好让数值可复现）
const W_Q: readonly (readonly number[])[] = [
  [ 0.9, -0.1,  0.3,  0.2],
  [ 0.2,  0.8, -0.2,  0.1],
  [-0.3,  0.4,  0.7, -0.1],
  [ 0.1,  0.2,  0.1,  0.9],
]
const W_K: readonly (readonly number[])[] = [
  [ 0.8,  0.2, -0.1,  0.1],
  [-0.1,  0.9,  0.2,  0.3],
  [ 0.4, -0.2,  0.8,  0.1],
  [ 0.2,  0.1,  0.3,  0.7],
]

function matmul(A: readonly (readonly number[])[], B: readonly (readonly number[])[]): number[][] {
  return A.map((row) =>
    B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)),
  )
}

const Q = matmul(X, W_Q)
const K = matmul(X, W_K)
const SQRT_D = Math.sqrt(D)

/** 缩放点积分数 S = QKᵀ/√d，(n×n)。第 i 行 = query i 对所有 key 的打分。 */
export const SCORES: number[][] = Q.map((qi) =>
  K.map((kj) => qi.reduce((s, v, t) => s + v * kj[t], 0) / SQRT_D),
)

export function softmaxRow(row: readonly number[]): number[] {
  const finite = row.filter((v) => Number.isFinite(v))
  const mx = finite.length ? Math.max(...finite) : 0
  const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - mx) : 0))
  const sum = exps.reduce((a, b) => a + b, 0)
  return sum === 0 ? row.map(() => 0) : exps.map((e) => e / sum)
}

// ── 三种模式 ───────────────────────────────────────────────────────────────────
// none  : 不掩码——每个 token 都能看到整句，包括它后面的词
// causal: 掩码加在 softmax 之前，把未来位置置 −∞
// after : 掩码加在 softmax 之后，把未来位置的权重抹成 0（常见的错误做法）
type Mode = 'none' | 'causal' | 'after'

const MODE_LABEL: Record<Mode, string> = {
  none:   '不掩码',
  causal: '因果掩码（softmax 之前置 −∞）',
  after:  'softmax 之后抹零',
}

/** 掩码后的分数矩阵：causal 模式下上三角为 −Infinity。 */
export function maskedScores(mode: Mode): number[][] {
  return SCORES.map((row, i) =>
    row.map((v, j) => (mode === 'causal' && j > i ? -Infinity : v)),
  )
}

/** 最终权重矩阵。after 模式先正常 softmax 再抹零，所以行和会掉下来。 */
export function weightsOf(mode: Mode): number[][] {
  const s = maskedScores(mode)
  const w = s.map(softmaxRow)
  if (mode !== 'after') return w
  return w.map((row, i) => row.map((v, j) => (j > i ? 0 : v)))
}

const S_MIN = Math.min(...SCORES.flat())
const S_MAX = Math.max(...SCORES.flat())

// 热图底色的透明度上限压在 0.45。再深下去，格子里那行 10.5px 的数字（#2c3036）
// 对比度就掉到 4.5:1 以下了；换成白字又要在中间那段浓度上翻车，两头不讨好。
// 0.45 封顶后一律用深色字，任何取值都稳过 AA，梯度也还看得出来。
const MAX_ALPHA = 0.45
function alphaOf(v: number, kind: 'score' | 'weight'): number {
  const t = kind === 'weight' ? v : (v - S_MIN) / (S_MAX - S_MIN)
  return Math.max(0, Math.min(1, t)) * MAX_ALPHA
}

// ── 分数 / 权重矩阵 ────────────────────────────────────────────────────────────
function Grid({
  title, sub, cells, mode, selected, onSelect, kind,
}: {
  title: string
  sub: string
  cells: number[][]
  mode: Mode
  selected: number
  onSelect: (i: number) => void
  kind: 'score' | 'weight'
}) {
  const CELL = 46
  const LABEL_W = 40

  return (
    <div style={{ flexShrink: 0 }}>
      <div style={{
        fontFamily: 'var(--mono)', fontWeight: 700, fontSize: 12, color: IKB,
        borderBottom: `2px solid ${IKB}`, paddingBottom: 5, marginBottom: 10,
        whiteSpace: 'nowrap',
      }}>
        {title}
      </div>

      {/* 列头 = key（被看的 token） */}
      <div style={{ display: 'flex', marginLeft: LABEL_W }}>
        {TOKENS.map((t, j) => (
          <div key={j} style={{ width: CELL, textAlign: 'center', fontSize: 11, color: 'var(--ink-soft)', marginBottom: 3 }}>
            {t}
          </div>
        ))}
      </div>

      {cells.map((row, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 2 }}>
          <button
            onClick={() => onSelect(i)}
            aria-pressed={i === selected}
            style={{
              width: LABEL_W, fontSize: 11, textAlign: 'right', paddingRight: 5,
              color: i === selected ? RUST_INK : 'var(--ink-soft)', fontWeight: i === selected ? 700 : 400,
              flexShrink: 0, background: 'none', border: 'none', cursor: 'pointer',
              fontFamily: 'var(--sans)',
            }}
          >
            {TOKENS[i]}
          </button>
          {row.map((v, j) => {
            const future = j > i
            // 「被堵死」的格子画成灰底淡字，一眼看出上三角。注意两张图的时机不同：
            // 分数矩阵只有 causal 模式才被改（−∞）；after 模式动的是 softmax 之后的权重，
            // 分数原封不动，所以这时左图不该变灰。
            const blocked = kind === 'score' ? mode === 'causal' && future : mode !== 'none' && future
            const bg = blocked ? '#f2f3f5' : `rgba(0,47,167,${alphaOf(v, kind).toFixed(3)})`
            const text = blocked && kind === 'score' ? '−∞' : v.toFixed(2)
            return (
              <div key={j} style={{
                width: CELL, height: CELL, background: bg,
                outline: i === selected ? `2px solid ${RUST}` : '1px solid #fff',
                outlineOffset: i === selected ? '-1px' : 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10.5, fontFamily: 'var(--mono)', flexShrink: 0,
                color: blocked ? '#666c74' : '#2c3036',
              }}>
                {text}
              </div>
            )
          })}
        </div>
      ))}

      <div style={{ display: 'flex', marginTop: 10, fontSize: 10.5, color: 'var(--ink-soft)', fontFamily: 'var(--mono)' }}>
        <div style={{ width: LABEL_W, textAlign: 'right', paddingRight: 5 }}>q↓</div>
        <div>k→</div>
      </div>
      <p style={{ fontSize: 11.5, color: 'var(--ink-soft)', margin: '6px 0 0', lineHeight: 1.45, maxWidth: LABEL_W + CELL * N }}>
        {sub}
      </p>
    </div>
  )
}

// ── 行和检查 ───────────────────────────────────────────────────────────────────
function RowSums({ weights, mode }: { weights: number[][]; mode: Mode }) {
  const sums = weights.map((r) => r.reduce((a, b) => a + b, 0))
  const allOne = sums.every((s) => Math.abs(s - 1) < 1e-9)
  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 6 }}>
        每行权重之和（注意力必须是一组概率，行和 = 1）
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {sums.map((s, i) => {
          const ok = Math.abs(s - 1) < 1e-9
          return (
            <span key={i} style={{
              fontFamily: 'var(--mono)', fontSize: 12.5, padding: '4px 10px',
              borderRadius: 3, border: `1px solid ${ok ? 'rgba(10,125,82,0.35)' : 'rgba(178,58,0,0.4)'}`,
              background: ok ? '#f0faf5' : '#fff6f1', color: ok ? OK : 'var(--warn)',
            }}>
              {TOKENS[i]} ：{s.toFixed(3)}
            </span>
          )
        })}
      </div>
      <p className={`verdict ${allOne ? 'verdict--eq' : 'verdict--neq'}`} style={{ marginTop: 14 }}>
        {mode === 'after'
          ? <>先 softmax 再抹零，被抹掉的那部分概率<em>凭空消失了</em>：行和从 1 掉到不足 1，
              抹掉的越多塌得越狠。第 1 个 token 只有自己一个合法候选，行和只剩{' '}
              {sums[0].toFixed(3)}；最后一个 token 没什么可抹的，还是 1.000。
              行和不为 1 意味着输出不再是 Value 的加权平均，而是被整体缩小了一个不确定的倍数——
              这就是掩码必须加在 softmax <em>之前</em> 的原因。</>
          : <>行和全部为 1.000：每个 token 的注意力仍是一组完整的概率分布，
              掩码只是把候选集合从「整句」缩到「自己和前文」。</>}
      </p>
    </div>
  )
}

// ── 选中行的信息来源 ───────────────────────────────────────────────────────────
function Provenance({ weights, i }: { weights: number[][]; i: number }) {
  const row = weights[i]
  const leak = row.reduce((s, v, j) => (j > i ? s + v : s), 0)
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginBottom: 8 }}>
        位置 {i + 1}「{TOKENS[i]}」的输出向量，由哪些 token 的 Value 加权而成
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        {row.map((w, j) => {
          const future = j > i
          return (
            <div key={j} style={{ textAlign: 'center', minWidth: 54 }}>
              <div style={{
                height: Math.max(2, w * 90), width: '100%',
                background: future ? RUST : IKB, opacity: w < 0.02 ? 0.25 : 1,
                borderRadius: '2px 2px 0 0',
              }} />
              <div style={{ fontSize: 11, marginTop: 4, color: future ? RUST_INK : 'var(--ink)' }}>{TOKENS[j]}</div>
              <div style={{ fontSize: 10.5, fontFamily: 'var(--mono)', color: 'var(--ink-soft)' }}>{w.toFixed(3)}</div>
            </div>
          )
        })}
      </div>
      <p style={{ fontSize: 14.5, marginTop: 14, lineHeight: 1.7 }}>
        {leak > 1e-9
          ? <>其中 <strong style={{ color: RUST_INK }}>{(leak * 100).toFixed(1)}%</strong> 来自
              位置 {i + 1} <em>之后</em> 的词。训练时「位置 {i + 1} 的下一个词」正是{' '}
              「{TOKENS[i + 1] ?? '句末'}」——模型直接把答案抄进了输入。</>
          : <>全部权重都落在位置 {i + 1} 及其之前，未来那几列是 0.000。
              位置 {i + 1} 的输出只依赖它自己和前文，可以放心用来预测
              「{TOKENS[i + 1] ?? '句末'}」。</>}
      </p>
    </div>
  )
}

// ── Code snippet ───────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_attention(Q, K, V):
    # Q, K, V : (n, d)
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)          # (n, n)  第 ${chNum('self-attention')} 节

    # 上三角（j > i，即「未来」）置 −∞。
    # np.triu(..., k=1) 取严格上三角，对角线本身保留——每个 token 看得见自己。
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)

    attn = softmax(scores, axis=-1)        # exp(−∞)=0，第 ${chNum('softmax')} 节
    return attn @ V                        # (n, d)

# 实现上常写 -1e9 而不是 -np.inf：fp16 下 inf 参与运算容易出 NaN，
# 一个足够大的负数在 softmax 里效果相同。

# PyTorch：F.scaled_dot_product_attention(q, k, v, is_causal=True)
#          内部就是这套掩码，且不会真的把 (n, n) 矩阵物化出来。`

// ── Main ──────────────────────────────────────────────────────────────────────
export function CausalMask() {
  const [mode, setMode] = useState<Mode>('causal')
  const [selected, setSelected] = useState(1)

  const scores = maskedScores(mode)
  const weights = weightsOf(mode)

  return (
    <ChapterShell
      slug="causal-mask"
      part="第八部分 · 合成：亲手拼出注意力"
      sub="softmax(QKᵀ/√d + M)·V"
      lede={
        <>
          上一节算出的注意力是<strong>全连通</strong>的：n×n 的分数矩阵里每个格子都有值，
          第 1 个 token 能看到第 5 个。可 GPT 的训练目标是「根据前文猜下一个词」——
          要是位置 1 能看见位置 2，那它猜位置 2 就等于抄答案，loss 会降到很低，
          模型却什么都没学会，生成时更是无从下手（那时候后文根本还不存在）。
          <strong>因果掩码</strong>解决这件事，手法简单到有点朴素：
          在 softmax 之前，把分数矩阵的上三角全部改成 −∞。
        </>
      }
    >

      {/* ── 控制区 ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">掩码方式</span>
            <span style={{ fontSize: 14, color: 'var(--ink-soft)' }}>
              切换三种做法，看分数矩阵和权重矩阵怎么变
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {(['none', 'causal', 'after'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  padding: '9px 18px',
                  border: `2px solid ${mode === m ? RUST : 'var(--line-strong)'}`,
                  background: mode === m ? 'rgba(199,91,57,0.07)' : '#fff',
                  color: mode === m ? RUST_INK : 'var(--ink)',
                  cursor: 'pointer', fontFamily: 'var(--sans)', fontSize: 15,
                  fontWeight: mode === m ? 700 : 400, borderRadius: 3,
                }}
              >
                {MODE_LABEL[m]}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── 两张矩阵 ── */}
      <section className="stage" style={{ gap: 28, alignItems: 'flex-start' }}>
        <Grid
          title="① 分数 S = QKᵀ/√d"
          sub={mode === 'causal'
            ? '上三角被换成 −∞。注意对角线还在：每个 token 看得见自己。'
            : mode === 'after'
              ? '这一路没动分数——错误做法是在下一步 softmax 之后才抹零。'
              : '每个格子 = 一个 query 对一个 key 的缩放点积，上三角就是「看未来」。'}
          cells={scores}
          mode={mode}
          selected={selected}
          onSelect={setSelected}
          kind="score"
        />
        <Grid
          title="② 权重 A = softmax(S)，逐行"
          sub={mode === 'causal'
            ? 'exp(−∞) = 0，上三角自动归零，剩下的权重重新分摊，行和仍是 1。'
            : mode === 'after'
              ? '先按整行 softmax 归一化，再把上三角抹成 0——行和因此不再是 1。'
              : '每行是一整句话上的概率分布，未来的词也分到了权重。'}
          cells={weights}
          mode={mode}
          selected={selected}
          onSelect={setSelected}
          kind="weight"
        />
      </section>

      {/* ── 行和检查 ── */}
      <section style={{ padding: '26px 0' }}>
        <h2 className="sec-h">行和还是 1 吗</h2>
        <RowSums weights={weights} mode={mode} />
      </section>

      {/* ── 信息来源 ── */}
      <section style={{ padding: '10px 0 26px' }}>
        <h2 className="sec-h">这个位置看见了谁</h2>
        <p style={{ fontSize: 15, color: 'var(--ink-soft)', margin: '0 0 16px' }}>
          点左侧任意一行的 token 名切换查看。<span style={{ color: RUST_INK }}>锈色柱</span>= 来自未来的信息。
        </p>
        <Provenance weights={weights} i={selected} />
      </section>

      {/* ── 为什么是 −∞ ── */}
      <section style={{ padding: '10px 0 26px' }}>
        <h2 className="sec-h">为什么是 −∞，而不是直接乘 0</h2>
        <p>
          掩码写成一个加法项：<code>softmax(QKᵀ/√d + M)</code>，其中 M 的上三角是 −∞、其余为 0。
          用加法而不是乘法，是因为它要在 softmax <em>之内</em> 生效：
          <ChRef slug="softmax" />讲过 softmax 先取 exp 再归一化，而 exp(−∞) = 0，
          于是未来位置的分子直接是 0，同时也不进分母。剩下的合法位置重新分摊掉全部概率，
          行和仍然精确等于 1。
        </p>
        <p>
          换成「softmax 之后抹零」就不成立了：分母里已经算进了未来位置的 exp，抹零只是把分子丢掉，
          总量守不住。上面切到第三种模式能直接看到行和塌下去，
          而且每一行塌的幅度还不一样——输出等于被乘上了一个逐行不同的随机缩放，
          后面的残差相加和归一化都会被它带偏。
        </p>
        <div className="note">
          <p>
            掩码不引入任何参数，也不改变形状：进去 (n×n)，出来还是 (n×n)。
            它只是把注意力的候选集合从「整句」削成「自己和前文」，
            所以<ChRef slug="transpose-shape" />那套形状推导一个字都不用改。
          </p>
        </div>
      </section>

      {/* ── 训练 vs 推理 ── */}
      <section style={{ padding: '10px 0 26px' }}>
        <h2 className="sec-h">一次前向，n 个训练信号</h2>
        <p>
          掩码真正的价值在训练效率。有了它，一句长度 n 的话喂进去跑<em>一次</em>前向，
          第 i 行的输出只依赖前 i 个词，于是能同时得到 n 个「预测下一个词」的训练样本：
          位置 1 预测第 2 个词、位置 2 预测第 3 个词……一直到位置 n−1。
          n 份<ChRef slug="cross-entropy" />的 loss 一起回传。
          没有掩码就只能把同一句话截成 n 段分别跑，慢 n 倍。
        </p>
        <div style={{
          display: 'flex', gap: 10, flexWrap: 'wrap', margin: '18px 0 0',
          fontFamily: 'var(--mono)', fontSize: 12.5,
        }}>
          {TOKENS.slice(0, N - 1).map((t, i) => (
            <span key={i} style={{
              padding: '6px 12px', border: `1px solid rgba(0,47,167,0.25)`,
              background: 'var(--ikb-soft)', color: IKB, borderRadius: 3,
            }}>
              位置 {i + 1}「{t}」→ 猜「{TOKENS[i + 1]}」
            </span>
          ))}
        </div>
        <p style={{ marginTop: 18 }}>
          推理时反过来：词是一个一个吐出来的，第 t 步只有前 t 个 token 存在，
          「看不见未来」不是靠掩码强加的，而是本来就没有未来。
          训练与推理因此对齐——这正是 decoder-only 架构成立的前提。
          也正因为第 t 步用到的 K、V 前 t−1 行与上一步完全相同，
          它们才能被缓存下来（KV Cache），把逐词生成的成本从 O(n²) 摊成每步 O(n)。
        </p>
      </section>

      <Bridge>
        <p>
          GPT 系列、LLaMA、Qwen 这些 <strong>decoder-only</strong> 模型，每一层的注意力都带因果掩码；
          BERT 那类 <strong>encoder</strong> 模型不带，因为它的任务是「完形填空」，
          本来就该看见两边。同一套 <code>softmax(QKᵀ/√d)·V</code>，
          加不加这个上三角，决定了它是个「生成模型」还是个「理解模型」。
        </p>
        <p>
          工程上有两处细节值得记：一是掩码值常写 <code>-1e9</code> 而非真正的 <code>-inf</code>，
          fp16 下 inf 参与减法会产生 NaN；二是 FlashAttention 这类实现根本不会把 (n×n)
          的分数矩阵物化到显存里，掩码是分块计算时顺手跳过的，
          <ChRef slug="self-attention" />说的 O(n²) 显存瓶颈，主要就是被这一步绕开的。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：加一个上三角</h2>
        <CodeBlock code={SNIPPET} language="python" title="causal_attention.py" />
      </section>
    </ChapterShell>
  )
}
