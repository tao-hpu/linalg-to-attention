import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── 颜色常量 ──────────────────────────────────────────────────────────────────
const IKB = '#002fa7'   // International Klein Blue — 概率柱
const RUST = '#c75b39'  // rust — argmax 柱 / temperature 控件

// ── 柱状图高度 ────────────────────────────────────────────────────────────────
const BAR_MAX_PX = 160  // 柱的最大像素高度（对应 prob = 1.0）

// ── Token 初始数据 ─────────────────────────────────────────────────────────────
interface TokenDef {
  readonly zh: string
  readonly defaultLogit: number
}

const TOKEN_DEFS: readonly TokenDef[] = [
  { zh: '猫', defaultLogit:  2.0 },
  { zh: '狗', defaultLogit:  1.5 },
  { zh: '鱼', defaultLogit:  0.5 },
  { zh: '跑', defaultLogit: -0.5 },
  { zh: '的', defaultLogit: -1.0 },
  { zh: '好', defaultLogit:  0.0 },
]

const N = TOKEN_DEFS.length  // 6

// ── Softmax 计算（减去 max → 数值稳定；shift-invariance 保证结果不变）──────────
interface SoftmaxResult {
  exps: readonly number[]
  Z: number
  probs: readonly number[]
}

function computeSoftmax(logits: readonly number[], T: number): SoftmaxResult {
  const maxZ = Math.max(...logits)
  const exps = logits.map((z) => Math.exp((z - maxZ) / T))
  const Z = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map((e) => e / Z)
  return { exps, Z, probs }
}

// ── Code snippet ───────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

def softmax(z, T=1.0):
    # 减去最大值防止指数溢出（shift-invariance 保证结果不变）
    e = np.exp((z - z.max()) / T)
    return e / e.sum()

z = np.array([2.0, 1.5, 0.5, -0.5, -1.0, 0.0])
print(softmax(z))           # T=1，正常锐度
print(softmax(z, T=0.3))   # 低温，近乎 one-hot
print(softmax(z, T=5.0))   # 高温，趋于均匀 1/6

# 验证：概率之和恒为 1
assert np.isclose(softmax(z).sum(), 1.0)

# shift-invariance：整体平移不影响结果
assert np.allclose(softmax(z), softmax(z + 100))`

// ── 格式化辅助 ─────────────────────────────────────────────────────────────────
const fmt2 = (n: number) => n.toFixed(2)
const fmt4 = (n: number) => n.toFixed(4)
const pct  = (n: number) => (n * 100).toFixed(1) + '%'

// ── 主组件 ─────────────────────────────────────────────────────────────────────
export function Softmax() {
  const [logits, setLogits] = useState<number[]>(() =>
    TOKEN_DEFS.map((t) => t.defaultLogit)
  )
  const [temp, setTemp] = useState(1.0)

  const { exps, Z, probs } = computeSoftmax(logits, temp)
  const argmax = probs.indexOf(Math.max(...probs))

  const me               = findChapter('softmax')!
  const { prev, next }   = neighbors('softmax')

  const argmaxZh  = TOKEN_DEFS[argmax].zh
  const argmaxZ   = logits[argmax]

  function setLogit(idx: number, val: number): void {
    setLogits((cur) => {
      const arr = [...cur]
      arr[idx] = val
      return arr
    })
  }

  return (
    <article className="page">

      {/* ── 页头 ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第七部分 · 概率视角：模型在「猜下一个词」
        </div>
        <div className="kicker">第 {me.num} 节 ★</div>
        <h1>
          Softmax 与概率分布
          <span className="zh-sub">把一排分数变成「权重」，让模型开口猜</span>
        </h1>
        <p className="lede">
          模型内部跑完一层层线性代数，最终得到一排<strong>分数（logits）</strong>——
          一个给「猫」，一个给「狗」……但分数可以是负数、可以很大，彼此之间没有可比基准。
          <strong> Softmax</strong> 把这排任意实数变成一个真正的
          <strong>概率分布（probability distribution）</strong>：
          全部非负、加起来等于 1、最大的分数对应最大的概率。
          公式：<code>softmax(z)ᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)</code>。
          这个操作出现在注意力权重和输出词表两个最核心的位置。
        </p>
      </header>

      {/* ── 控制区：logit 滑块 + temperature 滑块 ── */}
      <section className="controls">
        {TOKEN_DEFS.map((tok, i) => (
          <div className="control" key={tok.zh}>
            <div className="control-head">
              {/* argmax 那个换底色，不是换字色——胶囊是深底白字，改 color 会让字消失。 */}
              <span className={`slot-tag${i === argmax ? ' slot-tag--rust' : ''}`}>
                {tok.zh}
              </span>
              <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
                logit
              </span>
            </div>
            <label className="slider-row">
              <input
                type="range"
                min={-3}
                max={3}
                step={0.1}
                value={logits[i]}
                onChange={(e) => setLogit(i, Number(e.target.value))}
              />
              <span
                className="param-val"
                style={{ color: i === argmax ? RUST : undefined }}
              >
                {fmt2(logits[i])}
              </span>
            </label>
          </div>
        ))}

        {/* temperature 滑块 — 用 rust 边框标出它的特殊地位 */}
        <div
          className="control"
          style={{ borderLeft: `3px solid ${RUST}`, paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag slot-tag--rust">T</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
              temperature（越小越锐，越大越平）
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range"
              min={0.2}
              max={5}
              step={0.1}
              value={temp}
              onChange={(e) => setTemp(Number(e.target.value))}
            />
            <span className="param-val" style={{ color: RUST }}>{fmt2(temp)}</span>
          </label>
        </div>
      </section>

      {/* ── 柱状图 ── */}
      <section className="stage">
        <div
          role="img"
          aria-label="Softmax 概率分布柱状图，共六个 token"
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: '0.75rem',
            height: `${BAR_MAX_PX + 56}px`,
            padding: '0 0.5rem',
          }}
        >
          {TOKEN_DEFS.map((tok, i) => {
            const prob  = probs[i]
            const barPx = Math.round(prob * BAR_MAX_PX)
            const isMax = i === argmax
            return (
              <div
                key={tok.zh}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  flex: 1,
                  minWidth: 0,
                }}
              >
                {/* 概率百分比标注 */}
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: isMax ? 700 : 400,
                    color: isMax ? RUST : '#555',
                    marginBottom: '4px',
                    letterSpacing: '-0.01em',
                  }}
                >
                  {pct(prob)}
                </span>

                {/* 柱体 */}
                <div
                  style={{
                    width: '100%',
                    height: `${Math.max(barPx, 2)}px`,
                    backgroundColor: isMax ? RUST : IKB,
                    borderRadius: '3px 3px 0 0',
                    transition: 'height 0.12s ease, background-color 0.12s ease',
                  }}
                />

                {/* token 汉字 */}
                <span
                  style={{
                    fontSize: '0.9rem',
                    fontWeight: isMax ? 700 : 400,
                    color: isMax ? RUST : '#222',
                    marginTop: '6px',
                  }}
                >
                  {tok.zh}
                </span>

                {/* logit 值 */}
                <span style={{ fontSize: '0.65rem', color: '#999', marginTop: '2px' }}>
                  z = {fmt2(logits[i])}
                </span>
              </div>
            )
          })}
        </div>
        <p
          style={{
            textAlign: 'center',
            fontSize: '0.74rem',
            color: '#888',
            marginTop: '0.5rem',
          }}
        >
          柱高 ∝ softmax 概率 · 所有柱加起来 = 100% ·{' '}
          <span style={{ color: RUST, fontWeight: 600 }}>锈色柱</span> = argmax
        </p>
      </section>

      {/* ── 逐步展示：logit → exp(z/T) → prob ── */}
      <section className="readouts">
        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.82rem',
            marginTop: '0.5rem',
          }}
        >
          <thead>
            <tr style={{ borderBottom: `2px solid ${IKB}` }}>
              <th style={{ textAlign: 'left',  padding: '4px 8px', color: IKB }}>token</th>
              <th style={{ textAlign: 'right', padding: '4px 8px', color: IKB }}>logit z</th>
              <th style={{ textAlign: 'right', padding: '4px 8px', color: IKB }}>exp((z−z_max) / T)</th>
              <th style={{ textAlign: 'right', padding: '4px 8px', color: IKB }}>prob</th>
              <th style={{ textAlign: 'right', padding: '4px 8px', color: IKB }}>%</th>
            </tr>
          </thead>
          <tbody>
            {TOKEN_DEFS.map((tok, i) => {
              const prob    = probs[i]
              const expVal  = exps[i]
              const logitV  = logits[i]
              const isMax   = i === argmax
              return (
                <tr
                  key={tok.zh}
                  style={{
                    background: isMax ? 'rgba(199,91,57,0.07)' : 'transparent',
                    fontWeight: isMax ? 700 : 400,
                  }}
                >
                  <td style={{ padding: '4px 8px', color: isMax ? RUST : '#222' }}>
                    {tok.zh}
                  </td>
                  <td style={{ textAlign: 'right', padding: '4px 8px', fontFamily: 'monospace' }}>
                    {fmt2(logitV)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '4px 8px', fontFamily: 'monospace' }}>
                    {fmt4(expVal)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '4px 8px', fontFamily: 'monospace' }}>
                    {fmt4(prob)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '4px 8px', color: isMax ? RUST : '#444' }}>
                    {pct(prob)}
                  </td>
                </tr>
              )
            })}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: `2px solid ${IKB}` }}>
              <td
                colSpan={2}
                style={{ padding: '4px 8px', color: '#666', fontSize: '0.74rem' }}
              >
                Z（归一化分母 = Σ exp((z−z_max) / T)）
              </td>
              <td
                style={{
                  textAlign: 'right',
                  padding: '4px 8px',
                  fontFamily: 'monospace',
                  color: IKB,
                  fontWeight: 700,
                }}
              >
                {fmt4(Z)}
              </td>
              <td
                style={{
                  textAlign: 'right',
                  padding: '4px 8px',
                  fontFamily: 'monospace',
                  color: IKB,
                  fontWeight: 700,
                }}
              >
                1.0000
              </td>
              <td
                style={{
                  textAlign: 'right',
                  padding: '4px 8px',
                  color: IKB,
                  fontWeight: 700,
                }}
              >
                100.0% ✓
              </td>
            </tr>
          </tfoot>
        </table>
      </section>

      {/* ── 要点说明 ── */}
      <section className="verdict">
        <p>
          <strong>Shift-invariance：</strong>把所有 logit 同时加或减一个常数，softmax 输出
          <em>完全不变</em>。因为分子分母都乘了同一个 <code>e^(c/T)</code>，直接消掉。
          代码里减去 <code>max(z)</code> 正是利用这一性质防止 <code>exp</code> 上溢——
          最大那项变成 <code>exp(0) = 1</code>，其余都 ≤ 1。
          <strong>上面表格第三列和 Z 显示的就是减完 max 之后的值</strong>，
          所以最大那行永远是 1.0000；换成不减 max 的写法，这两列会整体乘上
          <code> e^(z_max/T)</code>，而最右边的概率一个都不会变。
        </p>
        <p>
          <strong>Soft argmax：</strong>softmax 是可微的 argmax。
          logit 最大的那个 token 永远得到最大概率——但不是 0 / 1 的硬截断，
          所有输出都有非零梯度，模型可以从每次错误里学习。
          <span style={{ color: RUST }}>
            {' '}当前 argmax =「{argmaxZh}」（z = {fmt2(argmaxZ)}）。
          </span>
        </p>
        <p>
          <strong>Temperature T</strong> 控制分布锐度：
          T → 0 趋近 one-hot（只有 argmax 接近 1，其余趋近 0）；
          T → ∞ 趋近均匀分布（每个 token 各占 {(100 / N).toFixed(1)}%）。
          拖动上方 T 滑块，看柱子如何戏剧性地收窄或铺开。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            Softmax 在 Transformer 里出现在两个关键位置。
            <strong>① 注意力权重</strong>——<code>softmax(QKᵀ / √d)</code>{' '}
            把 token 间的打分变成「每个 token 分配多少注意力」的概率；
            分母的 <code>√d</code> 本质上就是在调 temperature（d 越大，内积越大，除以 √d 降回合理范围），
            与第 06 节讲的范数缩放直接相连。
            <strong>② 输出层</strong>——把最后一层 logits 变成词表上的概率分布，
            模型从中采样或取 argmax 生成下一个词（采样细节见第 36 节）。
          </p>
          <p>
            推理时你看到的 temperature 参数就是这里的 T：
            调低让模型更「保守」（概率集中，输出稳定），
            调高让模型更「冒险」（低概率词也有机会，输出更多样）。
          </p>
        </div>
      </section>

      {/* ── 代码 ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：softmax 怎么算的</h2>
        <CodeBlock code={SNIPPET} language="python" title="softmax.py" />
      </section>

      {/* ── 翻页 ── */}
      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一节</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : (
          <span />
        )}
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
        ) : (
          <span />
        )}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
