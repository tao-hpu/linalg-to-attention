import { useState } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { ChRef, chNum } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'

// ── Color constants ────────────────────────────────────────────────────────────
const IKB  = '#002fa7'  // International Klein Blue — surviving bars
const RUST = '#c75b39'  // rust — picked token / nucleus boundary
const GREY = '#d4d7da'  // filtered-out tail

// ── Bar chart height ───────────────────────────────────────────────────────────
const BAR_MAX_PX = 160

// ── Token definitions (realistic long-tail distribution over ~10 candidates) ──
interface TokenDef {
  readonly token: string
  readonly logit: number
}

const TOKEN_DEFS: readonly TokenDef[] = [
  { token: 'the',    logit:  4.20 },
  { token: 'a',      logit:  3.10 },
  { token: 'an',     logit:  1.80 },
  { token: 'its',    logit:  1.20 },
  { token: 'one',    logit:  0.50 },
  { token: 'this',   logit:  0.00 },
  { token: 'some',   logit: -0.80 },
  { token: 'any',    logit: -1.50 },
  { token: 'every',  logit: -2.30 },
  { token: 'each',   logit: -3.10 },
]

// ── Decoding mode ──────────────────────────────────────────────────────────────
type DecodingMode = 'greedy' | 'pure' | 'top-k' | 'top-p'

// ── Math helpers ───────────────────────────────────────────────────────────────

/** Softmax with temperature. Numerically stable (subtract max). */
function computeSoftmax(logits: readonly number[], T: number): number[] {
  const maxZ = Math.max(...logits)
  const exps  = logits.map((z) => Math.exp((z - maxZ) / T))
  const Z     = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / Z)
}

/** Renormalize a non-negative array so its entries sum to 1. */
function renormalize(probs: readonly number[]): number[] {
  const sum = probs.reduce((a, b) => a + b, 0)
  if (sum < 1e-12) return probs.map(() => 0)
  return probs.map((p) => p / sum)
}

/** Boolean mask: the top-k highest-probability tokens are true. */
function topKMask(probs: readonly number[], k: number): boolean[] {
  const indexed = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p)
  const keepSet = new Set(indexed.slice(0, k).map(({ i }) => i))
  return probs.map((_, i) => keepSet.has(i))
}

/**
 * Nucleus mask: include tokens sorted by descending probability until
 * their cumulative probability >= p (the smallest such set).
 * Returns the mask and the original-array index of the last token added
 * (the nucleus boundary, marked visually in rust).
 */
function topPMask(
  probs: readonly number[],
  p: number,
): { mask: boolean[]; boundaryIdx: number } {
  const sorted  = probs.map((prob, i) => ({ prob, i })).sort((a, b) => b.prob - a.prob)
  const keepSet = new Set<number>()
  let cumsum      = 0
  let boundaryIdx = sorted[0].i  // at least one token always kept

  for (const { prob, i } of sorted) {
    keepSet.add(i)
    cumsum += prob
    if (cumsum >= p) {
      boundaryIdx = i
      break
    }
  }

  return { mask: probs.map((_, i) => keepSet.has(i)), boundaryIdx }
}

/**
 * Deterministic pseudo-sample — no Math.random / Date.now，保证可复现。
 *
 * 阈值取黄金比例的低差异序列 frac(step · φ⁻¹)：它在 [0,1) 上均匀跳跃，
 * 长期频率正好收敛到各 token 的概率。注意不能用 (step % N) / N 这类
 * 单调递增的阈值——那样前若干次点击会全部落在概率最大的那个 token 上
 * （默认 p(the)≈0.66，要连点 60 多次才会出现第二个词），
 * 而这一页的全部意义就是让读者看到采样是有随机性的。
 */
const GOLDEN = 0.6180339887498949

function deterministicSample(probs: readonly number[], step: number): number {
  const threshold = (step * GOLDEN) % 1
  let cumsum = 0
  for (let i = 0; i < probs.length; i++) {
    cumsum += probs[i]
    if (cumsum > threshold) return i
  }
  return probs.length - 1
}

// ── Formatting ─────────────────────────────────────────────────────────────────
const fmt2 = (n: number): string => n.toFixed(2)
const pct  = (n: number): string => (n * 100).toFixed(1) + '%'

// ── Code snippet ───────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

def decode_next(logits, T=1.0, top_k=None, top_p=None):
    # 1. logits / T → softmax（temperature 在这里生效，连回第 ${chNum('softmax')} 节）
    z = logits / T
    z -= z.max()                          # 数值稳定：shift-invariance
    probs = np.exp(z) / np.exp(z).sum()   # softmax

    # 2. top-k mask（保留概率最高的 k 个，其余清零）
    if top_k is not None:
        cutoff = np.sort(probs)[::-1][top_k - 1]
        probs = np.where(probs >= cutoff, probs, 0.0)

    # 3. top-p nucleus mask（最小集合使累积概率 >= p）
    if top_p is not None:
        sorted_desc = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_desc)
        cutoff_idx = np.searchsorted(cumsum, top_p)
        cutoff = sorted_desc[min(cutoff_idx, len(sorted_desc) - 1)]
        probs = np.where(probs >= cutoff, probs, 0.0)

    # 4. 重新归一化（幸存 token 概率重新加回 1）
    probs = probs / probs.sum()

    # 5. 采样（greedy = argmax；其余 = 从归一化分布里随机抽一个）
    return int(np.random.choice(len(probs), p=probs))`

// ── Mode label map ─────────────────────────────────────────────────────────────
const MODE_LABELS: Record<DecodingMode, string> = {
  greedy:  'greedy (argmax)',
  pure:    'pure sampling',
  'top-k': 'top-k',
  'top-p': 'top-p / nucleus',
}

// ── Main component ─────────────────────────────────────────────────────────────
export function SamplingDecoding() {
  const [temp, setTemp] = useState(1.0)
  const [topK, setTopK] = useState(5)
  const [topP, setTopP] = useState(0.9)
  const [mode, setMode] = useState<DecodingMode>('top-p')
  const [step, setStep] = useState(0)

  // ── Compute pipeline ──────────────────────────────────────────────────────────
  const logits    = TOKEN_DEFS.map((t) => t.logit)
  const baseProbs = computeSoftmax(logits, temp)

  // Survive mask — which tokens pass the filter
  let surviveMask: boolean[] = TOKEN_DEFS.map(() => true)
  let boundaryIdx = -1  // nucleus edge (top-p only)

  switch (mode) {
    case 'greedy':
    case 'pure':
      // all tokens survive; no filter applied
      break
    case 'top-k':
      surviveMask = topKMask(baseProbs, topK)
      break
    case 'top-p': {
      const result = topPMask(baseProbs, topP)
      surviveMask  = result.mask
      boundaryIdx  = result.boundaryIdx
      break
    }
  }

  const filteredProbs = baseProbs.map((p, i) => (surviveMask[i] ? p : 0))
  const displayProbs  = renormalize(filteredProbs)
  const numSurvivors  = surviveMask.filter(Boolean).length

  // ── Picked token ──────────────────────────────────────────────────────────────
  // Only shown after first "采样一次" click (step > 0).
  let pickedIdx: number | null = null
  if (step > 0) {
    if (mode === 'greedy') {
      // argmax of displayProbs (= baseProbs when all survive)
      pickedIdx = displayProbs.indexOf(Math.max(...displayProbs))
    } else {
      pickedIdx = deterministicSample(displayProbs, step)
    }
  }

  // ── Button handler ────────────────────────────────────────────────────────────
  function handleSample(): void {
    setStep((s) => s + 1)
  }

  function handleModeChange(m: DecodingMode): void {
    setMode(m)
    setStep(0)
  }

  // ── Bar color ─────────────────────────────────────────────────────────────────
  function barColor(i: number): string {
    if (!surviveMask[i]) return GREY
    if (pickedIdx === i) return RUST
    return IKB
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
      <ChapterShell
        slug="sampling-decoding"
        part="第九部分 · 尾声：接到 LLM 工程"
        sub="模型怎么从概率分布里选出下一个词？"
        lede={
          <>
        模型前向传播结束，输出词表上一排 logit；经过 softmax（<ChRef slug="softmax" />）
        变成<strong>概率分布</strong>。但分布本身不是词——还需要一步
        「<strong>decoding</strong>」：从分布里<em>挑出</em>下一个 token。
        <strong> Greedy（argmax）</strong>永远选概率最高的那个，稳定但容易重复；
        <strong> temperature</strong> 在 softmax 之前重塑分布——低温收窄、高温铺开；
        <strong> top-k</strong> 只保留最可能的 k 个候选再采样；
        <strong> top-p（nucleus sampling）</strong>保留累积概率刚好够 p 的最小候选集合，
        是当前生产环境最常见的折中。下面亲手调参，观察分布如何被过滤和重新归一化。
          </>
        }
      >


      {/* ── Controls ── */}
      <section className="controls">

        {/* Decoding mode selector */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">策略</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
              decoding 模式
            </span>
          </div>
          <div
            style={{
              display: 'flex',
              gap: '0.5rem',
              flexWrap: 'wrap',
              marginTop: '0.5rem',
            }}
          >
            {(['greedy', 'pure', 'top-k', 'top-p'] as const).map((m) => (
              <label
                key={m}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  cursor: 'pointer',
                  padding: '0.22rem 0.65rem',
                  border: `1.5px solid ${mode === m ? IKB : '#ccc'}`,
                  borderRadius: '4px',
                  fontSize: '0.82rem',
                  color: mode === m ? IKB : '#555',
                  fontWeight: mode === m ? 700 : 400,
                  background: mode === m ? 'rgba(0,47,167,0.06)' : 'transparent',
                  transition: 'border-color 0.1s, color 0.1s',
                  userSelect: 'none',
                }}
              >
                <input
                  type="radio"
                  name="decoding-mode"
                  value={m}
                  checked={mode === m}
                  onChange={() => handleModeChange(m)}
                  style={{ accentColor: IKB, margin: 0 }}
                />
                {MODE_LABELS[m]}
              </label>
            ))}
          </div>
        </div>

        {/* Temperature slider — always active */}
        <div
          className="control"
          style={{ borderLeft: `3px solid ${RUST}`, paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag slot-tag--rust">T</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
              temperature（低 → 锐，高 → 平；连回<ChRef slug="softmax" /> softmax）
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range"
              min={0.1}
              max={3.0}
              step={0.1}
              value={temp}
              onChange={(e) => { setTemp(Number(e.target.value)); setStep(0) }}
            />
            <span className="param-val" style={{ color: RUST }}>{fmt2(temp)}</span>
          </label>
        </div>

        {/* Top-k slider — dimmed when not in top-k mode */}
        <div
          className="control"
          style={{
            opacity: mode === 'top-k' ? 1 : 0.38,
            transition: 'opacity 0.15s',
          }}
        >
          <div className="control-head">
            <span className="slot-tag">k</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
              top-k 候选数
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={topK}
              disabled={mode !== 'top-k'}
              onChange={(e) => { setTopK(Number(e.target.value)); setStep(0) }}
            />
            <span className="param-val">{topK}</span>
          </label>
        </div>

        {/* Top-p slider — dimmed when not in top-p mode */}
        <div
          className="control"
          style={{
            opacity: mode === 'top-p' ? 1 : 0.38,
            transition: 'opacity 0.15s',
          }}
        >
          <div className="control-head">
            <span className="slot-tag">p</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.3rem' }}>
              top-p 累积概率阈值（nucleus sampling）
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range"
              min={0.50}
              max={1.00}
              step={0.01}
              value={topP}
              disabled={mode !== 'top-p'}
              onChange={(e) => { setTopP(Number(e.target.value)); setStep(0) }}
            />
            <span className="param-val">{fmt2(topP)}</span>
          </label>
        </div>

        {/* Sample button + result */}
        <div className="control">
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
              flexWrap: 'wrap',
            }}
          >
            <button
              onClick={handleSample}
              style={{
                padding: '0.38rem 1.1rem',
                background: IKB,
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                fontSize: '0.88rem',
                cursor: 'pointer',
                fontWeight: 600,
                letterSpacing: '0.01em',
                flexShrink: 0,
              }}
            >
              采样一次
            </button>

            {step > 0 && pickedIdx !== null && (
              <span style={{ fontSize: '0.86rem', lineHeight: 1.4 }}>
                第 {step} 次 →{' '}
                <strong style={{ color: RUST }}>
                  &ldquo;{TOKEN_DEFS[pickedIdx].token}&rdquo;
                </strong>
                {' '}（{pct(displayProbs[pickedIdx])}）
                {mode === 'greedy' && (
                  <span style={{ color: '#888', fontSize: '0.78rem' }}>
                    {' '}· greedy 每次都选这个
                  </span>
                )}
              </span>
            )}
          </div>
        </div>
      </section>

      {/* ── Bar chart ── */}
      <section className="stage">
        <div
          role="img"
          aria-label="采样候选 token 的概率分布柱状图，共十个 token"
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            gap: '0.45rem',
            height: `${BAR_MAX_PX + 68}px`,
            padding: '0 0.25rem',
          }}
        >
          {TOKEN_DEFS.map((tok, i) => {
            const prob       = displayProbs[i]
            const barPx      = Math.round(prob * BAR_MAX_PX)
            const color      = barColor(i)
            const isPicked   = pickedIdx === i
            const isBoundary = mode === 'top-p' && i === boundaryIdx && !isPicked
            const isSurvive  = surviveMask[i]

            return (
              <div
                key={tok.token}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  flex: 1,
                  minWidth: 0,
                }}
              >
                {/* Renormalized probability label */}
                <span
                  style={{
                    fontSize: '0.64rem',
                    fontWeight: isPicked ? 700 : 400,
                    color: isPicked ? RUST : isSurvive ? '#444' : '#bbb',
                    marginBottom: '3px',
                    letterSpacing: '-0.01em',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {isSurvive ? pct(prob) : '—'}
                </span>

                {/* Bar body */}
                <div
                  style={{
                    width: '100%',
                    height: `${Math.max(barPx, 2)}px`,
                    backgroundColor: color,
                    borderRadius: '3px 3px 0 0',
                    outline: isBoundary ? `2px solid ${RUST}` : 'none',
                    outlineOffset: '1px',
                    transition: 'height 0.12s ease, background-color 0.12s ease',
                  }}
                />

                {/* Token label */}
                <span
                  style={{
                    fontSize: '0.76rem',
                    fontWeight: isPicked ? 700 : 400,
                    color: isPicked ? RUST : isSurvive ? '#222' : '#aaa',
                    marginTop: '5px',
                    fontFamily: 'monospace',
                  }}
                >
                  {tok.token}
                </span>

                {/* Original logit */}
                <span style={{ fontSize: '0.59rem', color: '#c0c0c0', marginTop: '2px' }}>
                  {fmt2(tok.logit)}
                </span>
              </div>
            )
          })}
        </div>

        <p
          style={{
            textAlign: 'center',
            fontSize: '0.73rem',
            color: '#888',
            marginTop: '0.6rem',
            lineHeight: 1.5,
          }}
        >
          柱高 ∝ 过滤后重归一化概率 ·{' '}
          <span style={{ color: IKB, fontWeight: 600 }}>蓝色</span> = 幸存候选 ·{' '}
          <span style={{ color: RUST, fontWeight: 600 }}>锈色</span> = 已采样 token
          {mode === 'top-p' && (
            <>
              {' '}· <span style={{ fontWeight: 600 }}>锈色边框</span> = nucleus 边界
            </>
          )}
          {' '}· <span style={{ color: GREY, fontWeight: 600 }}>灰色</span> = 已过滤
        </p>
      </section>

      {/* ── Live readouts ── */}
      <section className="readouts">
        <p
          style={{
            fontSize: '0.8rem',
            color: '#555',
            marginBottom: '0.5rem',
            lineHeight: 1.6,
          }}
        >
          策略：
          <strong style={{ color: IKB }}>{MODE_LABELS[mode]}</strong>
          {mode === 'top-k' && <span style={{ color: '#666' }}>{' '}(k = {topK})</span>}
          {mode === 'top-p' && <span style={{ color: '#666' }}>{' '}(p = {fmt2(topP)})</span>}
          {' '}· 幸存候选：
          <strong style={{ color: IKB }}>{numSurvivors} / {TOKEN_DEFS.length}</strong>
          {' '}· temperature：
          <strong style={{ color: RUST }}>T = {fmt2(temp)}</strong>
        </p>

        <table
          style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '0.79rem',
          }}
        >
          <thead>
            <tr style={{ borderBottom: `2px solid ${IKB}` }}>
              <th style={{ textAlign: 'left',  padding: '4px 6px', color: IKB }}>token</th>
              <th style={{ textAlign: 'right', padding: '4px 6px', color: IKB }}>logit</th>
              <th style={{ textAlign: 'right', padding: '4px 6px', color: IKB }}>
                基础概率（T={fmt2(temp)}）
              </th>
              <th style={{ textAlign: 'right', padding: '4px 6px', color: IKB }}>
                过滤后重归一化
              </th>
              <th style={{ textAlign: 'right', padding: '4px 6px', color: IKB }}>状态</th>
            </tr>
          </thead>
          <tbody>
            {TOKEN_DEFS.map((tok, i) => {
              const isPicked   = pickedIdx === i
              const isSurvive  = surviveMask[i]
              const isBoundary = mode === 'top-p' && i === boundaryIdx

              return (
                <tr
                  key={tok.token}
                  style={{
                    background: isPicked
                      ? 'rgba(199,91,57,0.07)'
                      : isSurvive
                      ? 'transparent'
                      : 'rgba(212,215,218,0.15)',
                    fontWeight: isPicked ? 700 : 400,
                  }}
                >
                  <td
                    style={{
                      padding: '3px 6px',
                      fontFamily: 'monospace',
                      color: isPicked ? RUST : isSurvive ? '#222' : '#aaa',
                    }}
                  >
                    {tok.token}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      padding: '3px 6px',
                      fontFamily: 'monospace',
                      color: '#777',
                    }}
                  >
                    {fmt2(tok.logit)}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      padding: '3px 6px',
                      fontFamily: 'monospace',
                      color: isSurvive ? '#444' : '#ccc',
                    }}
                  >
                    {pct(baseProbs[i])}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      padding: '3px 6px',
                      fontFamily: 'monospace',
                      color: isPicked ? RUST : isSurvive ? IKB : '#ccc',
                    }}
                  >
                    {isSurvive ? pct(displayProbs[i]) : '—'}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      padding: '3px 6px',
                      fontSize: '0.71rem',
                    }}
                  >
                    {!isSurvive ? (
                      <span style={{ color: '#aaa' }}>已过滤</span>
                    ) : isPicked ? (
                      <span style={{ color: RUST, fontWeight: 700 }}>已采样</span>
                    ) : isBoundary ? (
                      <span style={{ color: RUST }}>nucleus 边界</span>
                    ) : (
                      <span style={{ color: IKB }}>候选</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>

      {/* ── Verdict ── */}
      <section className="verdict">
        <p>
          <strong>Greedy / argmax：</strong>
          永远选概率最高的词，结果可复现但容易单调——模型会在高概率词上正反馈，
          输出反复出现相同短语。适合需要确定性输出的场景（如分类、问答）。
        </p>
        <p>
          <strong>Temperature：</strong>
          在 softmax 之前把 logit 除以 T，再归一化。
          T &lt; 1 让分布更锐（低概率词接近零，高概率词接近 1）；
          T &gt; 1 让分布更平（低概率词也能脱颖而出）。
          拖动 T 滑块，观察柱子整体收窄或铺开——这就是<ChRef slug="softmax" /> softmax 里 T 参数在推理时的直接体现。
        </p>
        <p>
          <strong>Top-k：</strong>
          保留概率最高的 k 个候选，其余清零后重新归一化再采样。
          k = 1 等价于 greedy；k = 10（词表大小）等价于 pure sampling。
          问题在于固定的 k 无法适应分布形状——有时 top-5 已覆盖 99%，有时只覆盖 40%。
        </p>
        <p>
          <strong>Top-p（nucleus sampling）：</strong>
          把 token 从高概率到低概率排列，累加直到累积概率 ≥ p，停止——这个最小集合就是 nucleus。
          分布锐时 nucleus 可能只有 2–3 个词；分布平时可能有七八个词，自适应分布形状。
          将 p 调到 1.0，所有 token 都进入 nucleus，等价于 pure sampling。
          这比固定的 top-k 更能应对长尾分布，是目前生产环境最常见的解码策略。
        </p>
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            这就是 <code>model.generate(...)</code> 里 <code>temperature</code>、
            <code>top_k</code>、<code>top_p</code> 那几个参数在干的事——推理时
            <strong>逐 token 重复</strong>：出分布 → 按策略筛 → 采一个 → 喂回去
            → 再出分布……它不改模型权重，只改「怎么从分布里选词」。
            温度高更有创意但容易跑飞；top-p 是当前最常用的折中。
          </p>
          <p>
            连回前面：分布从哪来——<ChRef slug="softmax" /> softmax 把 logit 变成概率；
            训练时的分布目标——<ChRef slug="cross-entropy" />交叉熵要求模型把概率压到正确词上。
            推理时你的采样参数调的是「怎么使用这个分布」，
            而模型本身（logit 怎么算出来）完全没变。
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：完整解码流水线</h2>
        <CodeBlock code={SNIPPET} language="python" title="sampling_decoding.py" />
      </section>

      {/* ── Pager ── */}
      </ChapterShell>
  )
}
