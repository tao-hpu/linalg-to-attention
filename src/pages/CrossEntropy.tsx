import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ─── Math helpers ────────────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const m = Math.max(...logits)
  const exps = logits.map((l) => Math.exp(l - m))
  const s = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / s)
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x))
}

// ─── −log(p) curve constants & helpers ───────────────────────────────────────

const CW = 264
const CH = 156
const CPL = 36
const CPT = 10
const CPB = 34
const CPR = 10
const CPW = CW - CPL - CPR
const CPH = CH - CPT - CPB
const MAX_Y = 3.5

function pToX(p: number): number {
  return CPL + p * CPW
}

function nlToY(nl: number): number {
  return CPT + CPH * (1 - clamp(nl, 0, MAX_Y) / MAX_Y)
}

const CURVE_D: string = (() => {
  const pts: string[] = []
  for (let i = 0; i <= 200; i++) {
    const p = 0.005 + 0.995 * (i / 200)
    pts.push(
      `${i === 0 ? 'M' : 'L'}${pToX(p).toFixed(1)},${nlToY(-Math.log(p)).toFixed(1)}`
    )
  }
  return pts.join(' ')
})()

// ─── Token labels ─────────────────────────────────────────────────────────────

const TOKENS: string[] = ['the', 'cat', 'sat', 'on', 'a']
const SEQ_TOKENS: string[] = ['The', 'cat', 'sat', 'on']
const SEQ_INIT_P: number[] = [0.72, 0.35, 0.85, 0.18]

// ─── Bar chart dimensions ─────────────────────────────────────────────────────

const BW = 46
const BGAP = 10
const BPL = 10
const BPT = 14
const BPB = 40
const BPH = 108
const N_TOK = TOKENS.length
const BSVG_W = BPL * 2 + N_TOK * BW + (N_TOK - 1) * BGAP
const BSVG_H = BPT + BPH + BPB

// ─── NegLogCurve component ────────────────────────────────────────────────────

function NegLogCurve({ p }: { p: number }) {
  const safeP = clamp(p, 1e-6, 1 - 1e-6)
  const nl = -Math.log(safeP)
  const mx = pToX(safeP)
  const my = nlToY(clamp(nl, 0, MAX_Y))
  const axisBottom = CPT + CPH
  const axisLeft = CPL

  return (
    <figure style={{ margin: 0 }}>
      <svg
        width={CW}
        height={CH}
        style={{ display: 'block', overflow: 'visible' }}
        role="img"
        aria-label="−log(p) 曲线：p 越小损失越大"
      >
        {/* axes */}
        <line x1={axisLeft} y1={CPT} x2={axisLeft} y2={axisBottom} stroke="#c7cbd0" strokeWidth={1} />
        <line x1={axisLeft} y1={axisBottom} x2={axisLeft + CPW} y2={axisBottom} stroke="#c7cbd0" strokeWidth={1} />

        {/* x ticks */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const tx = pToX(t)
          return (
            <g key={t}>
              <line x1={tx} y1={axisBottom} x2={tx} y2={axisBottom + 4} stroke="#c7cbd0" strokeWidth={1} />
              <text x={tx} y={axisBottom + 15} textAnchor="middle" fontSize={10} fill="#5b6168">{t}</text>
            </g>
          )
        })}

        {/* y ticks */}
        {[0, 1, 2, 3].map((t) => {
          const ty = nlToY(t)
          return (
            <g key={t}>
              <line x1={axisLeft - 4} y1={ty} x2={axisLeft} y2={ty} stroke="#c7cbd0" strokeWidth={1} />
              <text x={axisLeft - 7} y={ty + 4} textAnchor="end" fontSize={10} fill="#5b6168">{t}</text>
            </g>
          )
        })}

        {/* curve */}
        <path d={CURVE_D} fill="none" stroke="#c75b39" strokeWidth={2} strokeLinecap="round" />

        {/* drop lines to marker */}
        <line x1={mx} y1={my} x2={mx} y2={axisBottom} stroke="#002fa7" strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />
        <line x1={axisLeft} y1={my} x2={mx} y2={my} stroke="#002fa7" strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />

        {/* marker */}
        <circle cx={mx} cy={my} r={5} fill="#c75b39" stroke="#fff" strokeWidth={1.5} />

        {/* axis labels */}
        <text x={axisLeft + CPW / 2} y={CH - 2} textAnchor="middle" fontSize={11} fill="#5b6168">p</text>
        <text
          x={axisLeft - 26}
          y={CPT + CPH / 2}
          textAnchor="middle"
          fontSize={11}
          fill="#5b6168"
          transform={`rotate(-90,${axisLeft - 26},${CPT + CPH / 2})`}
        >
          −log(p)
        </text>
      </svg>
      <figcaption style={{ textAlign: 'center', fontSize: 12, color: '#5b6168', marginTop: 4 }}>
        p = {safeP.toFixed(3)} → −log(p) ={' '}
        <span style={{ color: '#c75b39', fontWeight: 700 }}>
          {nl > 9.99 ? '>9.99' : nl.toFixed(3)}
        </span>
        {nl > MAX_Y - 0.01 && (
          <span style={{ fontSize: 11 }}> · 超出图示范围</span>
        )}
      </figcaption>
    </figure>
  )
}

// ─── ProbBars component ───────────────────────────────────────────────────────

function ProbBars({
  probs,
  trueClass,
  onTrueClass,
}: {
  probs: number[]
  trueClass: number
  onTrueClass: (i: number) => void
}) {
  return (
    <svg
      width={BSVG_W}
      height={BSVG_H}
      style={{ display: 'block' }}
      role="img"
      aria-label="预测概率分布（点击柱选择正确词）"
    >
      {/* baseline */}
      <line
        x1={BPL}
        y1={BPT + BPH}
        x2={BPL + N_TOK * BW + (N_TOK - 1) * BGAP}
        y2={BPT + BPH}
        stroke="#c7cbd0"
        strokeWidth={1}
      />

      {probs.map((p, i) => {
        const barH = Math.max(p * BPH, 2)
        const bx = BPL + i * (BW + BGAP)
        const by = BPT + BPH - barH
        const isTrue = i === trueClass
        const barFill = isTrue ? '#c75b39' : '#002fa7'
        const barOpacity = isTrue ? 1 : 0.65
        return (
          <g key={TOKENS[i]} onClick={() => onTrueClass(i)} style={{ cursor: 'pointer' }}>
            {/* invisible expanded hitbox */}
            <rect x={bx} y={BPT} width={BW} height={BPH + 10} fill="transparent" />
            {/* bar */}
            <rect x={bx} y={by} width={BW} height={barH} fill={barFill} fillOpacity={barOpacity} rx={2} />
            {/* prob label above bar */}
            <text
              x={bx + BW / 2}
              y={by - 4}
              textAnchor="middle"
              fontSize={10}
              fill={barFill}
              fontFamily="SF Mono, JetBrains Mono, Menlo, monospace"
            >
              {p.toFixed(2)}
            </text>
            {/* token label below baseline */}
            <text
              x={bx + BW / 2}
              y={BPT + BPH + 16}
              textAnchor="middle"
              fontSize={13}
              fill={isTrue ? '#c75b39' : '#5b6168'}
              fontWeight={isTrue ? '700' : '400'}
            >
              {TOKENS[i]}
            </text>
            {/* true indicator */}
            {isTrue && (
              <text
                x={bx + BW / 2}
                y={BPT + BPH + 32}
                textAnchor="middle"
                fontSize={10}
                fill="#c75b39"
                fontWeight="700"
              >
                ★ true
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

// ─── MultiSection component ───────────────────────────────────────────────────

function MultiSection() {
  const [pVals, setPVals] = useState<number[]>(SEQ_INIT_P)

  const ces = pVals.map((p) => -Math.log(clamp(p, 1e-10, 1)))
  const n = pVals.length
  const totalCE = ces.reduce((a, b) => a + b, 0) / n
  const logLikelihood = pVals.reduce((acc, p) => acc + Math.log(clamp(p, 1e-10, 1)), 0)
  const likelihood = pVals.reduce((acc, p) => acc * clamp(p, 1e-10, 1), 1)
  const perplexity = Math.exp(totalCE)

  function updateP(i: number, val: number) {
    setPVals((prev) => prev.map((p, j) => (j === i ? val : p)))
  }

  return (
    <section style={{ margin: '40px 0' }}>
      <h2 className="sec-h">多样本视角：likelihood → log-likelihood → CE</h2>
      <p style={{ color: 'var(--ink-soft)', fontSize: 15.5, marginBottom: 24, maxWidth: '62ch' }}>
        将模型应用于一段 4-token 序列。在每个位置，模型预测下一个词，真实词获得概率 p_i。
        拖动滑块改变各位置的 p_i，观察 likelihood / log-likelihood / cross-entropy 三者如何联动。
      </p>

      {/* Token cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 24 }}>
        {SEQ_TOKENS.map((tok, i) => (
          <div
            key={tok}
            style={{
              border: '1px solid var(--line)',
              borderTop: '3px solid #c75b39',
              padding: '14px 14px 16px',
            }}
          >
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--ink-soft)', marginBottom: 6 }}>
              位置 {i + 1}
            </div>
            <div style={{ fontSize: 17, fontWeight: 700, marginBottom: 10, color: 'var(--ink)' }}>
              "{tok}"
            </div>
            {/* Mini prob bar */}
            <div style={{
              height: 5,
              background: 'var(--line)',
              borderRadius: 3,
              marginBottom: 10,
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%',
                width: `${pVals[i] * 100}%`,
                background: '#c75b39',
                borderRadius: 3,
              }} />
            </div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <span style={{ fontSize: 11, color: 'var(--ink-soft)' }}>p_correct</span>
              <input
                type="range"
                min={0.01}
                max={0.99}
                step={0.01}
                value={pVals[i]}
                onChange={(e) => updateP(i, Number(e.target.value))}
                style={{ accentColor: '#c75b39' }}
              />
              <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--ink)' }}>
                {pVals[i].toFixed(2)}
              </span>
            </label>
            {/* Per-token CE */}
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--line)' }}>
              <div style={{ fontSize: 11, color: 'var(--ink-soft)', fontFamily: 'var(--mono)' }}>CE_i = −log pᵢ</div>
              <div style={{ fontSize: 15, fontWeight: 700, color: '#c75b39', fontFamily: 'var(--mono)' }}>
                {ces[i].toFixed(3)}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Summary box */}
      <div style={{ background: '#fafbfc', border: '1px solid var(--line)', padding: '20px 24px' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 20,
          marginBottom: 20,
          fontFamily: 'var(--mono)',
        }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginBottom: 4 }}>likelihood = Π pᵢ</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)' }}>
              {likelihood < 1e-4 ? likelihood.toExponential(2) : likelihood.toFixed(4)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginTop: 4 }}>（极小，易下溢）</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginBottom: 4 }}>log-likelihood = Σ log pᵢ</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)' }}>
              {logLikelihood.toFixed(3)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginTop: 4 }}>= −n × CE</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginBottom: 4 }}>cross-entropy CE</div>
            <div style={{ fontSize: 16, fontWeight: 700, color: '#c75b39' }}>
              {totalCE.toFixed(3)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginTop: 4 }}>= −log-likelihood / n</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginBottom: 4 }}>perplexity = exp(CE)</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--ink)' }}>
              {perplexity.toFixed(2)}
            </div>
            <div style={{ fontSize: 11, color: 'var(--ink-soft)', marginTop: 4 }}>exp(CE)</div>
          </div>
        </div>

        {/* Identity strip */}
        <div style={{
          borderTop: '1px solid var(--line)',
          paddingTop: 16,
          display: 'flex',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 10,
          fontSize: 14,
        }}>
          <span style={{ background: 'var(--ikb-soft)', color: 'var(--ikb)', padding: '4px 10px', borderRadius: 3, fontWeight: 600 }}>
            最大化 likelihood
          </span>
          <span style={{ color: 'var(--ink-soft)', fontSize: 16 }}>⟺</span>
          <span style={{ background: 'var(--ikb-soft)', color: 'var(--ikb)', padding: '4px 10px', borderRadius: 3, fontWeight: 600 }}>
            最大化 log-likelihood
          </span>
          <span style={{ color: 'var(--ink-soft)', fontSize: 16 }}>⟺</span>
          <span style={{ background: '#fff0ec', color: '#c75b39', padding: '4px 10px', borderRadius: 3, fontWeight: 700 }}>
            最小化 cross-entropy
          </span>
          <span style={{ color: 'var(--ink-soft)', fontSize: 12, marginLeft: 4 }}>← LLM 训练目标</span>
        </div>
      </div>
    </section>
  )
}

// ─── Code snippet ─────────────────────────────────────────────────────────────

const SNIPPET = `import numpy as np

# 假设 probs: (n, V) — softmax 后的概率矩阵
#      targets: (n,)  — 每个位置真实词的索引

n         = len(targets)
p_correct = probs[np.arange(n), targets]     # 每位置「正确词」的概率

# 单样本：CE = −log(p[target])，one-hot 标签的交叉熵
ce_i = -np.log(np.maximum(p_correct, 1e-10)) # 逐 token；p→0 → loss→∞
#   p = 1   →  ce_i = 0   （模型完全确信，无需惩罚）
#   p = 0.5 →  ce_i ≈ 0.693
#   p → 0   →  ce_i → ∞   （自信地犯错，惩罚极大）

# 批量平均 CE = NLL（负对数似然）
ce = ce_i.mean()                              # CE = −(1/n) Σ log pᵢ

# ── likelihood / log-likelihood / CE 恒等关系 ────────────────────────
likelihood     = np.prod(p_correct)           # Π pᵢ（极小，易下溢）
log_likelihood = np.sum(np.log(p_correct))    # Σ log pᵢ（数值稳定）
ce_check       = -log_likelihood / n         # 与 ce 完全相等 ✓

perplexity = np.exp(ce)                       # perplexity = exp(CE)

# PyTorch 等价（内部 log_softmax + NLL）：
# loss = F.cross_entropy(logits, targets)`

// ─── Main page component ──────────────────────────────────────────────────────

export function CrossEntropy() {
  const [logits, setLogits] = useState<number[]>([2.0, 1.0, 0.5, -0.5, -1.0])
  const [trueClass, setTrueClass] = useState<number>(0)

  const me = findChapter('cross-entropy')!
  const { prev, next } = neighbors('cross-entropy')

  const probs = softmax(logits)
  const pTrue = clamp(probs[trueClass], 1e-10, 1)
  const ce = -Math.log(pTrue)
  const verdictGood = ce < 0.5

  const verdictMsg = (() => {
    if (pTrue > 0.85) return `模型把 ${(pTrue * 100).toFixed(0)}% 的概率压在正确词上，loss 趋近零 —— 这正是训练收敛时的样子。`
    if (pTrue > 0.5)  return `模型给正确词 ${(pTrue * 100).toFixed(0)}% 的概率，loss 适中但仍有提升空间。`
    if (pTrue > 0.1)  return `正确词只拿到 ${(pTrue * 100).toFixed(0)}%，大部分概率押在了其它词上，loss 偏高。`
    return `模型几乎把所有概率给了错误词，正确词仅 ${(pTrue * 100).toFixed(0)}%，loss 趋向无穷 —— 越自信地犯错，惩罚越重。`
  })()

  function updateLogit(i: number, val: number) {
    setLogits((prev) => prev.map((l, j) => (j === i ? val : l)))
  }

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第七部分 · 概率视角：模型在「猜下一个词」
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          交叉熵与极大似然
          <span className="zh-sub">LLM 的训练目标，就是经典统计里的 MLE</span>
        </h1>
        <p className="lede">
          统计里有个核心思想叫<strong>极大似然（MLE）</strong>：给定数据，寻找最大化其联合概率的参数。
          LLM 的训练损失 <code>F.cross_entropy</code>
          ——数学上完全等价，换了件衣服而已。
          这一页把 NLL = CE = −MLE 这个恒等式从纸上拉进可交互的直觉。
        </p>
      </header>

      {/* ── Controls ────────────────────────────────────────────────────── */}
      <section className="controls">
        {/* Left: logit sliders */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">logits</span>
            <span style={{ fontSize: 13, color: 'var(--ink-soft)' }}>→ softmax → 概率分布</span>
          </div>
          {TOKENS.map((tok, i) => (
            <label key={tok} className="slider-row" style={{ marginBottom: 8 }}>
              <span style={{
                fontFamily: 'var(--mono)',
                fontSize: 13,
                width: 28,
                flexShrink: 0,
                color: i === trueClass ? '#c75b39' : 'var(--ink-soft)',
                fontWeight: i === trueClass ? 700 : 400,
              }}>
                {tok}
              </span>
              <input
                type="range"
                min={-3}
                max={3}
                step={0.1}
                value={logits[i]}
                onChange={(e) => updateLogit(i, Number(e.target.value))}
              />
              <span className="param-val">{logits[i].toFixed(1)}</span>
            </label>
          ))}
        </div>

        {/* Right: true class selector */}
        <div className="control" style={{ borderTopColor: '#c75b39' }}>
          <div className="control-head">
            <span className="slot-tag" style={{ background: '#c75b39' }}>true</span>
            <span style={{ fontSize: 13, color: 'var(--ink-soft)' }}>正确词（或点击柱）</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {TOKENS.map((tok, i) => (
              <label
                key={tok}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  cursor: 'pointer',
                  padding: '6px 10px',
                  borderRadius: 4,
                  background: i === trueClass ? '#fff0ec' : 'transparent',
                  border: `1px solid ${i === trueClass ? '#c75b39' : 'transparent'}`,
                }}
              >
                <input
                  type="radio"
                  name="true-class"
                  checked={i === trueClass}
                  onChange={() => setTrueClass(i)}
                  style={{ accentColor: '#c75b39' }}
                />
                <span style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 14,
                  color: i === trueClass ? '#c75b39' : 'var(--ink)',
                  fontWeight: i === trueClass ? 700 : 400,
                  flex: 1,
                }}>
                  {tok}
                </span>
                {i === trueClass && (
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: '#c75b39' }}>
                    p = {probs[i].toFixed(3)}
                  </span>
                )}
              </label>
            ))}
          </div>
        </div>
      </section>

      {/* ── Stage: bar chart + −log(p) curve ────────────────────────────── */}
      <section className="stage" style={{ alignItems: 'flex-start', gap: 40 }}>
        <div>
          <div style={{ fontSize: 13, color: 'var(--ink-soft)', marginBottom: 8 }}>
            预测概率分布（点击柱切换正确词）
          </div>
          <ProbBars probs={probs} trueClass={trueClass} onTrueClass={setTrueClass} />
        </div>
        <div>
          <div style={{ fontSize: 13, color: 'var(--ink-soft)', marginBottom: 8 }}>
            −log(p) 曲线 · 损失随正确词置信度的变化
          </div>
          <NegLogCurve p={pTrue} />
        </div>
      </section>

      {/* ── Readouts: CE value ───────────────────────────────────────────── */}
      <section className="readouts" style={{ justifyContent: 'center', gap: 44 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-soft)', marginBottom: 6 }}>
            p_true
          </div>
          <div style={{ fontSize: 34, fontFamily: 'var(--mono)', fontWeight: 700, color: '#002fa7' }}>
            {pTrue.toFixed(3)}
          </div>
          <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginTop: 4 }}>正确词的预测概率</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-soft)', marginBottom: 6 }}>
            cross-entropy = NLL
          </div>
          <div style={{ fontSize: 34, fontFamily: 'var(--mono)', fontWeight: 700, color: '#c75b39' }}>
            {ce > 9.99 ? '>9.99' : ce.toFixed(3)}
          </div>
          <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginTop: 4 }}>= −log(p_true)</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 11, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--ink-soft)', marginBottom: 6 }}>
            校验
          </div>
          <div style={{ fontSize: 20, fontFamily: 'var(--mono)', fontWeight: 500, color: 'var(--ink)', marginTop: 8 }}>
            CE = NLL ✓
          </div>
          <div style={{ fontSize: 12, color: 'var(--ink-soft)', marginTop: 4 }}>两种叫法，同一个数</div>
        </div>
      </section>

      {/* ── Verdict ─────────────────────────────────────────────────────── */}
      <section className={`verdict ${verdictGood ? 'verdict--eq' : 'verdict--neq'}`}>
        <p>{verdictMsg}</p>
      </section>

      {/* ── Multi-example section ────────────────────────────────────────── */}
      <MultiSection />

      {/* ── Bridge ──────────────────────────────────────────────────────── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            <code>F.cross_entropy(logits, targets)</code> 是 LLM 预训练的损失函数。
            每个 token 位置，模型先用 softmax 把 logits 变成概率分布（第 29 节），
            再取真实下一个词的概率算 −log p，最后全序列平均——
            正是你在上面交互区里操作的每一步。
          </p>
          <p>
            <strong>它等于极大似然（MLE）：</strong>从「最大化联合似然」出发，
            对两边取 log 再除以 −n，就得到「最小化 cross-entropy」。
            NLL（负对数似然）= CE，统计与深度学习的术语在此完全合流——
            不是巧合，是同一件事换了名字。
          </p>
          <p>
            <strong>困惑度 perplexity = exp(CE)</strong> 是语言模型最常见的评估指标：
            perplexity 越低，模型对真实序列的平均置信度越高。
            GPT-2 在 WebText 上约 18，GPT-4 级别的模型已降至个位数。
            这里的 CE 就是第 36 节采样的底层概率的分数——Softmax 产出分布，CE 打分，采样取词，三者首尾相接。
          </p>
        </div>
      </section>

      {/* ── Code block ──────────────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：从 likelihood 到 cross-entropy</h2>
        <CodeBlock code={SNIPPET} language="python" title="cross_entropy.py" />
      </section>

      {/* ── Pager ───────────────────────────────────────────────────────── */}
      <nav className="pager">
        {prev
          ? <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
              <span className="pager-dir">← 上一章</span>
              <span className="pager-title">{prev.num} {prev.title}</span>
            </Link>
          : <span />}
        {next
          ? <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
              <span className="pager-dir">下一章 →</span>
              <span className="pager-title">{next.num} {next.title}{next.status !== 'live' && ' · 规划中'}</span>
            </Link>
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
