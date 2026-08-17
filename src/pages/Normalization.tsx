import { useState, type CSSProperties } from 'react'
import { CodeBlock } from '../components/CodeBlock'
import { ChapterShell } from '../components/ChapterShell'

// ── colour palette ────────────────────────────────────────────────
const IKB = '#002fa7'   // International Klein Blue — normalised bars
const RUST = '#c75b39'  // rust — original / out-of-range bars

const EPS = 1e-5

// ── shared style objects ──────────────────────────────────────────
const btnStyle: CSSProperties = {
  padding: '5px 12px',
  border: '1px solid #c7cbd0',
  borderRadius: 3,
  background: '#f3f5f7',
  color: '#2c3036',
  fontSize: 13,
  cursor: 'pointer',
  fontFamily: 'inherit',
}

const readoutRowStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'baseline',
  gap: 16,
  padding: '9px 0',
  borderBottom: '1px solid #e4e6e9',
  fontSize: 14,
  color: '#5b6168',
}

const colHeadStyle: CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color: '#5b6168',
  marginBottom: 6,
}

// ── math helpers ──────────────────────────────────────────────────
function mean(xs: number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length
}

function variance(xs: number[], mu: number): number {
  return xs.reduce((s, x) => s + (x - mu) ** 2, 0) / xs.length
}

function layerNorm(xs: number[], gamma: number, beta: number): number[] {
  const mu = mean(xs)
  const v = variance(xs, mu)
  return xs.map((x) => ((x - mu) / Math.sqrt(v + EPS)) * gamma + beta)
}

function rmsNorm(xs: number[], gamma: number): number[] {
  const r = Math.sqrt(xs.reduce((s, x) => s + x * x, 0) / xs.length + EPS)
  return xs.map((x) => (x / r) * gamma)
}

function fmt(n: number, d = 2): string {
  return n.toFixed(d)
}

// ── bar chart (SVG, handles negatives via centre baseline) ────────
interface StdBand { lo: number; hi: number }

function BarChart({
  values,
  barColor,
  title,
  meanLine,
  stdBand,
}: {
  values: number[]
  barColor: string
  title: string
  meanLine?: number
  stdBand?: StdBand
}) {
  const H = 160
  const BAR_W = 22
  const GAP = 8
  const PAD_X = 14
  const W = values.length * (BAR_W + GAP) - GAP + PAD_X * 2

  const absMax = Math.max(...values.map((v) => Math.abs(v)), 0.5) * 1.2
  const baseline = H / 2

  function toY(v: number): number {
    return baseline - (v / absMax) * (H / 2 - 8)
  }

  function barProps(v: number): { y: number; h: number } {
    const y0 = baseline
    const y1 = toY(v)
    return { y: Math.min(y0, y1), h: Math.max(Math.abs(y1 - y0), 1) }
  }

  const bandTop = stdBand !== undefined ? toY(stdBand.hi) : 0
  const bandH =
    stdBand !== undefined ? Math.max(toY(stdBand.lo) - toY(stdBand.hi), 0) : 0

  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#2c3036', marginBottom: 6 }}>
        {title}
      </div>
      <svg
        width={W}
        height={H + 20}
        style={{ display: 'block', margin: '0 auto', overflow: 'visible' }}
      >
        {/* std band (±1σ from mean) */}
        {stdBand !== undefined && (
          <rect
            x={0}
            y={bandTop}
            width={W}
            height={bandH}
            fill="rgba(170,185,215,0.2)"
          />
        )}
        {/* baseline — zero line */}
        <line x1={0} x2={W} y1={baseline} y2={baseline} stroke="#c7cbd0" strokeWidth={1} />
        {/* bars */}
        {values.map((v, i) => {
          const { y, h } = barProps(v)
          return (
            <rect
              key={i}
              x={PAD_X + i * (BAR_W + GAP)}
              y={y}
              width={BAR_W}
              height={h}
              fill={barColor}
              rx={2}
              opacity={0.85}
            />
          )
        })}
        {/* mean line — gray dashed */}
        {meanLine !== undefined && (
          <line
            x1={0}
            x2={W}
            y1={toY(meanLine)}
            y2={toY(meanLine)}
            stroke="#888"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
        )}
        {/* mean label */}
        {meanLine !== undefined && (
          <text
            x={W - 2}
            y={toY(meanLine) - 3}
            textAnchor="end"
            fontSize={9}
            fill="#888"
            fontFamily="monospace"
          >
            μ={fmt(meanLine, 2)}
          </text>
        )}
        {/* zero label */}
        <text
          x={PAD_X - 2}
          y={baseline + 3}
          textAnchor="end"
          fontSize={9}
          fill="#c7cbd0"
          fontFamily="monospace"
        >
          0
        </text>
      </svg>
    </div>
  )
}

// ── preset vectors ────────────────────────────────────────────────
const DEFAULT_VALUES: number[] = [2.1, -0.4, 3.7, 1.2, -1.8, 4.5, 0.6, -0.9]
// large scale: big variance, big absolute values
const LARGE_SCALE: number[] = [14.2, -11.8, 13.5, -9.3, 12.7, -14.5, 10.1, -8.6]
// large mean offset: distribution shifted far from zero
const HIGH_MEAN: number[] = [8.2, 7.4, 9.1, 6.8, 10.3, 7.9, 8.7, 9.5]

// ── code snippet ──────────────────────────────────────────────────
const SNIPPET = `import torch

x = torch.tensor([2.1, -0.4, 3.7, 1.2, -1.8, 4.5, 0.6, -0.9])

# LayerNorm: 减均值 → 除标准差 → 仿射变换
mu    = x.mean()                       # 均值 μ
var   = x.var(unbiased=False)          # 方差 σ²（有偏，population）
x_ln  = (x - mu) / (var + 1e-5).sqrt()

print(f"before  μ={mu:.3f}  σ={x.std(unbiased=False):.3f}")
print(f"LN out  μ={x_ln.mean():.3f}  σ={x_ln.std(unbiased=False):.3f}")
# → before  μ=1.125  σ=2.069
# → LN out  μ=0.000  σ=1.000

gamma, beta = torch.ones(8), torch.zeros(8)
out_ln = x_ln * gamma + beta   # γ=1, β=0 → 不变；训练后学出最优值

# RMSNorm: 只除均方根，跳过均值对中（LLaMA / Mistral 采用）
rms     = (x.pow(2).mean() + 1e-5).sqrt()
out_rms = x / rms * gamma      # 无 β；μ 不归零，但尺度受控`

// ── main component ────────────────────────────────────────────────
export function Normalization() {
  const [values, setValues] = useState<number[]>(DEFAULT_VALUES)
  const [mode, setMode] = useState<'layernorm' | 'rmsnorm'>('layernorm')
  const [gamma, setGamma] = useState(1.0)
  const [beta, setBeta] = useState(0.0)

  // stats on raw input
  const mu = mean(values)
  const vari = variance(values, mu)
  const sigma = Math.sqrt(vari)
  const rawRms = Math.sqrt(values.reduce((s, x) => s + x * x, 0) / values.length)

  // normalised outputs
  const lnValues = layerNorm(values, gamma, beta)
  const rmsValues = rmsNorm(values, gamma)

  const lnMu = mean(lnValues)
  const lnSigma = Math.sqrt(variance(lnValues, lnMu))
  const rmsMu = mean(rmsValues)
  const rmsSigma = Math.sqrt(variance(rmsValues, rmsMu))

  const activeNormed = mode === 'layernorm' ? lnValues : rmsValues
  const activeMu = mode === 'layernorm' ? lnMu : rmsMu
  const activeSigma = mode === 'layernorm' ? lnSigma : rmsSigma
  // RMSNorm 的承诺是「输出 RMS = γ」，不是「输出 σ = γ」——μ≠0 时两者不等。
  const activeRms = Math.sqrt(
    activeNormed.reduce((s, x) => s + x * x, 0) / activeNormed.length,
  )

  function randomize() {
    setValues(Array.from({ length: 8 }, () => +(Math.random() * 14 - 5).toFixed(1)))
  }

  return (
      <ChapterShell
        slug="normalization"
        part="第六部分 · 学习：模型怎么变聪明"
        sub="为什么每层后面都要「拉平」一下？"
        lede={
          <>
        训练过程中，激活值会漂移到悬殊的尺度——均值偏大、方差极宽，
        导致梯度爆炸或消失。<strong>LayerNorm</strong> 把一层的激活向量
        减均值、除标准差，强制拉到 mean 0、variance 1，
        再用可学习的 γ scale 和 β shift 恢复灵活性。
        <strong>RMSNorm</strong> 是更省的现代变体：跳过均值对中，
        只除 root-mean-square。两者核心目标相同——把数字圈回安全范围，
        让梯度稳稳流过几十上百层。
          </>
        }
      >

      {/* ── 控制区 ── */}
      <section className="controls">
        {/* 8 维激活向量 — 跨两列 */}
        <div className="control" style={{ gridColumn: '1 / -1' }}>
          <div className="control-head">
            <span className="slot-tag">x</span>
            <span style={{ fontSize: 14, color: '#5b6168' }}>
              8 维激活向量 — 拖动每条滑块改变数值
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px 28px' }}>
            {values.map((v, i) => (
              <label key={i} className="slider-row">
                <span
                  style={{
                    fontFamily: 'var(--mono)',
                    fontSize: 12,
                    color: '#888',
                    width: 20,
                    flexShrink: 0,
                  }}
                >
                  x{i}
                </span>
                <input
                  type="range"
                  min={-15}
                  max={15}
                  step={0.1}
                  value={Math.max(-15, Math.min(15, v))}
                  style={{ flex: 1, accentColor: RUST }}
                  onChange={(e) => {
                    const next = [...values]
                    next[i] = Number(e.target.value)
                    setValues(next)
                  }}
                />
                <span className="param-val">{fmt(v, 1)}</span>
              </label>
            ))}
          </div>
          <div style={{ marginTop: 14, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button onClick={randomize} style={btnStyle}>
              随机
            </button>
            <button onClick={() => setValues(LARGE_SCALE)} style={btnStyle}>
              放大（大 scale）
            </button>
            <button onClick={() => setValues(HIGH_MEAN)} style={btnStyle}>
              偏移（大均值）
            </button>
            <button onClick={() => setValues(DEFAULT_VALUES)} style={btnStyle}>
              重置
            </button>
          </div>
        </div>

        {/* 模式切换 */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">模式</span>
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            {(['layernorm', 'rmsnorm'] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  ...btnStyle,
                  background: mode === m ? IKB : '#f3f5f7',
                  color: mode === m ? '#fff' : '#2c3036',
                  fontWeight: mode === m ? 700 : 400,
                  borderColor: mode === m ? IKB : '#c7cbd0',
                }}
              >
                {m === 'layernorm' ? 'LayerNorm' : 'RMSNorm'}
              </button>
            ))}
          </div>
          <p style={{ margin: 0, fontSize: 13, color: '#5b6168', lineHeight: 1.55 }}>
            {mode === 'layernorm'
              ? 'LayerNorm 先减均值（对中），再除标准差 → mean 0、variance 1；适合所有 Transformer 场景。'
              : 'RMSNorm 跳过均值对中，只除 root-mean-square → 省一步减法，LLaMA / Mistral 等现代模型采用。'}
          </p>
        </div>

        {/* γ β 控制 */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">γ β</span>
            <span style={{ fontSize: 13, color: '#5b6168' }}>可学习仿射参数</span>
          </div>
          <label className="slider-row">
            <span style={{ fontSize: 13, minWidth: 52, color: '#5b6168', flexShrink: 0 }}>
              γ scale
            </span>
            <input
              type="range"
              min={0.1}
              max={3}
              step={0.05}
              value={gamma}
              style={{ flex: 1, accentColor: IKB }}
              onChange={(e) => setGamma(Number(e.target.value))}
            />
            <span className="param-val">{fmt(gamma, 2)}</span>
          </label>
          {mode === 'layernorm' && (
            <label className="slider-row" style={{ marginTop: 8 }}>
              <span style={{ fontSize: 13, minWidth: 52, color: '#5b6168', flexShrink: 0 }}>
                β shift
              </span>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.05}
                value={beta}
                style={{ flex: 1, accentColor: IKB }}
                onChange={(e) => setBeta(Number(e.target.value))}
              />
              <span className="param-val">{fmt(beta, 2)}</span>
            </label>
          )}
          <p style={{ margin: '10px 0 0', fontSize: 12.5, color: '#5b6168', lineHeight: 1.5 }}>
            初始化 γ=1、β=0；梯度下降后，模型自动学出最优 scale / shift。
            RMSNorm 无 β（不做平移）。
          </p>
        </div>
      </section>

      {/* ── 舞台：bar charts ── */}
      <section className="stage">
        <BarChart
          values={values}
          barColor={RUST}
          title="原始 activations"
          meanLine={mu}
          stdBand={{ lo: mu - sigma, hi: mu + sigma }}
        />
        <div className="arrow-sep">→</div>
        <BarChart
          values={activeNormed}
          barColor={IKB}
          title={mode === 'layernorm' ? 'LayerNorm 后' : 'RMSNorm 后'}
          meanLine={activeMu}
          stdBand={{ lo: activeMu - activeSigma, hi: activeMu + activeSigma }}
        />
      </section>

      {/* ── 读数区 ── */}
      <section className="readouts">
        <div style={{ minWidth: 210 }}>
          <div style={colHeadStyle}>归一化前</div>
          <div style={readoutRowStyle}>
            <span>均值 μ</span>
            <code>{fmt(mu, 3)}</code>
          </div>
          <div style={readoutRowStyle}>
            <span>标准差 σ</span>
            <code>{fmt(sigma, 3)}</code>
          </div>
          <div style={readoutRowStyle}>
            <span>RMS</span>
            <code>{fmt(rawRms, 3)}</code>
          </div>
        </div>

        <div style={{ minWidth: 210 }}>
          <div style={colHeadStyle}>
            {mode === 'layernorm' ? 'LayerNorm 后' : 'RMSNorm 后'}
          </div>
          <div style={readoutRowStyle}>
            <span>均值 μ</span>
            <code
              style={{
                color:
                  mode === 'layernorm' && Math.abs(activeMu - beta) < 0.005
                    ? '#0a7d52'
                    : undefined,
              }}
            >
              {fmt(activeMu, 4)}
            </code>
          </div>
          <div style={readoutRowStyle}>
            <span>标准差 σ</span>
            <code
              style={{
                // 只有 LayerNorm 承诺 σ = γ；RMSNorm 不承诺，就别打绿勾。
                color:
                  mode === 'layernorm' && Math.abs(activeSigma - gamma) < 0.01
                    ? '#0a7d52'
                    : undefined,
              }}
            >
              {fmt(activeSigma, 4)}
            </code>
          </div>
          <div style={readoutRowStyle}>
            <span>RMS</span>
            <code
              style={{
                color:
                  mode === 'rmsnorm' && Math.abs(activeRms - gamma) < 0.01
                    ? '#0a7d52'
                    : undefined,
              }}
            >
              {fmt(activeRms, 4)}
            </code>
          </div>
          <div style={readoutRowStyle}>
            <span>{mode === 'layernorm' ? 'σ 应 ≈ γ' : 'RMS 应 ≈ γ'}</span>
            <code style={{ color: '#5b6168' }}>{fmt(gamma, 2)}</code>
          </div>
        </div>
      </section>

      <section className={`verdict ${mode === 'layernorm' ? 'verdict--eq' : 'verdict--neq'}`}>
        {mode === 'layernorm' ? (
          <p>
            LayerNorm 后：<strong>μ = {fmt(lnMu, 4)}</strong>（≈ β = {fmt(beta, 2)}），
            <strong> σ = {fmt(lnSigma, 4)}</strong>（≈ γ = {fmt(gamma, 2)}）。
            无论原始激活值的分布多悬殊，LayerNorm 都把它拉回这个可控范围——
            γ 和 β 让模型保留「重缩放」的自由度，不失去表达能力。
          </p>
        ) : (
          <p>
            RMSNorm 后：<strong>RMS = {fmt(gamma, 4)}</strong>（被精确拉到 γ，这才是它的承诺），
            <strong> μ = {fmt(rmsMu, 4)}</strong>（未归零），
            <strong> σ = {fmt(rmsSigma, 4)}</strong>。
            注意 <strong>σ 一般不等于 γ</strong>：RMSNorm 除的是均方根而不是标准差，
            两者只在 μ = 0 时相等。点「偏移（大均值）」看得最清楚——
            μ 越远离 0，RMS 就越被均值撑大，σ 被压得越小。
            少一步减法换来更快的速度，代价就是这点：它只保证尺度，不保证分布居中。
          </p>
        )}
      </section>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            现代 Transformer 的每个子层（自注意力、MLP）前后都挂着 LayerNorm 或 RMSNorm。
            主流做法是 <strong>pre-norm</strong>：先归一化、再进子层。
            它和<strong>残差连接</strong>配合，让几十上百层能稳稳堆起来——
            激活值不会滚雪球般放大（梯度不爆），第 25 节链式法则那串乘积也被稳住了（梯度不消）。
          </p>
          <p>
            <strong>RMSNorm</strong> 因为更省、效果相当，被 LLaMA、Mistral 等现代模型采用。
            归一化不改变「方向」，只管「尺度」——把数字圈回安全范围，
            让每一层都能专注学习有用的特征，而不是在救火。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：LayerNorm 与 RMSNorm</h2>
        <CodeBlock code={SNIPPET} language="python" title="normalization.py" />
      </section>

      </ChapterShell>
  )
}
