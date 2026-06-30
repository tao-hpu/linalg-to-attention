import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// 前 15 个权重固定。第 16 个（w15）是可拖动的「离群值」——
// 拖大它就会迫使 scale 变大，让小权重大量塌缩到 0，
// 从而直观地演示 per-channel quantization 存在的必要。
const FIXED_WEIGHTS: readonly number[] = [
  -1.00, -0.75, -0.55, -0.40, -0.28, -0.18, -0.08,  0.04,
   0.14,  0.26,  0.38,  0.52,  0.68,  0.82,  0.95,
]
const OUTLIER_MIN = 0.95   // 等于次大权重 → 实际上没有离群值
const OUTLIER_MAX = 8.0    // 极端离群值
const OUTLIER_DEFAULT = 5.60

// per-channel 时把这一组权重切成两个「通道」，每通道单独定 scale。
// 离群值落在第 2 通道，于是第 1 通道（全是小权重）能用很细的 scale 恢复精度。
const CHANNEL_SIZE = 8

function buildWeights(outlier: number): number[] {
  return [...FIXED_WEIGHTS, outlier]
}

type BitWidth = 'fp32' | 8 | 4 | 2

interface BitOption {
  bits: BitWidth
  label: string
  sub: string
}

const BIT_OPTIONS: readonly BitOption[] = [
  { bits: 'fp32', label: 'fp32', sub: '4 字节 · 无损' },
  { bits: 8,      label: 'int8', sub: '1 字节 · 4×'  },
  { bits: 4,      label: 'int4', sub: '½ 字节 · 8×'  },
  { bits: 2,      label: 'int2', sub: '¼ 字节 · 16×' },
]

interface QuantResult {
  scales:  readonly number[]   // 每个通道一个 scale（per-tensor 时长度为 1）
  wHat:    readonly number[]
  errors:  readonly number[]
  meanErr: number
  maxErr:  number
  bytesPerWeight: number
}

// 对单个通道做对称量化：scale = max|w| / (2^(bits-1) - 1)
function quantChannel(
  weights: readonly number[],
  maxInt:  number,
  minInt:  number,
): { scale: number; wHat: number[]; errors: number[] } {
  const maxAbs = Math.max(...weights.map(w => Math.abs(w))) || 1e-9
  const scale  = maxAbs / maxInt
  const wHat:   number[] = []
  const errors: number[] = []
  for (const w of weights) {
    const q  = Math.max(minInt, Math.min(maxInt, Math.round(w / scale)))
    const wh = q * scale
    wHat.push(wh)
    errors.push(Math.abs(w - wh))
  }
  return { scale, wHat, errors }
}

// 对称量化。perChannel = false → 整组共用一个 scale；
// perChannel = true → 按 CHANNEL_SIZE 切分通道，每通道单独定 scale。
function runQuant(
  weights:    readonly number[],
  bits:       BitWidth,
  perChannel: boolean,
): QuantResult {
  if (bits === 'fp32') {
    return {
      scales:  [],
      wHat:    [...weights],
      errors:  weights.map(() => 0),
      meanErr: 0,
      maxErr:  0,
      bytesPerWeight: 4,
    }
  }
  const maxInt = Math.pow(2, bits - 1) - 1
  const minInt = -Math.pow(2, bits - 1)
  const step   = perChannel ? CHANNEL_SIZE : weights.length

  const scales: number[] = []
  const wHat:   number[] = []
  const errors: number[] = []

  for (let start = 0; start < weights.length; start += step) {
    const chunk = weights.slice(start, start + step)
    const res   = quantChannel(chunk, maxInt, minInt)
    scales.push(res.scale)
    for (let k = 0; k < chunk.length; k++) {
      wHat.push(res.wHat[k]!)
      errors.push(res.errors[k]!)
    }
  }

  const meanErr = errors.reduce((a, b) => a + b, 0) / errors.length
  const maxErr  = Math.max(...errors)
  const bytesPerWeight = bits === 8 ? 1 : bits === 4 ? 0.5 : 0.25

  return { scales, wHat, errors, meanErr, maxErr, bytesPerWeight }
}

// ─── 权重条形图 ────────────────────────────────────────────────────────────────

function WeightBars({ weights, result, perChannel }: {
  weights:    readonly number[]
  result:     QuantResult
  perChannel: boolean
}) {
  const maxAbsAll = Math.max(...weights.map(w => Math.abs(w)))
  const BAR = 96 // 最大条宽 px

  const rows = weights.map((w, i) => ({
    w,
    wh:  result.wHat[i]!,
    err: result.errors[i]!,
  }))

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '30px 1fr 14px 1fr 84px',
        gap: '2px 8px',
        alignItems: 'center',
        fontSize: 11,
        fontFamily: 'monospace',
        minWidth: 460,
      }}>
        {/* 表头 */}
        <div style={{ color: '#aaa' }}>#</div>
        <div style={{ fontFamily: 'sans-serif', fontWeight: 600, fontSize: 12, color: '#002fa7' }}>
          原始 float
        </div>
        <div />
        <div style={{ fontFamily: 'sans-serif', fontWeight: 600, fontSize: 12 }}>
          量化后（dequant）
        </div>
        <div style={{ fontFamily: 'sans-serif', fontWeight: 600, fontSize: 12, color: '#c75b39' }}>
          误差 |w−ŵ|
        </div>

        {rows.map(({ w, wh, err }, i) => (
          <div key={i} style={{ display: 'contents' }}>
            {/* 通道分隔线：per-channel 时在通道边界处划一条横线 */}
            {perChannel && i > 0 && i % CHANNEL_SIZE === 0 && (
              <div style={{
                gridColumn: '1 / -1',
                height: 0,
                borderTop: '1px dashed #002fa7',
                opacity: 0.35,
                margin: '3px 0',
              }} />
            )}
            {/* 序号 */}
            <div style={{ color: '#bbb', textAlign: 'right' }}>w{i}</div>

            {/* 原始浮点条 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{
                width:       Math.round((Math.abs(w) / maxAbsAll) * BAR),
                height:      10,
                background:  '#002fa7',
                opacity:     w < 0 ? 0.40 : 0.80,
                borderRadius: 2,
                flexShrink:  0,
              }} />
              <span style={{ color: '#444' }}>{w.toFixed(2)}</span>
            </div>

            <div style={{ color: '#ccc', textAlign: 'center' }}>→</div>

            {/* 量化后条 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{
                width:       Math.round((Math.abs(wh) / maxAbsAll) * BAR),
                height:      10,
                background:  '#1b1f24',
                opacity:     wh < 0 ? 0.30 : 0.60,
                borderRadius: 2,
                flexShrink:  0,
              }} />
              <span style={{ color: '#444' }}>{wh.toFixed(3)}</span>
            </div>

            {/* 误差条 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {err > 0 && (
                <div style={{
                  width:       Math.round((err / maxAbsAll) * BAR),
                  height:      10,
                  background:  '#c75b39',
                  opacity:     0.80,
                  borderRadius: 2,
                  flexShrink:  0,
                  minWidth:    1,
                }} />
              )}
              <span style={{ color: err > 0.05 ? '#c75b39' : '#aaa' }}>
                {err.toFixed(3)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── 数轴 ──────────────────────────────────────────────────────────────────────

function NumberLine({ weights, result, bits }: {
  weights: readonly number[]
  result:  QuantResult
  bits:    BitWidth
}) {
  const VIEW_MIN = -1.8
  const VIEW_MAX =  8.6
  const span = VIEW_MAX - VIEW_MIN

  const toP = (v: number): string =>
    `${Math.max(0, Math.min(100, ((v - VIEW_MIN) / span) * 100)).toFixed(2)}%`

  // 量化格点（int8 有 256 个，每隔 16 显一条；int4/int2 全显）。
  // per-channel 时每个通道有各自的 scale → 画出各自的格点。
  const gridTicks: number[] = []
  if (bits !== 'fp32') {
    const maxInt = Math.pow(2, bits - 1) - 1
    const minInt = -Math.pow(2, bits - 1)
    const step   = bits === 8 ? 16 : 1
    for (const scale of result.scales) {
      for (let q = minInt; q <= maxInt; q += step) {
        const gp = q * scale
        if (gp >= VIEW_MIN - 0.1 && gp <= VIEW_MAX + 0.1) {
          gridTicks.push(gp)
        }
      }
    }
  }

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#333', marginBottom: 8 }}>
        数轴：浮点值 snap 到量化格点
      </div>
      <div style={{ position: 'relative', height: 70, userSelect: 'none' }}>

        {/* 轴线 */}
        <div style={{
          position: 'absolute', top: 22, left: 0, right: 0,
          height: 1, background: '#d0d5dd',
        }} />

        {/* 量化格点刻度（IKB 蓝，浅） */}
        {gridTicks.map((gp, i) => (
          <div key={i} style={{
            position: 'absolute', top: 14, left: toP(gp),
            width: 1, height: 16,
            background: '#002fa7', opacity: 0.28,
            transform: 'translateX(-50%)',
          }} />
        ))}

        {/* 零刻度（稍深） */}
        <div style={{
          position: 'absolute', top: 10, left: toP(0),
          width: 2, height: 24,
          background: '#002fa7', opacity: 0.55,
          transform: 'translateX(-50%)',
        }} />

        {/* 原始浮点点（上排，IKB 蓝） */}
        {weights.map((w, i) => (
          <div key={i} style={{
            position: 'absolute', top: 14, left: toP(w),
            width: 8, height: 8, borderRadius: '50%',
            background: '#002fa7', opacity: 0.75,
            transform: 'translateX(-50%)',
            cursor: 'default',
          }} title={`w${i} = ${w}`} />
        ))}

        {/* 量化后点（下排，rust） */}
        {bits !== 'fp32' && result.wHat.map((wh, i) => (
          <div key={i} style={{
            position: 'absolute', top: 32, left: toP(wh),
            width: 8, height: 8, borderRadius: '50%',
            background: '#c75b39', opacity: 0.75,
            transform: 'translateX(-50%)',
            cursor: 'default',
          }} title={`ŵ${i} = ${wh.toFixed(3)}`} />
        ))}

        {/* 图例 */}
        <div style={{
          position: 'absolute', bottom: 2, left: 0,
          fontSize: 11, color: '#888',
          display: 'flex', gap: 14,
        }}>
          <span><span style={{ color: '#002fa7' }}>●</span> 原始 float（上排）</span>
          {bits !== 'fp32' && (
            <span><span style={{ color: '#c75b39' }}>●</span> 量化后（下排）</span>
          )}
          {bits !== 'fp32' && gridTicks.length > 0 && (
            <span style={{ color: '#bbb' }}>│ = 格点</span>
          )}
        </div>

      </div>
    </div>
  )
}

// ─── 代码片段 ──────────────────────────────────────────────────────────────────

const SNIPPET = `import numpy as np

def quantize_symmetric(w, bits, axis=None):
    """对称量化：q = round(w / scale)，scale = max|w| / (2**(bits-1) - 1)

    axis=None → per-tensor：整组共用一个 scale，离群值会把 scale 拉大
    axis=1    → per-channel：每行单独定 scale，离群值只影响自己那一行
    """
    q_max =  2 ** (bits - 1) - 1          # int8 → 127，int4 → 7，int2 → 1
    q_min = -2 ** (bits - 1)              # int8 → -128，int4 → -8，int2 → -2

    max_abs = np.max(np.abs(w), axis=axis, keepdims=True)
    scale   = max_abs / q_max             # 把 float 值域映射到整数格

    q     = np.clip(np.round(w / scale), q_min, q_max)   # 量化 + clip 防溢出
    w_hat = q * scale                                    # 反量化（dequantize）
    err   = np.abs(w - w_hat)
    return q.astype(np.int8), w_hat, scale, err


# 两行 = 两个通道；第 2 行含离群值 5.60
W = np.array([[-1.00, -0.40, -0.08,  0.26,  0.52,  0.82],   # 全是小权重
              [ 0.10,  0.05, -0.12,  0.30, -0.20,  5.60]])  # 含离群值

# per-tensor：整张张量一个 scale，小权重被离群值挤到 round 成 0
_, w_hat_t, _, err_t = quantize_symmetric(W, bits=4, axis=None)

# per-channel：按行（axis=1）各自定 scale，小权重那行精度恢复
_, w_hat_c, _, err_c = quantize_symmetric(W, bits=4, axis=1)

print(err_t.mean(), err_c.mean())     # per-channel 的平均误差明显更小

# 内存：fp32 每权重 4 字节；int8 = 1 B（4×）；int4 = 0.5 B（8×）
factor = 32 / 4                        # 压缩倍数：32 bit / 4 bit = 8×`

// ─── 主组件 ────────────────────────────────────────────────────────────────────

export function Quantization() {
  const [bits, setBits]             = useState<BitWidth>('fp32')
  const [outlier, setOutlier]       = useState<number>(OUTLIER_DEFAULT)
  const [perChannel, setPerChannel] = useState<boolean>(false)

  const weights = buildWeights(outlier)
  const result  = runQuant(weights, bits, perChannel)
  const me      = findChapter('quantization')!
  const { prev, next } = neighbors('quantization')

  const n           = weights.length
  const fp32Bytes   = n * 4
  const curBytes    = n * result.bytesPerWeight
  const compFactor  = fp32Bytes / curBytes   // 1× (fp32) / 4× (int8) / 8× (int4) / 16× (int2)

  // 含离群值那一通道的 scale（per-tensor 时即唯一 scale）与小权重通道的 scale
  const outlierScale = result.scales.length ? result.scales[result.scales.length - 1]! : 0
  const smallScale   = result.scales.length ? result.scales[0]! : 0

  return (
    <article className="page">

      {/* ── 页眉 ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第九部分 · 尾声：接到 LLM 工程
        </div>
        <div className="kicker">第 {me.num} 节 · 终章</div>
        <h1>
          量化与数值
          <span className="zh-sub">把大模型塞进小显卡</span>
        </h1>
        <p className="lede">
          一个典型 LLM 里每个权重都是 <code>float32</code>，占 4 字节。
          <strong>量化（quantization）</strong>的思路是：选一个 <strong>scale</strong>，
          把浮点值域映射到整数格点，只存整数；用时乘回 scale 还原（<strong>dequantize</strong>）。
          代价是轻微的 <strong>rounding error</strong>，收益是内存和速度的 4×–8× 提升。
          拨动下方的位宽，看精度与内存如何此消彼长。
        </p>
      </header>

      {/* ── 位宽选择器 ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">精度 / bits</span>
            <span style={{ color: '#888', fontSize: 13 }}>
              越少的 bit 存储越省——但 rounding error 越大
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
            {BIT_OPTIONS.map(({ bits: b, label, sub }) => (
              <button
                key={label}
                onClick={() => setBits(b)}
                style={{
                  padding:        '7px 18px 6px',
                  border:         `2px solid ${bits === b ? '#002fa7' : '#d0d5dd'}`,
                  borderRadius:   6,
                  background:     bits === b ? '#002fa7' : '#fff',
                  color:          bits === b ? '#fff' : '#333',
                  cursor:         'pointer',
                  fontFamily:     'monospace',
                  fontSize:       13,
                  display:        'flex',
                  flexDirection:  'column',
                  alignItems:     'center',
                  lineHeight:     1.4,
                  transition:     'border-color 0.15s, background 0.15s',
                }}
              >
                <span style={{ fontWeight: 700 }}>{label}</span>
                <span style={{ fontSize: 10, opacity: 0.80 }}>{sub}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 离群值滑块 — 用 rust 边框标出它的特殊地位 */}
        <div
          className="control"
          style={{ borderLeft: '3px solid #c75b39', paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag" style={{ color: '#c75b39' }}>w15</span>
            <span style={{ color: '#888', fontSize: 13, marginLeft: '0.3rem' }}>
              离群值（outlier）——拖大它，看 scale 被撑大、小权重塌缩到 0
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range"
              min={OUTLIER_MIN}
              max={OUTLIER_MAX}
              step={0.05}
              value={outlier}
              onChange={(e) => setOutlier(Number(e.target.value))}
            />
            <span className="param-val" style={{ color: '#c75b39' }}>
              {outlier.toFixed(2)}
            </span>
          </label>
        </div>

        {/* per-channel 开关 */}
        <div
          className="control"
          style={{ borderLeft: '3px solid #002fa7', paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag" style={{ color: '#002fa7' }}>per-channel</span>
            <span style={{ color: '#888', fontSize: 13, marginLeft: '0.3rem' }}>
              关 = 整组共用一个 scale；开 = 每 {CHANNEL_SIZE} 个权重分一通道、各自定 scale
            </span>
          </div>
          <button
            onClick={() => setPerChannel(v => !v)}
            aria-pressed={perChannel}
            style={{
              marginTop: 8,
              padding:      '7px 18px',
              border:       `2px solid ${perChannel ? '#002fa7' : '#d0d5dd'}`,
              borderRadius: 6,
              background:   perChannel ? '#002fa7' : '#fff',
              color:        perChannel ? '#fff' : '#333',
              cursor:       'pointer',
              fontFamily:   'monospace',
              fontSize:     13,
              fontWeight:   700,
              transition:   'border-color 0.15s, background 0.15s',
            }}
          >
            per-channel：{perChannel ? '开（分通道）' : '关（per-tensor）'}
          </button>
        </div>
      </section>

      {/* ── 可视化主区 ── */}
      <section className="stage" style={{ display: 'block' }}>
        <WeightBars weights={weights} result={result} perChannel={perChannel} />
        <div style={{ marginTop: 32 }}>
          <NumberLine weights={weights} result={result} bits={bits} />
        </div>
      </section>

      {/* ── 读数 ── */}
      <section className="readouts">

        <div style={{
          padding: '10px 16px', border: '1px solid #e0e4ea',
          borderRadius: 6, minWidth: 120,
        }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            内存（{n} 个权重）
          </div>
          <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', marginTop: 4 }}>
            {curBytes % 1 === 0 ? curBytes : curBytes.toFixed(1)} B
          </div>
          <div style={{ fontSize: 11, color: '#aaa' }}>fp32 基线：{fp32Bytes} B</div>
        </div>

        <div style={{
          padding: '10px 16px', border: '1px solid #e0e4ea',
          borderRadius: 6, minWidth: 120,
        }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            压缩倍数
          </div>
          <div style={{
            fontSize: 20, fontWeight: 700, fontFamily: 'monospace', marginTop: 4,
            color: compFactor <= 1 ? '#aaa' : '#002fa7',
          }}>
            {compFactor}×
          </div>
          <div style={{ fontSize: 11, color: '#aaa' }}>
            {bits === 'fp32' ? '无压缩' : `每权重 ${result.bytesPerWeight} 字节`}
          </div>
        </div>

        <div style={{
          padding: '10px 16px', border: '1px solid #e0e4ea',
          borderRadius: 6, minWidth: 120,
        }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            平均误差
          </div>
          <div style={{
            fontSize: 20, fontWeight: 700, fontFamily: 'monospace', marginTop: 4,
            color: result.meanErr > 0.10 ? '#c75b39' : '#1b1f24',
          }}>
            {result.meanErr.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: '#aaa' }}>mean |w − ŵ|</div>
        </div>

        <div style={{
          padding: '10px 16px', border: '1px solid #e0e4ea',
          borderRadius: 6, minWidth: 120,
        }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            最大误差
          </div>
          <div style={{
            fontSize: 20, fontWeight: 700, fontFamily: 'monospace', marginTop: 4,
            color: result.maxErr > 0.30 ? '#c75b39' : '#1b1f24',
          }}>
            {result.maxErr.toFixed(4)}
          </div>
          <div style={{ fontSize: 11, color: '#aaa' }}>max |w − ŵ|</div>
        </div>

        {bits !== 'fp32' && (
          <div style={{
            padding: '10px 16px', border: '1px solid #e0e4ea',
            borderRadius: 6, minWidth: 120,
          }}>
            <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Scale{perChannel ? `（${result.scales.length} 通道）` : ''}
            </div>
            <div style={{
              fontSize: perChannel ? 15 : 20, fontWeight: 700,
              fontFamily: 'monospace', marginTop: 4,
              display: 'flex', flexDirection: 'column', lineHeight: 1.35,
            }}>
              {result.scales.map((s, i) => (
                <span key={i} style={{ color: perChannel && i === 0 ? '#002fa7' : '#1b1f24' }}>
                  {perChannel ? `ch${i}: ` : ''}{s.toFixed(4)}
                </span>
              ))}
            </div>
            <div style={{ fontSize: 11, color: '#aaa' }}>
              max|w| / {Math.pow(2, bits - 1) - 1}
            </div>
          </div>
        )}

      </section>

      {/* ── 解读 / 离群值说明 ── */}
      <section className={`verdict ${result.meanErr > 0.08 || bits === 2 ? 'verdict--neq' : 'verdict--eq'}`}>
        {bits === 'fp32' && (
          <p>
            <strong>fp32：零误差，完全精确。</strong>{' '}
            浮点数可以落在数轴上任意位置，不存在 rounding。
            这是量化的参照基线——后面每一个精度损失，都是相对它来算的。
          </p>
        )}
        {bits === 8 && (
          <p>
            <strong>int8：4× 压缩，误差几乎可忽略不计。</strong>{' '}
            256 个格点足以覆盖绝大多数权重分布，平均误差通常很低。
            注意 w15 = {outlier.toFixed(2)} 这个离群值（outlier）：它把 scale 拉到
            约 {outlierScale.toFixed(4)}，虽然小权重的格点间距随之变大，
            但 256 级分辨率仍然够用。把上面的离群值滑块拖到底，再切到
            int4，就能看清 outlier 真正的破坏力。int8 是生产中最常见的量化精度。
          </p>
        )}
        {bits === 4 && !perChannel && (
          <p>
            <strong>int4：离群值（outlier）吃掉了精度。</strong>{' '}
            w15 = {outlier.toFixed(2)} 迫使整组共用 scale = {outlierScale.toFixed(4)}，
            16 个格点间距达 {outlierScale.toFixed(3)}——
            结果是 ±{(outlierScale * 0.5).toFixed(3)} 以内的小权重全部被 round 到 0，
            条形图里你能看到多行误差变红。
            打开上面的 <strong>per-channel</strong> 开关：把 outlier 所在的通道单独处理，
            其他通道用更细的 scale，精度立刻大幅恢复。
          </p>
        )}
        {bits === 4 && perChannel && (
          <p>
            <strong>int4 + per-channel：精度回来了。</strong>{' '}
            现在两个通道各自定 scale——含离群值的通道仍是
            {' '}{outlierScale.toFixed(4)}，但全是小权重的第 1 通道用上了细得多的
            scale = {smallScale.toFixed(4)}，格点间距只有 per-tensor 的一个零头。
            条形图里第 1 通道的误差红条几乎消失：这就是 per-channel quantization
            存在的全部理由——别让一个 outlier 毁掉整组小权重。
          </p>
        )}
        {bits === 2 && (
          <p>
            <strong>int2：只有 4 个格点，精度已接近崩溃。</strong>{' '}
            含离群值通道的 scale = {outlierScale.toFixed(2)}，整个权重空间只被
            {' '}{(-2 * outlierScale).toFixed(2)}、{(-1 * outlierScale).toFixed(2)}、
            0、{outlierScale.toFixed(2)} 四个值覆盖。
            绝大多数小权重都塌到 0，语义信息严重丢失。
            int2 是压力测试——实际工程止步于 int4，且必须搭配精心设计的校准数据（calibration）。
          </p>
        )}
      </section>

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            量化是把大模型塞进小显卡、手机乃至浏览器的关键手段：
            <strong>int8 省一半内存，int4 省到 1/4</strong>。
            配合 LoRA 就是 <strong>QLoRA</strong>（连回第 35 节）——
            用量化的基础模型加载进内存，再在其上做低秩微调，
            让普通 GPU 也能 fine-tune 百亿参数的模型。
            量化不改架构，只改「每个数用几个 bit 存」，
            在精度和资源之间找到工程上的最优平衡。
          </p>
          <p>
            到这里，你从一个向量一路搭到了完整的 LLM 工程。
            几何那条线（向量 → 矩阵 → SVD → 注意力）
            和概率那条线（softmax → 交叉熵 → MLE）
            在 Transformer 里合流，又落到训练、微调、推理——
            量化是工程链上最后一块拼图。
          </p>
          <p>
            <strong>恭喜——这门预科课你修完了。</strong>
            从第 01 节的「一个向量是什么」到这里的「把模型压进内存」，
            你走完了一条从线性代数几何直觉到 LLM 工程实践的完整路径。
            这不是终点，而是你真正读懂 Transformer 论文、
            动手跑 fine-tune 的起点。
          </p>
        </div>
      </section>

      {/* ── 代码 ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：对称量化（symmetric int quant）</h2>
        <CodeBlock code={SNIPPET} language="python" title="quantize.py" />
      </section>

      {/* ── 翻页导航 ── */}
      <nav className="pager">
        {prev
          ? <Link
              className="pager-link prev"
              to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
            >
              <span className="pager-dir">← 上一章</span>
              <span className="pager-title">
                {prev.num} {prev.title}{prev.status !== 'live' && ' · 规划中'}
              </span>
            </Link>
          : <span />}
        {next
          ? <Link
              className="pager-link next"
              to={next.status === 'live' ? `/ch/${next.slug}` : '/'}
            >
              <span className="pager-dir">下一章 →</span>
              <span className="pager-title">
                {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
              </span>
            </Link>
          : <span />}
      </nav>

      <p className="page-foot">
        共 {allChapters.length} 节 · 你在第 {me.num} 节 · 全课完结
      </p>

    </article>
  )
}
