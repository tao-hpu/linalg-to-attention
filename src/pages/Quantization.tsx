import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// 固定权重样本（16 个）。最后一个 5.60 是故意的「离群值」——
// 它会迫使 int4 采用过大的 scale，让小权重大量塌缩到 0，
// 从而直观地演示 per-channel quantization 存在的必要。
const WEIGHTS: readonly number[] = [
  -1.00, -0.75, -0.55, -0.40, -0.28, -0.18, -0.08,  0.04,
   0.14,  0.26,  0.38,  0.52,  0.68,  0.82,  0.95,  5.60,
]

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
  scale: number
  wHat: readonly number[]
  errors: readonly number[]
  meanErr: number
  maxErr: number
  bytesPerWeight: number
}

// 对称量化：scale = max|w| / (2^(bits-1) - 1)
function runQuant(weights: readonly number[], bits: BitWidth): QuantResult {
  if (bits === 'fp32') {
    return {
      scale: 0,
      wHat: [...weights],
      errors: weights.map(() => 0),
      meanErr: 0,
      maxErr: 0,
      bytesPerWeight: 4,
    }
  }
  const maxAbs = Math.max(...weights.map(w => Math.abs(w)))
  const maxInt = Math.pow(2, bits - 1) - 1
  const minInt = -Math.pow(2, bits - 1)
  const scale  = maxAbs / maxInt

  const pairs = weights.map(w => {
    const q  = Math.max(minInt, Math.min(maxInt, Math.round(w / scale)))
    const wh = q * scale
    return { wh, err: Math.abs(w - wh) }
  })

  const wHat   = pairs.map(p => p.wh)
  const errors = pairs.map(p => p.err)
  const meanErr = errors.reduce((a, b) => a + b, 0) / errors.length
  const maxErr  = Math.max(...errors)
  const bytesPerWeight = bits === 8 ? 1 : bits === 4 ? 0.5 : 0.25

  return { scale, wHat, errors, meanErr, maxErr, bytesPerWeight }
}

// ─── 权重条形图 ────────────────────────────────────────────────────────────────

function WeightBars({ weights, result }: {
  weights: readonly number[]
  result:  QuantResult
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
  const VIEW_MAX =  7.0
  const span = VIEW_MAX - VIEW_MIN

  const toP = (v: number): string =>
    `${Math.max(0, Math.min(100, ((v - VIEW_MIN) / span) * 100)).toFixed(2)}%`

  // 量化格点（int8 有 256 个，每隔 16 显一条；int4/int2 全显）
  const gridTicks: number[] = []
  if (bits !== 'fp32') {
    const maxInt = Math.pow(2, bits - 1) - 1
    const minInt = -Math.pow(2, bits - 1)
    const step   = bits === 8 ? 16 : 1
    for (let q = minInt; q <= maxInt; q += step) {
      const gp = q * result.scale
      if (gp >= VIEW_MIN - 0.1 && gp <= VIEW_MAX + 0.1) {
        gridTicks.push(gp)
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

const SNIPPET = `// 对称量化（Symmetric Int Quantization）
function quantize(weights: number[], bits: number) {
  const maxAbs = Math.max(...weights.map(Math.abs))
  const maxInt = 2 ** (bits - 1) - 1          // int8 → 127，int4 → 7
  const minInt = -(2 ** (bits - 1))            // int8 → -128，int4 → -8
  const scale  = maxAbs / maxInt               // 把 float 值域映射到整数格

  // 量化：除以 scale 取整，clip 防溢出
  const q     = weights.map(w =>
    Math.max(minInt, Math.min(maxInt, Math.round(w / scale)))
  )
  // 反量化（dequantize）：乘回 scale
  const wHat  = q.map(qi => qi * scale)
  const error = weights.map((w, i) => Math.abs(w - wHat[i]))

  // 内存：fp32 每权重 4 字节；int8 = 1 B（4×）；int4 = 0.5 B（8×）
  const quantBytes = weights.length * (bits / 8)
  const factor     = (weights.length * 4) / quantBytes  // 压缩倍数
  return { scale, q, wHat, error, factor }
}`

// ─── 主组件 ────────────────────────────────────────────────────────────────────

export function Quantization() {
  const [bits, setBits] = useState<BitWidth>('fp32')

  const result = runQuant(WEIGHTS, bits)
  const me     = findChapter('quantization')!
  const { prev, next } = neighbors('quantization')

  const n           = WEIGHTS.length
  const fp32Bytes   = n * 4
  const curBytes    = n * result.bytesPerWeight
  const compFactor  = fp32Bytes / curBytes   // 1× (fp32) / 4× (int8) / 8× (int4) / 16× (int2)

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
      </section>

      {/* ── 可视化主区 ── */}
      <section className="stage" style={{ display: 'block' }}>
        <WeightBars weights={WEIGHTS} result={result} />
        <div style={{ marginTop: 32 }}>
          <NumberLine weights={WEIGHTS} result={result} bits={bits} />
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
              Scale
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'monospace', marginTop: 4 }}>
              {result.scale.toFixed(4)}
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
            256 个格点足以覆盖绝大多数权重分布，平均误差通常低于 0.01。
            注意 w15 = 5.60 这个离群值（outlier）：它把 scale 拉到
            约 {result.scale.toFixed(4)}，虽然小权重的格点间距随之变大，
            但 256 级分辨率仍然够用——
            <strong>per-channel quantization</strong> 会把每一层的每一「行」单独定 scale，
            进一步减小 outlier 的影响。int8 是生产中最常见的量化精度。
          </p>
        )}
        {bits === 4 && (
          <p>
            <strong>int4：离群值（outlier）吃掉了精度。</strong>{' '}
            w15 = 5.60 迫使 scale = {result.scale.toFixed(4)}，
            16 个格点间距达 {result.scale.toFixed(3)}——
            结果是 ±{(result.scale * 0.5).toFixed(3)} 以内的小权重全部被 round 到 0，
            条形图里你能看到多行误差变红。
            这就是 <strong>per-channel scale</strong> 存在的原因：
            把 outlier 所在的行单独处理，其他行用更细的 scale，
            精度就能大幅恢复。
          </p>
        )}
        {bits === 2 && (
          <p>
            <strong>int2：只有 4 个格点，精度已接近崩溃。</strong>{' '}
            scale = {result.scale.toFixed(2)}，整个权重空间只被
            {' '}{(-2 * result.scale).toFixed(2)}、{(-1 * result.scale).toFixed(2)}、
            0、{result.scale.toFixed(2)} 四个值覆盖。
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
            配合 LoRA 就是 <strong>QLoRA</strong>（连回第 34 节）——
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
        <CodeBlock code={SNIPPET} language="typescript" title="quantize.ts" />
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
