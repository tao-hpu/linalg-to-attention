import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Brand tokens ──────────────────────────────────────────────────────────────
const IKB  = '#002fa7'
const RUST = '#c75b39'

// ── Dimensions ────────────────────────────────────────────────────────────────
const D    = 8   // weight matrix demo dimension (d × d)
const CELL = 22  // px per heatmap cell
const GAP  = 1   // px gap between cells

// ── Deterministic matrix constructors ─────────────────────────────────────────
// W: fixed "pre-trained" frozen weight, built from trig to look interesting
function buildW(): number[][] {
  return Array.from({ length: D }, (_, i) =>
    Array.from({ length: D }, (_, j) =>
      Math.sin(i * 1.7 + j * 0.9) * Math.cos(i * 0.3 - j * 1.1)
    )
  )
}

// B: left trainable factor (d × r)
function buildB(r: number): number[][] {
  return Array.from({ length: D }, (_, i) =>
    Array.from({ length: r }, (_, k) =>
      Math.sin(i * 2.1 + k * 3.7) * 0.7
    )
  )
}

// A: right trainable factor (r × d)
function buildA(r: number): number[][] {
  return Array.from({ length: r }, (_, k) =>
    Array.from({ length: D }, (_, j) =>
      Math.cos(k * 2.9 + j * 1.3) * 0.7
    )
  )
}

// ── Matrix operations ─────────────────────────────────────────────────────────
// (m × inner) @ (inner × n) → (m × n)
function matmul(P: number[][], Q: number[][]): number[][] {
  const m     = P.length
  const inner = P[0].length
  const n     = Q[0].length
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      let s = 0
      for (let t = 0; t < inner; t++) s += P[i][t] * Q[t][j]
      return s
    })
  )
}

// Element-wise addition (same shape)
function matadd(P: number[][], Q: number[][]): number[][] {
  return P.map((row, i) => row.map((v, j) => v + Q[i][j]))
}

// Global min / max of a matrix
function matRange(M: number[][]): [number, number] {
  let lo = Infinity, hi = -Infinity
  for (const row of M) for (const v of row) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return [lo, hi]
}

// ── Color helpers ─────────────────────────────────────────────────────────────
// white → IKB (#002fa7 = rgb(0, 47, 167))
function ikbCell(val: number, vmin: number, vmax: number): string {
  const t = vmax > vmin ? (val - vmin) / (vmax - vmin) : 0.5
  const r = Math.round(255 * (1 - t))
  const g = Math.round(47  + (255 - 47)  * (1 - t))
  const b = Math.round(167 + (255 - 167) * (1 - t))
  return `rgb(${r},${g},${b})`
}

// white → RUST (#c75b39 = rgb(199, 91, 57))
function rustCell(val: number, vmin: number, vmax: number): string {
  const t = vmax > vmin ? (val - vmin) / (vmax - vmin) : 0.5
  const r = Math.round(199 + (255 - 199) * (1 - t))
  const g = Math.round(91  + (255 - 91)  * (1 - t))
  const b = Math.round(57  + (255 - 57)  * (1 - t))
  return `rgb(${r},${g},${b})`
}

// ── Heatmap component ─────────────────────────────────────────────────────────
type ColorMode = 'ikb' | 'ikb-frozen' | 'rust'

function Heatmap({
  matrix,
  label,
  sublabel,
  colorMode,
  sharedVmin,
  sharedVmax,
}: {
  matrix: number[][]
  label: string
  sublabel?: string
  colorMode: ColorMode
  sharedVmin?: number
  sharedVmax?: number
}) {
  const nrows = matrix.length
  const ncols = matrix[0].length
  const [lMin, lMax] = matRange(matrix)
  const vmin = sharedVmin ?? lMin
  const vmax = sharedVmax ?? lMax

  const pw = ncols * CELL + (ncols - 1) * GAP
  const ph = nrows * CELL + (nrows - 1) * GAP

  const labelColor =
    colorMode === 'rust'         ? RUST
    : colorMode === 'ikb-frozen' ? '#8899cc'
    : IKB

  return (
    <div style={{ textAlign: 'center', flexShrink: 0 }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${ncols}, ${CELL}px)`,
          gridTemplateRows:    `repeat(${nrows}, ${CELL}px)`,
          gap: GAP,
          width:  pw,
          height: ph,
          border: '1.5px solid #e6e8ea',
          borderRadius: 4,
          overflow: 'hidden',
          background: '#e6e8ea',
          opacity: colorMode === 'ikb-frozen' ? 0.5 : 1,
        }}
      >
        {matrix.flatMap((row, ri) =>
          row.map((val, ci) => (
            <div
              key={`${ri}-${ci}`}
              style={{
                background: colorMode === 'rust'
                  ? rustCell(val, vmin, vmax)
                  : ikbCell(val, vmin, vmax),
              }}
            />
          ))
        )}
      </div>
      <div style={{
        marginTop: '0.4rem',
        fontSize: '0.79rem',
        fontWeight: 700,
        color: labelColor,
        lineHeight: 1.3,
      }}>
        {label}
      </div>
      {sublabel !== undefined && (
        <div style={{ fontSize: '0.69rem', color: '#aaa', marginTop: '0.1rem' }}>
          {sublabel}
        </div>
      )}
    </div>
  )
}

// ── Operator separator ────────────────────────────────────────────────────────
function Op({ sym }: { sym: string }) {
  return (
    <div style={{
      fontSize: '1.35rem',
      color: '#bbb',
      alignSelf: 'center',
      flexShrink: 0,
      fontWeight: 300,
      lineHeight: 1,
      userSelect: 'none',
      padding: '0 0.05rem',
    }}>
      {sym}
    </div>
  )
}

// ── Code snippet ──────────────────────────────────────────────────────────────
const SNIPPET = `import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """冻结 W，只训练低秩增量 ΔW = B @ A  (rank ≤ r)"""
    def __init__(self, d: int, r: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d), requires_grad=False)  # frozen
        self.A = nn.Parameter(torch.zeros(r, d))   # shape (r, d)，初始化为 0
        self.B = nn.Parameter(torch.randn(d, r))   # shape (d, r)，randn 初始化
        # 可训练参数: 2·d·r  vs  全量微调 d·d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.W + self.B @ self.A).T
        # 等价省内存写法: x @ self.W.T + (x @ self.A.T) @ self.B.T

# 参数量对比 (d=4096, r=8)
d, r = 4096, 8
full = d * d        # 16,777,216
lora = 2 * d * r   # 65,536
print(f"full={full:,}  LoRA={lora:,}  省了 {full // lora}×")
# 推理合并（零额外延迟）: W_merged = W + B @ A`

// ── Main page export ──────────────────────────────────────────────────────────
export function LoraFinetuning() {
  const [r, setR] = useState(2)

  // W is fixed — built once
  const W = useMemo(() => buildW(), [])

  // Trainable factors and their product react to r
  const B    = useMemo(() => buildB(r), [r])
  const A    = useMemo(() => buildA(r), [r])
  const BA   = useMemo(() => matmul(B, A), [B, A])
  const Weff = useMemo(() => matadd(W, BA), [W, BA])

  // Shared color scale: W and W+BA use the same range for direct visual comparison
  const [sharedMin, sharedMax] = useMemo(() => {
    const [wlo, whi] = matRange(W)
    const [elo, ehi] = matRange(Weff)
    return [Math.min(wlo, elo), Math.max(whi, ehi)]
  }, [W, Weff])

  // Parameter counts
  const fullParams = D * D         // d²
  const loraParams = 2 * D * r    // 2dr
  const pctTrained = (loraParams / fullParams) * 100
  const hasGain    = loraParams < fullParams
  const savingsX   = hasGain ? fullParams / loraParams : 0

  const me           = findChapter('lora-finetuning')!
  const { prev, next } = neighbors('lora-finetuning')

  return (
    <article className="page">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第九部分 · 尾声：接到 LLM 工程
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          LoRA 与高效微调
          <span className="zh-sub">冻结大矩阵，只学一个低秩的小更新</span>
        </h1>
        <p className="lede">
          全量 fine-tuning 要更新权重矩阵里的每一个数——
          一张消费级显卡根本放不下几十亿参数。
          <strong>LoRA</strong>（Low-Rank Adaptation）的答案很优雅：
          把大矩阵 <code>W</code> 冻结（frozen），只让两个小矩阵{' '}
          <code>B</code>（d×r）和 <code>A</code>（r×d）参与训练；
          它们的乘积 <code>ΔW = B·A</code> 是 rank ≤ r 的低秩更新——
          这正是你在第 13 节（矩阵的秩）和第 22 节（低秩近似）学到的数学，
          在这里<strong>兑现</strong>。
          拖动下面的 rank r 滑块，亲手看节省效果和低秩结构如何随 r 变化。
        </p>
      </header>

      {/* ── Slider ──────────────────────────────────────────────────────── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">rank r</span>
            <span>
              r ={' '}
              <strong style={{ color: RUST }}>{r}</strong>
              {r === D
                ? '（= d，ΔW 可满秩，低秩约束消失）'
                : r >= D / 2
                ? '（接近 d，压缩优势减少）'
                : '（≪ d，低秩，压缩显著）'}
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range" min={1} max={D} step={1}
              value={r}
              onChange={(e) => setR(Number(e.target.value))}
            />
            <span className="param-val">r = {r} / {D}</span>
          </label>
        </div>
      </section>

      {/* ── Stage: heatmaps ─────────────────────────────────────────────── */}
      <section
        className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem' }}
      >

        {/* Main equation: W  +  [B · A = ΔW]  =  W+BA */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.8rem',
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}>

          {/* W — frozen */}
          <Heatmap
            matrix={W}
            label="W（frozen ❄️）"
            sublabel={`${D}×${D} · 不参与训练`}
            colorMode="ikb-frozen"
            sharedVmin={sharedMin}
            sharedVmax={sharedMax}
          />

          <Op sym="+" />

          {/* Low-rank trainable group: B · A = ΔW */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.55rem',
            padding: '0.75rem 0.9rem',
            border: `1.5px dashed ${RUST}55`,
            borderRadius: 8,
            background: `${RUST}09`,
          }}>
            <div style={{
              fontSize: '0.73rem',
              color: RUST,
              fontWeight: 700,
              letterSpacing: '0.02em',
            }}>
              ΔW = B · A（trainable, rank ≤ {r}）
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              flexWrap: 'wrap',
              justifyContent: 'center',
            }}>
              <Heatmap
                matrix={B}
                label="B"
                sublabel={`${D}×${r}`}
                colorMode="rust"
              />
              <Op sym="·" />
              <Heatmap
                matrix={A}
                label="A"
                sublabel={`${r}×${D}`}
                colorMode="rust"
              />
              <Op sym="=" />
              <Heatmap
                matrix={BA}
                label="BA = ΔW"
                sublabel={`${D}×${D} · rank ≤ ${r}`}
                colorMode="rust"
              />
            </div>
          </div>

          <Op sym="=" />

          {/* W+BA — effective weight */}
          <Heatmap
            matrix={Weff}
            label="W + BA（有效权重）"
            sublabel={`${D}×${D} · 推理时合并回 W`}
            colorMode="ikb"
            sharedVmin={sharedMin}
            sharedVmax={sharedMax}
          />

        </div>

        {/* Rank annotation */}
        <div style={{
          fontSize: '0.82rem',
          color: '#555',
          background: `${RUST}0c`,
          border: `1px solid ${RUST}2e`,
          borderLeft: `3px solid ${RUST}`,
          borderRadius: '0 6px 6px 0',
          padding: '0.55rem 1.1rem',
          maxWidth: 520,
          lineHeight: 1.65,
        }}>
          <strong style={{ color: RUST }}>rank(BA) ≤ r = {r}</strong>，由构造保证——
          B 只有 {r} 列，BA 的每一列都在 B 的 column space 里，
          最多 {r} 个独立方向（连回第 13 节：秩 = column space 维数）。
          {r === 1 &&
            ' r=1 时 BA 正是 rank-1 outer product，与第 13 节「外积」完全吻合。'}
          {r === D &&
            ` r=d=${D} 时 BA 可以满秩，低秩约束消失。`}
        </div>

      </section>

      {/* ── Readouts ────────────────────────────────────────────────────── */}
      <section className="readouts">
        <div style={{
          display: 'flex',
          gap: '1.5rem',
          flexWrap: 'wrap',
          justifyContent: 'center',
          alignItems: 'flex-start',
        }}>

          {/* Full fine-tuning */}
          <div style={{ textAlign: 'center', minWidth: 130 }}>
            <div style={{ fontSize: '0.77rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.5 }}>
              全参数 fine-tuning<br />
              <code style={{ fontSize: '0.72rem' }}>d² = {D}×{D}</code>
            </div>
            <div style={{ fontSize: '1.9rem', fontWeight: 700, color: '#555', lineHeight: 1 }}>
              {fullParams}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#bbb', marginTop: '0.2rem' }}>
              params（演示 d={D}）
            </div>
          </div>

          {/* LoRA params */}
          <div style={{ textAlign: 'center', minWidth: 130 }}>
            <div style={{ fontSize: '0.77rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.5 }}>
              LoRA<br />
              <code style={{ fontSize: '0.72rem' }}>2·d·r = 2×{D}×{r}</code>
            </div>
            <div style={{ fontSize: '1.9rem', fontWeight: 700, color: RUST, lineHeight: 1 }}>
              {loraParams}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#bbb', marginTop: '0.2rem' }}>
              trainable params
            </div>
          </div>

          {/* Savings */}
          <div style={{ textAlign: 'center', minWidth: 150 }}>
            <div style={{ fontSize: '0.77rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.5 }}>
              节省效果<br />
              <code style={{ fontSize: '0.72rem' }}>仅训 {pctTrained.toFixed(1)}%</code>
            </div>
            {hasGain ? (
              <>
                <div style={{ fontSize: '1.9rem', fontWeight: 700, color: '#1a8a4a', lineHeight: 1 }}>
                  省了 {savingsX.toFixed(1)}×
                </div>
                <div style={{ fontSize: '0.75rem', color: '#1a8a4a', marginTop: '0.2rem' }}>
                  仅训 {pctTrained.toFixed(1)}% 的参数
                </div>
              </>
            ) : (
              <>
                <div style={{ fontSize: '1.9rem', fontWeight: 700, color: RUST, lineHeight: 1 }}>
                  无压缩优势
                </div>
                <div style={{ fontSize: '0.75rem', color: RUST, marginTop: '0.2rem' }}>
                  2dr ≥ d²（r ≥ d/2）
                </div>
              </>
            )}
          </div>

        </div>

        {/* Real-world scale note */}
        <p style={{
          textAlign: 'center',
          fontSize: '0.75rem',
          color: '#aaa',
          marginTop: '1rem',
          lineHeight: 1.6,
        }}>
          † 演示用 d={D}。实际 LLM 中 d=4096，r=8：
          LoRA 参数 65,536 vs 全参数 16,777,216，省了{' '}
          <strong style={{ color: '#555' }}>256×</strong>，仅训练{' '}
          <strong style={{ color: '#555' }}>0.4%</strong>。
        </p>
      </section>

      {/* ── Verdict ─────────────────────────────────────────────────────── */}
      <section className={`verdict ${r < D && hasGain ? 'verdict--eq' : 'verdict--neq'}`}>
        {r === D ? (
          <p>
            <strong>r = d = {D}：ΔW 可以是满秩矩阵，低秩约束完全消失，LoRA 与全参数微调等价。</strong>
            {' '}实践中始终保持 r ≪ d（常用 r = 4、8、16，d = 4096 或更大），
            这才是 LoRA 的工作区间。
          </p>
        ) : hasGain ? (
          <p>
            <strong>
              r = {r}：LoRA 只训练 {loraParams} 个参数（{pctTrained.toFixed(1)}%），
              省了 {savingsX.toFixed(1)}×。
            </strong>
            {' '}有效权重 <code>W + BA</code> 与全量微调的结果接近——
            前提是「微调任务所需的改变量 ΔW 确实是低秩的」。
            这正是 LoRA 的核心假设，也是第 22 节低秩近似的底层逻辑。
            把 r 继续拖大，观察节省空间如何消失。
          </p>
        ) : (
          <p>
            <strong>
              r = {r}（d/2 = {D / 2}）：
              2·d·r = {loraParams} {'≥'} d² = {fullParams}，LoRA 已无压缩优势。
            </strong>
            {' '}注意：这是 d={D} 这个演示维度的边界效应。
            真实场景 d=4096，r 需超过 2048 才会触到此边界——
            而实际 LoRA 只用 r=4∼64，始终节省数百倍参数。
          </p>
        )}
      </section>

      {/* ── Bridge ──────────────────────────────────────────────────────── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            LoRA 是今天微调大模型的默认方法（<strong>PEFT</strong> 库的核心）。
            QLoRA 在它基础上叠加量化（连到第 36 节），
            把 70B 参数的模型压进一张消费级显卡。
            它能成立，全靠你前面学的两件事：
          </p>
          <p>
            <strong>一，权重更新是低秩的。</strong>
            让模型适应新任务，只需要在 W 的方向空间里加几个新方向——
            用 <code>ΔW = B·A</code>（rank ≤ r）这两个薄矩阵就够，
            参数量从 d² 压到 2dr（第 13 节：rank；第 22 节：低秩近似）。
          </p>
          <p>
            <strong>二，训练完可以把 BA 合并回 W。</strong>
            推理时有效权重 <code>W + BA</code> 直接一次矩阵乘法，
            零额外延迟——代价全部在训练阶段支付。
          </p>
          <p>
            一张消费级显卡微调几十亿参数的模型，靠的就是这个数学。
            这是「线性代数 → LLM 工程」最直接的兑现。
          </p>
        </div>
      </section>

      {/* ── Code ────────────────────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：LoRA 层实现</h2>
        <CodeBlock code={SNIPPET} language="python" title="lora_layer.py" />
      </section>

      {/* ── Pager ───────────────────────────────────────────────────────── */}
      <nav className="pager">
        {prev ? (
          <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
            <span className="pager-dir">← 上一章</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : <span />}
        {next ? (
          <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
            <span className="pager-dir">下一章 →</span>
            <span className="pager-title">
              {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
            </span>
          </Link>
        ) : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>

    </article>
  )
}
