import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Brand tokens ───────────────────────────────────────────────────────────────
const IKB  = '#002fa7'
const RUST = '#c75b39'
const GREY = '#d4d7da'

// ── SVD construction parameters ───────────────────────────────────────────────
// Build M by construction from known orthonormal rank-1 layers so the SVD is
// exact-by-design (no numerical SVD needed):
//   u_i = v_i = DCT-II basis vector φ_i  (orthonormal: φ_i · φ_j = δ_ij)
//   M = Σᵢ σᵢ φᵢ φᵢᵀ  →  symmetric PSD; σᵢ are the exact singular values.
const N      = 12                               // grid dimension
const SIGMAS = [9, 5, 2.5, 1.2, 0.6, 0.3]     // σ₁ ≥ σ₂ ≥ … ≥ σ₆ > 0
const K_MAX  = SIGMAS.length                    // 6 layers total
const TOTAL_ENERGY = SIGMAS.reduce((s, v) => s + v * v, 0)

// ── Heatmap cell sizing ────────────────────────────────────────────────────────
const CELL = 20   // px per cell
const GAP  = 1    // px gap between cells

// ── Bar-chart sizing ───────────────────────────────────────────────────────────
const BAR_H   = 110  // max bar height (px)
const BAR_W   = 32   // bar width (px)
const BAR_GAP = 14   // gap between bars (px)
const SVG_W   = K_MAX * (BAR_W + BAR_GAP) - BAR_GAP   // 262 px
const SVG_H   = BAR_H + 40                             // 150 px

// ── DCT-II orthonormal basis ───────────────────────────────────────────────────
// φ₀[j] = 1/√N                                          (DC component)
// φₖ[j] = √(2/N) · cos(π·k·(j+½)/N)    for k ≥ 1
// Satisfies: Σⱼ φᵢ[j]·φₖ[j] = δᵢₖ   (orthonormality)
function buildDCTBasis(n: number, k: number): number[][] {
  return Array.from({ length: k }, (_, i) => {
    const vec = new Array<number>(n)
    if (i === 0) {
      const s = 1 / Math.sqrt(n)
      for (let j = 0; j < n; j++) vec[j] = s
    } else {
      const s = Math.sqrt(2 / n)
      for (let j = 0; j < n; j++) {
        vec[j] = s * Math.cos((Math.PI * i * (j + 0.5)) / n)
      }
    }
    return vec
  })
}

// ── Rank-k reconstruction: M_k = Σᵢ<k σᵢ · basisᵢ · basisᵢᵀ ─────────────────
// k = K_MAX ⟹ M_k = M exactly (zero error), confirming the construction.
function computeRecon(basis: number[][], sigmas: number[], k: number): number[][] {
  const M: number[][] = Array.from({ length: N }, () => new Array<number>(N).fill(0))
  for (let i = 0; i < k; i++) {
    const sigma = sigmas[i]
    const bi    = basis[i]
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        M[r][c] += sigma * bi[r] * bi[c]
      }
    }
  }
  return M
}

// ── Value → IKB-tinted cell color (white at t=0, IKB #002fa7 at t=1) ─────────
function cellColor(val: number, vmin: number, vmax: number): string {
  const t = vmax > vmin ? (val - vmin) / (vmax - vmin) : 0.5
  const r = Math.round(255 * (1 - t))
  const g = Math.round(255 - t * 208)   // 255→47
  const b = Math.round(255 - t * 88)    // 255→167
  return `rgb(${r},${g},${b})`
}

// ── Heatmap grid ───────────────────────────────────────────────────────────────
function Heatmap({ matrix, vmin, vmax, label }: {
  matrix: number[][]
  vmin: number
  vmax: number
  label: string
}) {
  const n    = matrix.length
  const side = n * CELL + (n - 1) * GAP
  return (
    <div style={{ textAlign: 'center', flexShrink: 0 }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${n}, ${CELL}px)`,
          gridTemplateRows: `repeat(${n}, ${CELL}px)`,
          gap: GAP,
          width: side,
          height: side,
          border: '1.5px solid #e6e8ea',
          borderRadius: 4,
          overflow: 'hidden',
          background: '#e6e8ea',
        }}
      >
        {matrix.flatMap((row, r) =>
          row.map((val, c) => (
            <div
              key={`${r}-${c}`}
              style={{ background: cellColor(val, vmin, vmax) }}
            />
          ))
        )}
      </div>
      <div style={{ marginTop: '0.45rem', fontSize: '0.82rem', color: '#555', fontWeight: 600 }}>
        {label}
      </div>
    </div>
  )
}

// ── Singular-value bar chart with cutoff line ─────────────────────────────────
// Bars left of k → IKB (kept); bars right of k → grey (discarded).
// Rust dashed vertical line marks the cutoff.
function SigmaChart({ k }: { k: number }) {
  const maxSigma = SIGMAS[0]
  // x-midpoint of the gap between bar (k-1) and bar k
  const cutX = (k - 1) * (BAR_W + BAR_GAP) + BAR_W + BAR_GAP / 2
  return (
    <div style={{ textAlign: 'center', paddingTop: 10 }}>
      <div style={{ fontSize: '0.82rem', color: '#555', fontWeight: 600, marginBottom: '0.45rem' }}>
        singular values（蓝色 = 保留，灰色 = 丢弃）
      </div>
      <svg
        width={SVG_W}
        height={SVG_H}
        style={{ display: 'block', margin: '0 auto', overflow: 'visible' }}
      >
        {SIGMAS.map((s, i) => {
          const h    = (s / maxSigma) * BAR_H
          const x    = i * (BAR_W + BAR_GAP)
          const y    = BAR_H - h
          const kept = i < k
          const fill = kept ? IKB : GREY
          const tc   = kept ? IKB : '#9aa5b0'
          return (
            <g key={i}>
              <rect x={x} y={y} width={BAR_W} height={h} fill={fill} rx={2} />
              <text
                x={x + BAR_W / 2} y={y - 5}
                textAnchor="middle" fontSize={10} fill={tc}>
                {s}
              </text>
              <text
                x={x + BAR_W / 2} y={BAR_H + 17}
                textAnchor="middle" fontSize={11}
                fill={tc} fontWeight={kept ? 700 : 400}>
                σ{i + 1}
              </text>
            </g>
          )
        })}
        {/* Cutoff line — only rendered when at least one bar is discarded */}
        {k < K_MAX && (
          <line
            x1={cutX} y1={0}
            x2={cutX} y2={BAR_H + 26}
            stroke={RUST} strokeWidth={2} strokeDasharray="4,3"
          />
        )}
      </svg>
      {k < K_MAX && (
        <div style={{ fontSize: '0.78rem', color: RUST, marginTop: '0.25rem' }}>
          ← 保留 &nbsp; cutoff &nbsp; 丢弃 →
        </div>
      )}
    </div>
  )
}

// ── Code snippet ───────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

# SVD 截断：保留前 k 个奇异层（Eckart–Young 最优 rank-k 近似）
U, S, Vt = np.linalg.svd(M, full_matrices=False)
#   M = U @ diag(S) @ Vt  (thin SVD, full_matrices=False)
M_k = (U[:, :k] * S[:k]) @ Vt[:k, :]   # rank-k 近似

# 保留能量 & 相对误差（Frobenius 范数）
energy_ratio = np.sum(S[:k]**2) / np.sum(S**2)
rel_error    = np.sqrt(max(0.0, 1 - energy_ratio))   # ‖M−M_k‖_F / ‖M‖_F

# 存储节省：原矩阵 m×n，rank-k 只需 (m+n)×k 个数
m, n = M.shape
print(f"full: {m*n}  rank-{k}: {(m+n)*k}  压缩比: {m*n/((m+n)*k):.1f}×")

# LoRA：冻结预训练权重 W，只训练低秩增量  ΔW = B @ A
d, r = 4096, 8           # 模型维度 d，rank r ≪ d
B = np.random.randn(d, r)    # shape (d, r)
A = np.random.randn(r, d)    # shape (r, d)
delta_W = B @ A              # shape (d, d)，但 rank(ΔW) ≤ r
print(f"参数: d²={d*d:,}  LoRA 2dr={2*d*r:,}  压缩 {d//(2*r)}×")`

// ── Main export ────────────────────────────────────────────────────────────────
export function LowRank() {
  const [k, setK] = useState(1)

  // DCT-II basis: computed once (N, K_MAX are module constants)
  const basis = useMemo(() => buildDCTBasis(N, K_MAX), [])

  // Full matrix (all K_MAX layers) — constant, reference color scale
  const mFull = useMemo(() => computeRecon(basis, SIGMAS, K_MAX), [basis])

  // Rank-k reconstruction — reactive to slider
  const mRecon = useMemo(() => computeRecon(basis, SIGMAS, k), [basis, k])

  // Global min/max from full matrix: both heatmaps share the same color scale
  const { vmin, vmax } = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const row of mFull) {
      for (const v of row) {
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
    }
    return { vmin: lo, vmax: hi }
  }, [mFull])

  // Energy & error (Eckart–Young)
  const keptEnergy   = SIGMAS.slice(0, k).reduce((acc, v) => acc + v * v, 0)
  const energyRatio  = keptEnergy / TOTAL_ENERGY              // ∈ [0, 1]
  const relError     = Math.sqrt(Math.max(0, 1 - energyRatio)) // ‖M−M_k‖_F/‖M‖_F

  // Storage
  const fullStorage      = N * N                 // 144 numbers
  const rankKStorage     = 2 * N * k             // (m+n)·k numbers
  const compressionRatio = fullStorage / rankKStorage

  const me           = findChapter('low-rank')!
  const { prev, next } = neighbors('low-rank')

  return (
    <article className="page">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第五部分 · 降维：抓住主要矛盾
        </div>
        <div className="kicker">第 {me.num} 节 ★ 核心</div>
        <h1>
          低秩近似
          <span className="zh-sub">几层 singular value 就够了</span>
        </h1>
        <p className="lede">
          SVD 把任意矩阵写成一叠 <strong>rank-1 层</strong>，按重要性排好：
          <code>M = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ⋯</code>。
          只留前 k 层——丢掉小 σ 的部分——就得到最优的 <strong>rank-k 近似</strong>（<em>Eckart–Young 定理</em>）：
          在所有 rank-k 矩阵里，截断 SVD 的 Frobenius 误差最小。
          这是矩阵压缩的数学根基，也是 <strong>LoRA</strong> 的底层逻辑。
          拖动下面的滑块，亲眼看「扔掉小 σ 几乎不丢信息」。
        </p>
      </header>

      {/* ── Slider control ────────────────────────────────────────────────── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">k</span>
            <span>
              保留前{' '}
              <strong style={{ color: IKB }}>{k}</strong>
              {' '}个 singular value 层（共 {K_MAX} 层）
            </span>
          </div>
          <label className="slider-row">
            <input
              type="range" min={1} max={K_MAX} step={1}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
            <span className="param-val">k = {k} / {K_MAX}</span>
          </label>
        </div>
      </section>

      {/* ── Stage: heatmaps + bar chart ───────────────────────────────────── */}
      <section
        className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.75rem' }}
      >
        {/* Two heatmaps side by side */}
        <div style={{
          display: 'flex', gap: '2.5rem', flexWrap: 'wrap',
          justifyContent: 'center', alignItems: 'flex-end',
        }}>
          <Heatmap
            matrix={mFull}
            vmin={vmin} vmax={vmax}
            label={`原图 M（全 ${K_MAX} 层）`}
          />
          <Heatmap
            matrix={mRecon}
            vmin={vmin} vmax={vmax}
            label={`rank-${k} 近似 M_k`}
          />
        </div>

        {/* Singular-value bar chart */}
        <SigmaChart k={k} />
      </section>

      {/* ── Key-metric readouts ───────────────────────────────────────────── */}
      <section className="readouts">
        <div style={{
          display: 'flex', gap: '1.5rem', flexWrap: 'wrap',
          justifyContent: 'center', alignItems: 'flex-start',
        }}>

          {/* Energy retained */}
          <div style={{ textAlign: 'center', minWidth: 130 }}>
            <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.4 }}>
              保留能量<br />
              <code style={{ fontSize: '0.73rem' }}>Σᵢ≤k σᵢ² / Σσᵢ²</code>
            </div>
            <div style={{ fontSize: '2rem', fontWeight: 700, color: IKB, lineHeight: 1 }}>
              {(energyRatio * 100).toFixed(1)}%
            </div>
          </div>

          {/* Relative error */}
          <div style={{ textAlign: 'center', minWidth: 130 }}>
            <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.4 }}>
              相对误差<br />
              <code style={{ fontSize: '0.73rem' }}>‖M−M_k‖_F / ‖M‖_F</code>
            </div>
            <div style={{
              fontSize: '2rem', fontWeight: 700, lineHeight: 1,
              color: relError < 0.05 ? '#1a8a4a' : RUST,
            }}>
              {(relError * 100).toFixed(1)}%
            </div>
          </div>

          {/* Storage */}
          <div style={{ textAlign: 'center', minWidth: 150 }}>
            <div style={{ fontSize: '0.78rem', color: '#888', marginBottom: '0.3rem', lineHeight: 1.4 }}>
              存储量<br />
              <code style={{ fontSize: '0.73rem' }}>{N}×{N} vs (2×{N})×k</code>
            </div>
            <div style={{ fontSize: '1.15rem', fontWeight: 700, lineHeight: 1.3 }}>
              {fullStorage} → {rankKStorage} 个数
            </div>
            <div style={{ fontSize: '0.9rem', color: '#1a8a4a', fontWeight: 600, marginTop: '0.2rem' }}>
              {compressionRatio.toFixed(1)}× 压缩
            </div>
          </div>

        </div>
      </section>

      {/* ── Verdict ──────────────────────────────────────────────────────────── */}
      <section className={`verdict ${energyRatio >= 0.99 ? 'verdict--eq' : 'verdict--neq'}`}>
        {k === K_MAX ? (
          <p>
            <strong>k = {K_MAX}：保留全部 {K_MAX} 层，rank-{K_MAX} 近似 = 原矩阵，误差 0%。</strong>
            {' '}SVD 的完整性：<code>M = Σᵢ σᵢ uᵢ vᵢᵀ</code> 精确成立。
            每层 <code>σᵢ uᵢ vᵢᵀ</code> 是一个 outer product；叠加后无损还原整个矩阵。
          </p>
        ) : energyRatio >= 0.99 ? (
          <p>
            <strong>
              k = {k}：保留 {(energyRatio * 100).toFixed(1)}% 能量，误差仅 {(relError * 100).toFixed(1)}%。
            </strong>
            {' '}Eckart–Young 定理保证：在所有 rank-{k} 矩阵里，没有比截断 SVD 更小误差的选择。
            看两张热图几乎无差别——右侧重建已极为接近左侧原图。
          </p>
        ) : (
          <p>
            k = {k}：保留了 {(energyRatio * 100).toFixed(1)}% 能量，
            相对误差 {(relError * 100).toFixed(1)}%。
            继续拖动滑块——前几个大 σ 贡献了绝大部分「结构」，
            后面小 σ 的层只是细节。<em>大 σ 先，小 σ 后</em>，这是 SVD 的核心排序。
          </p>
        )}
      </section>

      {/* ── Bridge ───────────────────────────────────────────────────────────── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            <strong>LoRA</strong>（Low-Rank Adaptation）：微调大模型时冻结原权重 W，
            只学一个低秩增量 <code>ΔW = B·A</code>——B 是 d×r、A 是 r×d，r ≪ d。
            参数量从 d² 压到 2dr，和这一页「留前 k 个奇异层就够」是<strong>同一道理</strong>：
            权重的改变量 ΔW 近似低秩，让模型适应新任务只需要少数几个新方向。
          </p>
          <p>
            模型压缩、知识蒸馏、KV-cache 降维也都吃这套：找到主要方向（大 σ），
            扔掉次要细节（小 σ），在误差预算内做到最优（Eckart–Young）。
            秩 r = 你愿意保留的「主要方向数」——连回第 13 节（rank）、
            第 21 节（SVD），延伸到第 35 节（LoRA 微调）。
          </p>
        </div>
      </section>

      {/* ── Code block ───────────────────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：SVD 截断 · 能量比 · LoRA 参数量</h2>
        <CodeBlock code={SNIPPET} language="python" title="low_rank_approx.py" />
      </section>

      {/* ── Pager ────────────────────────────────────────────────────────────── */}
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
