import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── design tokens ─────────────────────────────────────────────────────────────
const IKB  = '#002fa7'
const RUST = '#c75b39'
const IKB_RGB  = '0,47,167'
const RUST_RGB = '199,91,57'

// ── fixed dimensions ──────────────────────────────────────────────────────────
const N = 5
const D = 4   // head_dim; sqrt(D) = 2 — intentionally clean

const TOKENS: readonly string[] = ['猫', '坐', '在', '垫子', '上']

// ── token embeddings X (N×D) ──────────────────────────────────────────────────
const X_BASE: number[][] = [
  [ 1.2,  0.5, -0.3,  0.8],   // 猫   animate noun
  [ 0.2,  1.0,  0.6, -0.4],   // 坐   action verb
  [-0.1,  0.3,  0.8,  0.2],   // 在   preposition
  [ 0.9,  0.4,  0.0,  0.7],   // 垫子 inanimate noun
  [-0.2,  0.0,  0.5,  0.4],   // 上   spatial
]

// ── projection weight matrices (D×D) ─────────────────────────────────────────
const W_Q: number[][] = [
  [ 0.5, -0.2,  0.3,  0.1],
  [ 0.0,  0.8, -0.1,  0.4],
  [-0.3,  0.1,  0.6,  0.2],
  [ 0.2,  0.3,  0.0,  0.7],
]
const W_K: number[][] = [
  [ 0.6,  0.1, -0.2,  0.3],
  [ 0.1,  0.7,  0.3, -0.1],
  [ 0.0, -0.2,  0.5,  0.4],
  [ 0.4,  0.2,  0.1,  0.6],
]
const W_V: number[][] = [
  [ 0.8,  0.0,  0.1, -0.2],
  [ 0.0,  0.6,  0.3,  0.1],
  [-0.1,  0.2,  0.7,  0.0],
  [ 0.3, -0.1,  0.0,  0.5],
]

// ── math helpers ──────────────────────────────────────────────────────────────
function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length
  const k = B.length
  const n = B[0].length
  const C: number[][] = Array.from({ length: m }, () => Array(n).fill(0))
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let p = 0; p < k; p++)
        C[i][j] += A[i][p] * B[p][j]
  return C
}

function transposeM(A: number[][]): number[][] {
  const m = A.length
  const n = A[0].length
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => A[j][i])
  )
}

function softmaxRow(row: number[]): number[] {
  const mx = Math.max(...row)
  const exps = row.map(x => Math.exp(x - mx))
  const s = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / s)
}

function softmaxMat(M: number[][]): number[][] {
  return M.map(row => softmaxRow(row))
}

function fmt2(n: number): string { return n.toFixed(2) }
function fmt3(n: number): string { return n.toFixed(3) }

// ── heatmap cell sizing ───────────────────────────────────────────────────────
const CELL = 34   // px per cell in full-size heatmaps
const SMALL = 24  // px per cell in W_Q / W_K / W_V heatmaps

// ── Heatmap component ─────────────────────────────────────────────────────────
function Heatmap({
  M, title, rowLabels, colLabels, highlightRow, cellPx = CELL,
}: {
  M: number[][]
  title?: string
  rowLabels?: string[]
  colLabels?: string[]
  highlightRow?: number
  cellPx?: number
}) {
  const allVals = M.flatMap(r => r)
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const range = maxV === minV ? 1 : maxV - minV
  const labelW = rowLabels ? 36 : 0

  return (
    <div style={{ display: 'inline-block' }}>
      {title && (
        <div style={{
          fontSize: 11, fontWeight: 700, color: '#555',
          marginBottom: 4, letterSpacing: '0.04em', textTransform: 'uppercase',
        }}>
          {title}
        </div>
      )}
      {/* col labels */}
      {colLabels && (
        <div style={{ display: 'flex', marginLeft: labelW }}>
          {colLabels.map((l, j) => (
            <div key={j} style={{
              width: cellPx, textAlign: 'center',
              fontSize: 10, color: '#888', fontFamily: 'monospace',
            }}>
              {l}
            </div>
          ))}
        </div>
      )}
      {M.map((row, i) => {
        const isHiRow = i === highlightRow
        return (
          <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
            {rowLabels && (
              <div style={{
                width: labelW, textAlign: 'right', paddingRight: 4,
                fontSize: 11, color: isHiRow ? RUST : '#666',
                fontWeight: isHiRow ? 700 : 400,
                flexShrink: 0,
              }}>
                {rowLabels[i]}
              </div>
            )}
            {row.map((val, j) => {
              const t = (val - minV) / range
              const alpha = 0.08 + t * 0.82
              const rgb = isHiRow ? RUST_RGB : IKB_RGB
              const textColor = t > 0.55 ? '#fff' : '#333'
              return (
                <div
                  key={j}
                  title={`${fmt3(val)}`}
                  style={{
                    width: cellPx, height: cellPx,
                    background: `rgba(${rgb},${alpha})`,
                    border: isHiRow
                      ? `1px solid rgba(${RUST_RGB},0.5)`
                      : `1px solid rgba(${IKB_RGB},0.12)`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: cellPx < 28 ? 8 : 9,
                    color: textColor,
                    fontFamily: 'monospace',
                    boxSizing: 'border-box',
                    transition: 'background 0.15s ease',
                  }}
                >
                  {fmt2(val)}
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}

// ── ShapeTag: inline "(m×k)·(k×n)=(m×n)" annotation ─────────────────────────
function ShapeTag({ expr }: { expr: string }) {
  return (
    <div style={{
      fontFamily: 'monospace', fontSize: 12,
      color: '#555', background: '#f0f4ff',
      border: `1px solid rgba(${IKB_RGB},0.2)`,
      borderRadius: 4, padding: '3px 8px',
      display: 'inline-block', marginTop: 6,
    }}>
      {expr}
    </div>
  )
}

// ── Attention arrows SVG ──────────────────────────────────────────────────────
const TOKEN_CX = [48, 134, 218, 318, 428]   // x-centres for 5 tokens (500px wide)
const TOKEN_CY = 40
const TOKEN_R  = 18

function AttentionArrows({
  weights, selectedQuery,
}: {
  weights: number[]
  selectedQuery: number
}) {
  const svgW = 500
  const svgH = 130

  return (
    <svg
      viewBox={`0 0 ${svgW} ${svgH}`}
      style={{ width: '100%', maxWidth: svgW, display: 'block', margin: '0 auto', overflow: 'visible' }}
      aria-label={`Token ${TOKENS[selectedQuery]} 的注意力分布`}
    >
      {/* Arcs (draw under circles) */}
      {TOKENS.map((_, j) => {
        const w = weights[j]
        if (w < 0.015) return null
        const qx = TOKEN_CX[selectedQuery]
        const kx = TOKEN_CX[j]
        const midx = (qx + kx) / 2
        const dist = Math.abs(kx - qx)
        const arcDepth = TOKEN_CY + TOKEN_R + 20 + dist * 0.22
        const d = `M ${qx} ${TOKEN_CY + TOKEN_R - 2} Q ${midx} ${arcDepth} ${kx} ${TOKEN_CY + TOKEN_R - 2}`
        return (
          <path
            key={j}
            d={d}
            fill="none"
            stroke={RUST}
            strokeWidth={Math.max(1.2, w * 10)}
            strokeOpacity={0.15 + w * 0.85}
            strokeLinecap="round"
          />
        )
      })}

      {/* Token circles */}
      {TOKENS.map((tok, i) => {
        const isQuery = i === selectedQuery
        return (
          <g key={i}>
            <circle
              cx={TOKEN_CX[i]}
              cy={TOKEN_CY}
              r={TOKEN_R}
              fill={isQuery ? `rgba(${RUST_RGB},0.18)` : `rgba(${IKB_RGB},0.10)`}
              stroke={isQuery ? RUST : IKB}
              strokeWidth={isQuery ? 2.5 : 1.5}
            />
            <text
              x={TOKEN_CX[i]}
              y={TOKEN_CY + 5}
              textAnchor="middle"
              fontSize={tok.length > 1 ? 11 : 14}
              fontWeight={700}
              fill={isQuery ? RUST : IKB}
              style={{ fontFamily: 'sans-serif', userSelect: 'none' }}
            >
              {tok}
            </text>
          </g>
        )
      })}

      {/* Weight labels below arcs */}
      {TOKENS.map((_, j) => {
        const w = weights[j]
        const qx = TOKEN_CX[selectedQuery]
        const kx = TOKEN_CX[j]
        const midx = (qx + kx) / 2
        const dist = Math.abs(kx - qx)
        const labelY = TOKEN_CY + TOKEN_R + 28 + dist * 0.22
        return (
          <text
            key={j}
            x={midx}
            y={Math.min(labelY, svgH - 6)}
            textAnchor="middle"
            fontSize={10}
            fill={RUST}
            fontWeight={600}
            fontFamily="monospace"
            fillOpacity={0.3 + w * 0.7}
          >
            {w.toFixed(2)}
          </text>
        )
      })}

      {/* "Query →" label */}
      <text
        x={TOKEN_CX[selectedQuery]}
        y={TOKEN_CY - TOKEN_R - 5}
        textAnchor="middle"
        fontSize={9}
        fill={RUST}
        fontWeight={700}
        letterSpacing={1}
        style={{ fontFamily: 'sans-serif' }}
      >
        Query
      </text>
    </svg>
  )
}

// ── code snippet ──────────────────────────────────────────────────────────────
const SNIPPET = `\
import numpy as np
import math

def self_attention(X, Wq, Wk, Wv):
    """
    X  : (n, d) — n token embeddings, each dim d
    Wq, Wk, Wv : (d, d) — projection weight matrices
    Returns output (n, d) and attention weights (n, n)
    """
    Q = X @ Wq          # (n, d)  — project to Query space
    K = X @ Wk          # (n, d)  — project to Key space
    V = X @ Wv          # (n, d)  — project to Value space

    d = Q.shape[-1]
    scores  = Q @ K.T / math.sqrt(d)   # (n, n) — scaled dot products
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)   # softmax, rows sum to 1

    out = weights @ V   # (n, n) @ (n, d) = (n, d) — weighted sum of Values

    # shapes at each step:
    # X:(n,d) → Q:(n,d), K:(n,d), V:(n,d)
    # Q:(n,d) @ Kᵀ:(d,n) = scores:(n,n)
    # softmax(scores):(n,n) @ V:(n,d) = out:(n,d)
    return out, weights

# ── demo ──────────────────────────────────────────────────────────────────────
n, d = 5, 4
rng  = np.random.default_rng(42)
X  = rng.normal(size=(n, d))
Wq = rng.normal(size=(d, d)) * 0.5
Wk = rng.normal(size=(d, d)) * 0.5
Wv = rng.normal(size=(d, d)) * 0.5

out, W = self_attention(X, Wq, Wk, Wv)
print("output shape:", out.shape)          # (5, 4)
print("weights shape:", W.shape)           # (5, 5)
print("row sums:", W.sum(axis=-1).round(6))  # [1. 1. 1. 1. 1.]`

// ── main page component ───────────────────────────────────────────────────────
export function SelfAttention() {
  const [selectedQuery, setSelectedQuery] = useState(0)
  const [useScaling, setUseScaling]       = useState(true)
  const [nudgeVal, setNudgeVal]           = useState(0)

  const me             = findChapter('self-attention')!
  const { prev, next } = neighbors('self-attention')

  // Build X with optional nudge on token 0 (猫), dimension 0
  const X: number[][] = X_BASE.map((row, i) =>
    i === 0
      ? [row[0] + nudgeVal, row[1], row[2], row[3]]
      : [...row]
  )

  const Q = matmul(X, W_Q)
  const K = matmul(X, W_K)
  const V = matmul(X, W_V)

  const KT        = transposeM(K)                               // (D×N)
  const rawScores = matmul(Q, KT)                               // (N×N)
  const scale     = useScaling ? Math.sqrt(D) : 1.0
  const scaledScores = rawScores.map(row => row.map(v => v / scale)) // (N×N)
  const attnWeights  = softmaxMat(scaledScores)                 // (N×N) rows sum to 1
  const output       = matmul(attnWeights, V)                   // (N×D)

  const selWeights = attnWeights[selectedQuery]
  const selOutput  = output[selectedQuery]

  // Verify softmax row sum for verdict
  const rowSum = selWeights.reduce((a, b) => a + b, 0)

  const tokenLabels = TOKENS.slice() as string[]
  const dimLabels   = ['d₀', 'd₁', 'd₂', 'd₃']

  // Column labels for N×N matrices: token names
  const nColLabels = TOKENS.slice() as string[]

  // saturation check: max weight in selected row
  const maxW = Math.max(...selWeights)

  return (
    <article className="page">

      {/* ── 页头 ── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第八部分 · 合成：亲手拼出注意力
        </div>
        <div className="kicker">第 {me.num} 节 ★</div>
        <h1>
          自注意力
          <span className="zh-sub">Attention(Q, K, V) = softmax(QKᵀ/√d) · V</span>
        </h1>
        <p className="lede">
          这是整门课的高潮。自注意力（self-attention）让序列里的每个 token
          都能<strong>看见每一个其他 token，并按相关性把它们的信息加权汇入自己</strong>。
          整个过程就一个公式：把每个 token 的 embedding 投影成
          <strong> Query、Key、Value</strong>（<code>Q=XW_Q</code>,
          {' '}<code>K=XW_K</code>, <code>V=XW_V</code>）；
          用点积打分（第 04 节），除以 √d 缩放（第 06 节）；
          softmax 归一化成权重（第 29 节）；
          最后做加权和 <code>output = softmax(QKᵀ/√d)·V</code>（第 10 节的形状逻辑在此收口）。
          下面你亲手跑完整个 pipeline，每一步都有实数。
        </p>
      </header>

      {/* ── 控制区 ── */}
      <section className="controls">
        {/* query 选择 */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">Query token</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.4rem' }}>
              点选一个 token，看它注意哪些位置
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            {TOKENS.map((tok, i) => (
              <button
                key={i}
                onClick={() => setSelectedQuery(i)}
                style={{
                  padding: '6px 14px',
                  fontWeight: 700,
                  fontSize: 15,
                  cursor: 'pointer',
                  border: `2px solid ${i === selectedQuery ? RUST : '#ccc'}`,
                  borderRadius: 6,
                  background: i === selectedQuery ? `rgba(${RUST_RGB},0.10)` : '#f8f8f8',
                  color: i === selectedQuery ? RUST : '#333',
                  transition: 'all 0.1s ease',
                  fontFamily: 'sans-serif',
                }}
              >
                {tok}
              </button>
            ))}
          </div>
        </div>

        {/* √d scaling toggle */}
        <div
          className="control"
          style={{ borderLeft: `3px solid ${IKB}`, paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag" style={{ color: IKB }}>√d 缩放</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.4rem' }}>
              关掉看 softmax 饱和
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button
              onClick={() => setUseScaling(true)}
              style={{
                padding: '5px 16px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
                border: `2px solid ${useScaling ? IKB : '#ccc'}`,
                borderRadius: 6,
                background: useScaling ? IKB : '#f8f8f8',
                color: useScaling ? '#fff' : '#555',
              }}
            >
              ÷√{D} = ÷{Math.sqrt(D).toFixed(0)} ✓
            </button>
            <button
              onClick={() => setUseScaling(false)}
              style={{
                padding: '5px 16px', fontWeight: 700, fontSize: 13, cursor: 'pointer',
                border: `2px solid ${!useScaling ? RUST : '#ccc'}`,
                borderRadius: 6,
                background: !useScaling ? `rgba(${RUST_RGB},0.10)` : '#f8f8f8',
                color: !useScaling ? RUST : '#555',
              }}
            >
              不缩放（÷1）
            </button>
          </div>
          {!useScaling && (
            <p style={{ fontSize: 12, color: RUST, margin: '6px 0 0', fontWeight: 600 }}>
              分数变大 → softmax 趋向 one-hot（饱和），梯度接近零。
              这就是为什么要除以 √d（第 06 节）。
            </p>
          )}
        </div>

        {/* 扰动滑块 */}
        <div
          className="control"
          style={{ borderLeft: `3px solid rgba(${IKB_RGB},0.4)`, paddingLeft: '0.75rem' }}
        >
          <div className="control-head">
            <span className="slot-tag">扰动「猫」embedding[0]</span>
            <span style={{ fontSize: '0.74rem', color: '#666', marginLeft: '0.4rem' }}>
              nudge X[0][0]，看注意力权重随之变化
            </span>
          </div>
          <label className="slider-row" style={{ marginTop: 8 }}>
            <input
              type="range" min={-1.5} max={1.5} step={0.05}
              value={nudgeVal}
              onChange={(e) => setNudgeVal(Number(e.target.value))}
            />
            <span className="param-val" style={{ color: IKB }}>
              {nudgeVal >= 0 ? '+' : ''}{fmt2(nudgeVal)}
            </span>
          </label>
          <p style={{ fontSize: 11, color: '#888', margin: '4px 0 0' }}>
            base = {fmt2(X_BASE[0][0])} → 当前 = {fmt2(X_BASE[0][0] + nudgeVal)}
          </p>
        </div>
      </section>

      {/* ── Stage 1: 投影 X → Q, K, V ── */}
      <section className="stage" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '1.8rem' }}>
        <h2 className="sec-h" style={{ marginBottom: '0.4rem' }}>
          第一步：投影 — X 变出 Query、Key、Value
        </h2>
        <p style={{ color: '#444', fontSize: '0.9rem', marginBottom: '1rem' }}>
          每个 token 的 embedding（X 的行）同时投影到三个子空间：
          <code style={{ margin: '0 4px' }}>Q = X·W_Q</code>，
          <code style={{ margin: '0 4px' }}>K = X·W_K</code>，
          <code style={{ margin: '0 4px' }}>V = X·W_V</code>。
          W_Q / W_K / W_V 是网络的可学习参数。下面的 heatmap 是每个矩阵的实际数值（悬停看精确值）。
        </p>

        {/* X + W matrices */}
        <div style={{
          display: 'flex', gap: '2.5rem', flexWrap: 'wrap', alignItems: 'flex-start',
        }}>
          {/* X */}
          <div>
            <Heatmap
              M={X}
              title={`X  (${N}×${D} token embeddings)`}
              rowLabels={tokenLabels}
              colLabels={dimLabels}
              highlightRow={selectedQuery}
            />
            <ShapeTag expr={`X : ${N}×${D}`} />
          </div>

          {/* W matrices stacked */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, justifyContent: 'center' }}>
            {([['W_Q', W_Q], ['W_K', W_K], ['W_V', W_V]] as [string, number[][]][]).map(([name, W]) => (
              <div key={name}>
                <Heatmap M={W} title={`${name} (${D}×${D})`} cellPx={SMALL} />
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, alignSelf: 'center' }}>
            <span style={{ fontSize: 28, color: '#bbb', lineHeight: 1 }}>→</span>
            <span style={{ fontSize: 28, color: '#bbb', lineHeight: 1 }}>→</span>
            <span style={{ fontSize: 28, color: '#bbb', lineHeight: 1 }}>→</span>
          </div>

          {/* Q, K, V stacked */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {([['Q', Q], ['K', K], ['V', V]] as [string, number[][]][]).map(([name, M]) => (
              <div key={name}>
                <Heatmap
                  M={M}
                  title={`${name} = X·W_${name}  (${N}×${D})`}
                  rowLabels={tokenLabels}
                  highlightRow={selectedQuery}
                  cellPx={SMALL}
                />
              </div>
            ))}
          </div>
        </div>

        <ShapeTag expr={`(${N}×${D}) · (${D}×${D}) = (${N}×${D})  →  Q, K, V 各自 ${N}×${D}`} />
      </section>

      {/* ── Stage 2: 打分 QKᵀ/√d + Softmax ── */}
      <section className="stage" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '1.2rem' }}>
        <h2 className="sec-h" style={{ marginBottom: '0.4rem' }}>
          第二步：打分 + Softmax — 得到注意力权重
        </h2>
        <p style={{ color: '#444', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
          Query 与所有 Key 做点积（第 04 节）：<code>QKᵀ</code> 的形状 {N}×{D} · {D}×{N} = {N}×{N}（第 10 节的转置逻辑在此体现）。
          除以 √d = √{D} = {Math.sqrt(D).toFixed(0)} 防止梯度消失（第 06 节）。
          再 softmax 每一行 → 权重，每行之和 = 1。
        </p>

        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
          {/* Raw score matrix */}
          <div>
            <Heatmap
              M={scaledScores}
              title={useScaling
                ? `QKᵀ / √${D}  — 缩放后分数 (${N}×${N})`
                : `QKᵀ / 1  — 未缩放分数 (${N}×${N})  ⚠`}
              rowLabels={tokenLabels}
              colLabels={nColLabels}
              highlightRow={selectedQuery}
            />
            <ShapeTag expr={`(${N}×${D}) · (${D}×${N}) = (${N}×${N})`} />
          </div>

          <div style={{ alignSelf: 'center' }}>
            <div style={{ textAlign: 'center', fontSize: 13, color: '#888', marginBottom: 4 }}>softmax</div>
            <span style={{ fontSize: 32, color: '#bbb' }}>→</span>
          </div>

          {/* Attention weight matrix */}
          <div>
            <Heatmap
              M={attnWeights}
              title={`softmax(QKᵀ/√d)  — 注意力权重 (${N}×${N})`}
              rowLabels={tokenLabels}
              colLabels={nColLabels}
              highlightRow={selectedQuery}
            />
            <ShapeTag expr={`每行之和 = 1  ·  当前 Query 行和 = ${rowSum.toFixed(4)} ✓`} />
          </div>
        </div>

        {/* Scaling effect note */}
        {!useScaling && (
          <div style={{
            padding: '10px 14px',
            background: `rgba(${RUST_RGB},0.07)`,
            border: `1.5px solid rgba(${RUST_RGB},0.35)`,
            borderRadius: 6, fontSize: 13, color: '#444',
            maxWidth: 560,
          }}>
            <strong style={{ color: RUST }}>饱和示例：</strong>
            当前所选 Query 行最大权重 = <strong style={{ color: RUST }}>{fmt3(maxW)}</strong>。
            未缩放时分数偏大，softmax 趋向 one-hot（注意力集中在某一个 token 上），
            梯度接近零，训练卡住。开启 √d 缩放，权重分布更均匀，梯度回流更顺畅。
          </div>
        )}

        {/* O(n²) 复杂度 note —— N×N 分数矩阵就是注意力贵的根源 */}
        <div style={{
          padding: '12px 16px',
          background: `rgba(${IKB_RGB},0.05)`,
          borderLeft: `3px solid ${IKB}`,
          borderRadius: 4, fontSize: 13.5, color: '#333',
          maxWidth: 620, lineHeight: 1.6,
        }}>
          <strong style={{ color: IKB }}>为什么注意力「贵」？</strong>
          看这个分数矩阵的形状：<code>QKᵀ</code> 是 <strong>{N}×{N}</strong> ——
          序列里每个 token 都要和<strong>其余每个 token</strong> 算一次点积。
          序列长 n 时，光这一个矩阵就是 <strong>n² 个分数</strong>，
          算力和显存都按 <strong>O(n²)</strong> 增长：上下文翻倍，注意力开销翻<strong>四倍</strong>。
          这就是长上下文为什么烧钱，也是 FlashAttention、稀疏注意力、线性注意力这些工作要解决的问题。
        </div>
      </section>

      {/* ── Stage 3: 注意力箭头 + 输出 ── */}
      <section className="stage" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '1.2rem' }}>
        <h2 className="sec-h" style={{ marginBottom: '0.4rem' }}>
          第三步：加权求和 — 用权重读出 Value
        </h2>
        <p style={{ color: '#444', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
          <strong style={{ color: RUST }}>「{TOKENS[selectedQuery]}」</strong> 这个 token 的输出
          = 它对每个 token 的注意力权重（下图弧线粗细）×
          对应的 Value 向量之和。弧线越粗 = 越「关注」那个 token。
        </p>

        {/* Arrows */}
        <div style={{ width: '100%', maxWidth: 520 }}>
          <AttentionArrows weights={selWeights} selectedQuery={selectedQuery} />
        </div>

        {/* Weight table for selected query */}
        <div style={{ overflowX: 'auto', width: '100%' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 13, minWidth: 420 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${RUST}` }}>
                <th style={{ padding: '4px 12px', color: RUST, textAlign: 'left' }}>Key token</th>
                <th style={{ padding: '4px 12px', color: RUST, textAlign: 'right' }}>attention weight</th>
                <th style={{ padding: '4px 12px', color: RUST, textAlign: 'right' }}>%</th>
                <th style={{ padding: '4px 12px', color: RUST, textAlign: 'right' }}>贡献 (weight × V 行)</th>
              </tr>
            </thead>
            <tbody>
              {TOKENS.map((tok, j) => {
                const w = selWeights[j]
                const isMax = w === maxW
                return (
                  <tr
                    key={j}
                    style={{
                      background: isMax ? `rgba(${RUST_RGB},0.07)` : 'transparent',
                      fontWeight: isMax ? 700 : 400,
                    }}
                  >
                    <td style={{ padding: '4px 12px', color: isMax ? RUST : '#333' }}>{tok}</td>
                    <td style={{ padding: '4px 12px', fontFamily: 'monospace', textAlign: 'right', color: isMax ? RUST : '#333' }}>
                      {fmt3(w)}
                    </td>
                    <td style={{ padding: '4px 12px', textAlign: 'right', color: '#666' }}>
                      {(w * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '4px 12px', fontFamily: 'monospace', textAlign: 'right', fontSize: 11, color: '#555' }}>
                      {V[j].map(v => fmt2(w * v)).join('  ')}
                    </td>
                  </tr>
                )
              })}
            </tbody>
            <tfoot>
              <tr style={{ borderTop: `2px solid ${RUST}` }}>
                <td style={{ padding: '4px 12px', color: RUST, fontWeight: 700 }}>输出 out[{selectedQuery}]</td>
                <td style={{ padding: '4px 12px', textAlign: 'right', fontFamily: 'monospace', color: RUST, fontWeight: 700 }}>
                  Σ = {fmt3(rowSum)}
                </td>
                <td style={{ padding: '4px 12px', textAlign: 'right', color: RUST, fontWeight: 700 }}>100%</td>
                <td style={{ padding: '4px 12px', fontFamily: 'monospace', textAlign: 'right', color: RUST, fontWeight: 700 }}>
                  {selOutput.map(v => fmt2(v)).join('  ')}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>

        {/* Output vector heatmap */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#555', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            out[{selectedQuery}]「{TOKENS[selectedQuery]}」的输出向量 (1×{D})
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            {selOutput.map((v, j) => {
              const absMax = Math.max(...selOutput.map(Math.abs), 0.01)
              const t = (v + absMax) / (2 * absMax)
              const alpha = 0.08 + t * 0.82
              return (
                <div
                  key={j}
                  title={`d${j}: ${fmt3(v)}`}
                  style={{
                    width: 64, height: 52,
                    background: `rgba(${RUST_RGB},${alpha})`,
                    border: `1px solid rgba(${RUST_RGB},0.25)`,
                    borderRadius: 4,
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    fontSize: 10, fontFamily: 'monospace',
                    color: t > 0.55 ? '#fff' : '#333',
                  }}
                >
                  <span style={{ fontSize: 9, opacity: 0.7 }}>d{j}</span>
                  <span style={{ fontWeight: 700 }}>{fmt2(v)}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Full output matrix */}
        <div style={{ marginTop: 8 }}>
          <Heatmap
            M={output}
            title={`output = softmax(QKᵀ/√d) · V  — 输出矩阵 (${N}×${D})`}
            rowLabels={tokenLabels}
            colLabels={dimLabels}
            highlightRow={selectedQuery}
          />
          <ShapeTag expr={`(${N}×${N}) · (${N}×${D}) = (${N}×${D})  ← 和 X 形状相同`} />
        </div>
      </section>

      {/* ── readouts: 一图看全公式形状 ── */}
      <section className="readouts">
        <h2 className="sec-h">形状总账</h2>
        <div style={{
          fontFamily: 'monospace', fontSize: 13, lineHeight: 2.2,
          color: '#333', background: '#f6f8ff',
          border: `1px solid rgba(${IKB_RGB},0.18)`,
          borderRadius: 6, padding: '14px 20px',
          overflowX: 'auto',
        }}>
          {[
            [`X          :  ${N}×${D}`, '   — token embeddings，每行一个 token'],
            [`W_Q, W_K, W_V : ${D}×${D}`, '   — 投影矩阵（可学习）'],
            [`Q = X·W_Q  :  ${N}×${D}`, '   — Query 矩阵'],
            [`K = X·W_K  :  ${N}×${D}`, '   — Key 矩阵'],
            [`V = X·W_V  :  ${N}×${D}`, '   — Value 矩阵'],
            [`Kᵀ         :  ${D}×${N}`, '   — 转置后内维对齐（第 10 节）'],
            [`QKᵀ        :  ${N}×${N}`, `   — 每个 token 对每个 token 的原始得分`],
            [`QKᵀ/√d    :  ${N}×${N}`, `   — 除以 √${D}=${Math.sqrt(D).toFixed(0)} 缩放（第 06 节）`],
            [`softmax(…) :  ${N}×${N}`, '   — 注意力权重，每行 = 概率分布（第 29 节）'],
            [`output     :  ${N}×${D}`, '   — 加权 Value 和，形状等于 X（第 18/22 节）'],
          ].map(([shape, note], idx) => (
            <div key={idx}>
              <span style={{ color: IKB, fontWeight: 700 }}>{shape}</span>
              <span style={{ color: '#888' }}>{note}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── verdict ── */}
      <section className="verdict verdict--eq">
        <p>
          <strong>数学验证：</strong>
          当前所选 Query「{TOKENS[selectedQuery]}」的注意力权重行之和
          = <strong style={{ color: IKB }}>{fmt3(rowSum)}</strong>（应等于 1.000 ✓）。
          输出矩阵形状 <code>{N}×{D}</code>，与输入 X 完全相同——
          self-attention 是「形状不变」的操作，token 数和 embedding 维度都保持不变。
        </p>
        <p>
          切换「不缩放」模式：最大权重
          {' '}<strong style={{ color: useScaling ? IKB : RUST }}>
            {fmt3(maxW)}
          </strong>
          {useScaling
            ? '（正常：权重分散，梯度健康）。'
            : `（偏大：softmax 已趋向 one-hot，梯度接近零）。切回 ÷√${D} 观察差异。`}
        </p>
      </section>

      {/* ── bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            你刚刚亲手算完了 Transformer 的心脏。
            <code>Attention(Q, K, V) = softmax(QKᵀ/√d) · V</code>
            里的每一块都是前面的章节：
            <strong>点积打分</strong>（第 04 节）、
            <strong>√d 缩放</strong>（第 06 节）、
            <strong>QKᵀ 对齐维度</strong>（第 10 节）、
            <strong>softmax 成权重</strong>（第 29 节）、
            <strong>按权重对 V 做加权和</strong>（投影/低秩读出，第 18/22 节）。
          </p>
          <p>
            <strong>看懂这一页 = 看懂注意力。</strong>
            GPT、BERT、LLaMA 的每一层，每一个 attention head，都在重复上面的六行。
            下面第 32 节把它分成<em>多头</em>（把 d 拆成 h 份并行运行），
            第 33 节把它装进完整的 Transformer block（加 残差 + LayerNorm + MLP）。
            地基在这里；一切后续都是在这个公式上加砖。
          </p>
        </div>
      </section>

      {/* ── code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：从零实现 self-attention</h2>
        <CodeBlock code={SNIPPET} language="python" title="self_attention.py" />
      </section>

      {/* ── pager ── */}
      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一章</span>
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
            <span className="pager-dir">下一章 →</span>
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
