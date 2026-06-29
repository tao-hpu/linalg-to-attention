import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── design tokens ──────────────────────────────────────────────────
const CELL = 22                // px per unit dim — boxes scale proportionally
const IKB = '#002fa7'          // Swiss-blue primary
const INNER_CLR = '#c75b39'    // warm accent — inner (consumed) dims
const OUTER_CLR = '#1a7a4a'    // calm green  — outer (surviving) dims + valid verdict
const INVALID_CLR = '#c0392b'  // muted red   — illegal multiply
const RESULT_BG = '#eaf0ff'    // IKB-soft    — result matrix background

// ── python/torch code snippet ──────────────────────────────────────
const SNIPPET = `\
import torch, math

n, d = 4, 3          # n = 序列长度（tokens），d = head dimension

Q = torch.randn(n, d)          # shape: (4, 3)
K = torch.randn(n, d)          # shape: (4, 3)
V = torch.randn(n, d)          # shape: (4, 3)

# Q @ K → RuntimeError! 内维 d=3 ≠ n=4（K 的行数）

K_T = K.transpose(-2, -1)      # (n, d) → (d, n)，即 shape: (3, 4)

# 现在内维对齐：Q 列数 d=3 == K_T 行数 d=3 ✓
scores = Q @ K_T               # shape: (4, 4) — n×n 打分矩阵
scores = scores / math.sqrt(d) # 缩放，防止梯度爆炸
weights = torch.softmax(scores, dim=-1)  # (4, 4)，每行和为 1

out = weights @ V              # (4, 4) @ (4, 3) = (4, 3) — 回到 n×d
# out[i] = 用 token_i 的注意力权重加权 V 的各行`

// ── helper components (defined before use for clarity) ─────────────

function Stepper({
  label, value, onChange, min = 1, max = 6,
}: {
  label: string
  value: number
  onChange: (n: number) => void
  min?: number
  max?: number
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
      <span style={{ fontSize: 13, color: '#555', minWidth: 34 }}>{label}</span>
      <button
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
        style={{
          width: 26, height: 26,
          cursor: value <= min ? 'default' : 'pointer',
          border: '1px solid #ccc', borderRadius: 4,
          background: '#f5f5f5', fontSize: 15, padding: 0,
        }}
      >−</button>
      <span style={{ minWidth: 18, textAlign: 'center', fontWeight: 700, fontSize: 15 }}>
        {value}
      </span>
      <button
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
        style={{
          width: 26, height: 26,
          cursor: value >= max ? 'default' : 'pointer',
          border: '1px solid #ccc', borderRadius: 4,
          background: '#f5f5f5', fontSize: 15, padding: 0,
        }}
      >+</button>
    </div>
  )
}

// Shape label "name rows×cols" — inner dim in INNER_CLR, outer in OUTER_CLR
function ShapeLabel({
  name, rows, cols, innerDim,
}: {
  name: string
  rows: number
  cols: number
  innerDim?: 'rows' | 'cols'
}) {
  return (
    <div style={{ textAlign: 'center' }}>
      <span style={{ fontWeight: 700, color: '#333', marginRight: 3, fontSize: 14 }}>{name}</span>
      <span style={{ fontFamily: 'monospace', fontSize: 13 }}>
        <span style={{ color: innerDim === 'rows' ? INNER_CLR : OUTER_CLR, fontWeight: 700 }}>
          {rows}
        </span>
        <span style={{ color: '#bbb' }}>×</span>
        <span style={{ color: innerDim === 'cols' ? INNER_CLR : OUTER_CLR, fontWeight: 700 }}>
          {cols}
        </span>
      </span>
    </div>
  )
}

// Proportional rectangle box: rows→height, cols→width, each unit = CELL px
function MatBox({
  rows, cols, bg = '#f0f4fa', borderColor = '#b0b8c8',
}: {
  rows: number
  cols: number
  bg?: string
  borderColor?: string
}) {
  return (
    <div
      style={{
        width: Math.max(cols * CELL, CELL),
        height: Math.max(rows * CELL, CELL),
        background: bg,
        border: `2px solid ${borderColor}`,
        borderRadius: 4,
      }}
    />
  )
}

// Small annotation below a box showing one dimension's role
function DimAnnotation({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ fontSize: 11, color, textAlign: 'center', lineHeight: 1.4, marginTop: 2 }}>
      <span style={{ fontWeight: 700, fontFamily: 'monospace' }}>{value}</span>
      <span style={{ color: '#bbb' }}> — </span>
      <span>{label}</span>
    </div>
  )
}

// Compact chip: shape label + proportional box, used in QKᵀ example
function MatrixChip({
  name, rows, cols, innerDim,
}: {
  name: string
  rows: number
  cols: number
  innerDim?: 'rows' | 'cols'
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
      <ShapeLabel name={name} rows={rows} cols={cols} innerDim={innerDim} />
      <MatBox rows={rows} cols={cols} />
    </div>
  )
}

// n×n attention score grid — diagonal highlighted (token with itself)
function ScoreGrid({ n }: { n: number }) {
  const toks = Array.from({ length: n }, (_, i) => i + 1)
  return (
    <div style={{ overflowX: 'auto', marginTop: 8 }}>
      <table style={{ borderCollapse: 'collapse', margin: '0 auto', fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ padding: '4px 8px', color: '#aaa', fontWeight: 400, fontSize: 11 }} />
            {toks.map((j) => (
              <th
                key={j}
                style={{
                  padding: '4px 10px',
                  color: INNER_CLR,
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  fontSize: 12,
                  borderBottom: `2px solid ${INNER_CLR}`,
                }}
              >
                key_{j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {toks.map((i) => (
            <tr key={i}>
              <td
                style={{
                  padding: '6px 10px',
                  color: IKB,
                  fontFamily: 'monospace',
                  fontWeight: 700,
                  fontSize: 12,
                  borderRight: `2px solid ${IKB}`,
                }}
              >
                q_{i}
              </td>
              {toks.map((j) => (
                <td
                  key={j}
                  style={{
                    padding: '7px 12px',
                    background: i === j ? '#d6e4ff' : RESULT_BG,
                    border: '1px solid #c5d5f5',
                    textAlign: 'center',
                    fontFamily: 'monospace',
                    fontSize: 11,
                    color: '#444',
                  }}
                >
                  q{i}·k{j}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── interactive shape playground ───────────────────────────────────

function ShapePlayground() {
  const [aRows, setARows] = useState(4)
  const [aCols, setACols] = useState(3)
  const [bRows, setBRows] = useState(5)
  const [bCols, setBCols] = useState(2)
  const [bTransposed, setBTransposed] = useState(false)

  // effective B shape after optional transpose
  const bEffRows = bTransposed ? bCols : bRows
  const bEffCols = bTransposed ? bRows : bCols

  const legal = aCols === bEffRows

  return (
    <>
      {/* Stepper controls */}
      <div className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">矩阵 A</span>
          </div>
          <Stepper label="行 m" value={aRows} onChange={setARows} />
          <Stepper label="列 k" value={aCols} onChange={setACols} />
        </div>
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">矩阵 B</span>
            <button
              onClick={() => setBTransposed((v) => !v)}
              style={{
                marginLeft: 10,
                padding: '3px 12px',
                fontSize: 13,
                cursor: 'pointer',
                border: `2px solid ${bTransposed ? IKB : '#ccc'}`,
                borderRadius: 5,
                background: bTransposed ? IKB : '#f5f5f5',
                color: bTransposed ? '#fff' : '#333',
                fontWeight: 600,
              }}
            >
              {bTransposed ? 'Bᵀ 已转置' : '转置 Bᵀ'}
            </button>
          </div>
          <Stepper label="行 k" value={bRows} onChange={setBRows} />
          <Stepper label="列 n" value={bCols} onChange={setBCols} />
          {bTransposed && (
            <p style={{ margin: '6px 0 0', fontSize: 12, color: INNER_CLR, fontWeight: 600 }}>
              转置后形状：{bCols}×{bRows}（行列互换）
            </p>
          )}
        </div>
      </div>

      {/* Visual: A · B_eff = result */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1.2rem',
          flexWrap: 'wrap',
          margin: '1.8rem 0',
          justifyContent: 'center',
        }}
      >
        {/* A */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
          <ShapeLabel name="A" rows={aRows} cols={aCols} innerDim="cols" />
          <MatBox rows={aRows} cols={aCols} borderColor={legal ? OUTER_CLR : '#b0b8c8'} />
          <DimAnnotation label="外维" value={aRows} color={OUTER_CLR} />
          <DimAnnotation label="内维" value={aCols} color={INNER_CLR} />
        </div>

        <span style={{ fontSize: 30, color: '#bbb', userSelect: 'none' }}>·</span>

        {/* B or Bᵀ */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
          <ShapeLabel
            name={bTransposed ? 'Bᵀ' : 'B'}
            rows={bEffRows}
            cols={bEffCols}
            innerDim="rows"
          />
          <MatBox rows={bEffRows} cols={bEffCols} borderColor={legal ? OUTER_CLR : '#b0b8c8'} />
          <DimAnnotation label="内维" value={bEffRows} color={INNER_CLR} />
          <DimAnnotation label="外维" value={bEffCols} color={OUTER_CLR} />
        </div>

        <span style={{ fontSize: 24, color: '#bbb', userSelect: 'none' }}>＝</span>

        {/* Result or ✗ */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
          {legal ? (
            <>
              <ShapeLabel name="结果" rows={aRows} cols={bEffCols} />
              <MatBox rows={aRows} cols={bEffCols} bg={RESULT_BG} borderColor={IKB} />
            </>
          ) : (
            <div
              style={{
                width: 72,
                height: 72,
                border: `2px dashed ${INVALID_CLR}`,
                borderRadius: 6,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 28,
                color: INVALID_CLR,
              }}
            >
              ✗
            </div>
          )}
        </div>
      </div>

      {/* Verdict */}
      <div className={`verdict ${legal ? 'verdict--eq' : 'verdict--neq'}`}>
        {legal ? (
          <p>
            <strong style={{ color: OUTER_CLR }}>合法 ✓</strong>　内维{' '}
            <strong style={{ color: INNER_CLR, fontFamily: 'monospace' }}>
              {aCols} = {bEffRows}
            </strong>
            ，两侧内维吻合，乘法成立。外维{' '}
            <strong style={{ color: OUTER_CLR }}>{aRows}</strong> 和{' '}
            <strong style={{ color: OUTER_CLR }}>{bEffCols}</strong> 流入结果——结果形状为{' '}
            <code>
              {aRows}×{bEffCols}
            </code>
            。内维消失，外维存活。
          </p>
        ) : (
          <p>
            <strong style={{ color: INVALID_CLR }}>非法 ✗</strong>　内维{' '}
            <strong style={{ color: INNER_CLR, fontFamily: 'monospace' }}>
              {aCols} ≠ {bEffRows}
            </strong>
            ，A 的列数对不上 {bTransposed ? 'Bᵀ' : 'B'} 的行数，乘法无法进行。
            {!bTransposed && <> 试试点「转置 Bᵀ」，看看能不能让内维对齐。</>}
          </p>
        )}
      </div>
    </>
  )
}

// ── QKᵀ attention worked example ──────────────────────────────────

function QKAttentionExample() {
  const n: number = 4  // token count
  const d: number = 3  // head dimension

  // Q·K: inner = Q.cols(d=3) vs K.rows(n=4) — illegal when d ≠ n
  const qkDirect = d === n  // false: 3 ≠ 4 → used in conditional below

  return (
    <section style={{ marginTop: '2.5rem' }}>
      <h2 className="sec-h">实战：为什么是 QKᵀ 不是 QK</h2>
      <p>
        取一个 <strong>n = {n}</strong> 个 token 的序列，每个 token 的 query / key 向量维度{' '}
        <strong>d = {d}</strong>。Q 和 K 都是 <code>{n}×{d}</code>。
      </p>

      {/* Attempt 1: Q · K → illegal */}
      <div style={{ margin: '1.4rem 0 0.6rem', fontWeight: 700, color: '#333' }}>
        尝试 1：直接 Q · K
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
          marginBottom: '0.8rem',
        }}
      >
        <MatrixChip name="Q" rows={n} cols={d} innerDim="cols" />
        <span style={{ fontSize: 26, color: '#bbb' }}>·</span>
        <MatrixChip name="K" rows={n} cols={d} innerDim="rows" />
        <span style={{ fontSize: 20, color: '#bbb' }}>=</span>
        <div
          style={{
            padding: '8px 14px',
            background: '#fdf0ed',
            border: `2px solid ${INVALID_CLR}`,
            borderRadius: 6,
            color: INVALID_CLR,
            fontWeight: 700,
            fontSize: 14,
          }}
        >
          非法 — 内维{' '}
          <span style={{ fontFamily: 'monospace' }}>
            {d} ≠ {n}
          </span>
        </div>
      </div>
      {!qkDirect && (
        <p style={{ fontSize: 14, color: '#555', marginBottom: '1.2rem' }}>
          Q 的列数是 d = {d}，K 的行数是 n = {n}，<strong>{d} ≠ {n}</strong>，直接相乘会报
          shape 错误。K 是「每个 token 的 key」，它的行方向是 token，和 Q 的列方向（feature
          维）对不上。解决办法：<strong>转置 K</strong>，把形状从 {n}×{d} 翻转成 {d}×{n}。
        </p>
      )}

      {/* Attempt 2: Q · Kᵀ → legal */}
      <div style={{ margin: '0.6rem 0', fontWeight: 700, color: '#333' }}>
        尝试 2：转置 K → Q · Kᵀ
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          flexWrap: 'wrap',
          marginBottom: '1rem',
        }}
      >
        <MatrixChip name="Q" rows={n} cols={d} innerDim="cols" />
        <span style={{ fontSize: 26, color: '#bbb' }}>·</span>
        <MatrixChip name="Kᵀ" rows={d} cols={n} innerDim="rows" />
        <span style={{ fontSize: 20, color: '#bbb' }}>=</span>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: OUTER_CLR }}>
            内维 d={d} = d={d} ✓
          </span>
          <MatBox rows={n} cols={n} bg={RESULT_BG} borderColor={IKB} />
          <span style={{ fontFamily: 'monospace', fontSize: 12, color: IKB, fontWeight: 700 }}>
            scores: {n}×{n}
          </span>
        </div>
      </div>

      <p style={{ fontSize: 14, color: '#444', marginBottom: '0.8rem' }}>
        Kᵀ 的形状是 <code>{d}×{n}</code>，内维恰好都是 d = {d}，两侧对齐。结果是{' '}
        <strong>
          {n}×{n}
        </strong>{' '}
        的矩阵：<strong>每个 token 对每个 token</strong> 的相似度分数。
        下面是那个 {n}×{n} 的分数矩阵——对角线（token 跟自己）通常分最高：
      </p>

      <ScoreGrid n={n} />

      <p style={{ fontSize: 13, color: '#777', marginTop: '0.8rem', lineHeight: 1.7 }}>
        格子 (i, j) = Q 的第 i 行（query<sub>i</sub>）与 Kᵀ 的第 j 列（key<sub>j</sub>
        ）的内积。经 softmax 后变成权重，再用{' '}
        <code>
          weights @ V
        </code>{' '}
        （{n}×{n} 乘 {n}×{d}）把 V 加权求和，得回{' '}
        <code>{n}×{d}</code>——形状和 Q 一模一样。
      </p>
    </section>
  )
}

// ── main exported page ─────────────────────────────────────────────

export function TransposeShape() {
  const me = findChapter('transpose-shape')!
  const { prev, next } = neighbors('transpose-shape')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第二部分 · 矩阵：一个动作
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          转置与形状
          <span className="zh-sub">QKᵀ 的那个 ᵀ 为什么必须有？</span>
        </h1>
        <p className="lede">
          矩阵乘法 (m×k)·(k×n) = (m×n) 有一条铁律：
          <strong>内维必须相等</strong>。Transpose（转置）把形状 n×d 翻转成 d×n——整个{' '}
          <code>QKᵀ</code> 里那个 ᵀ，不是装饰，是为了让内维 d 对齐、让乘法合法、
          让结果恰好是 n×n 的「每个 token 对每个 token 打分」矩阵。
          弄懂了维度对账，任何 attention 实现你都能一眼读懂。
        </p>
      </header>

      {/* Section: shape playground */}
      <section>
        <h2 className="sec-h">形状游乐场：拨动维度，看乘法能不能成立</h2>
        <p style={{ marginBottom: '1rem', color: '#444' }}>
          用 +/− 调整 A 和 B 的行列数，观察内维是否对齐。再点「转置 Bᵀ」，
          看转置如何把不合法变合法。颜色规则：
          <span style={{ color: INNER_CLR, fontWeight: 700 }}>
            橙色 = 内维（必须相等、相乘后消失）
          </span>
          ；
          <span style={{ color: OUTER_CLR, fontWeight: 700 }}>
            绿色 = 外维（流入结果）
          </span>
          。
        </p>
        <ShapePlayground />
      </section>

      {/* Section: QKᵀ worked example */}
      <QKAttentionExample />

      {/* Bridge */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            <code>attention_scores = Q @ K.transpose(-2, -1)</code> 里的{' '}
            <code>.transpose</code> 不是数学装饰——它是维度对账的关键。Q 是 (n, d)，K 原本也是
            (n, d)，转置后 Kᵀ 变成 (d, n)，内维 d 对齐，相乘得 (n, n)：
            每个 token 对每个 token 的相似度打分矩阵。
          </p>
          <p>
            再往后：softmax 作用在 n×n 的分数上（按行归一化），得到注意力权重；
            再乘 V（n×d）——(n, n)·(n, d) = (n, d)——输出形状和输入一模一样。
            <strong>维度对账是读懂任何 attention 实现的基本功。</strong>
          </p>
        </div>
      </section>

      {/* Code */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：形状怎么跟着走</h2>
        <CodeBlock code={SNIPPET} language="python" title="attention_shapes.py" />
      </section>

      {/* Pager */}
      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一章</span>
            <span className="pager-title">
              {prev.num} {prev.title}
            </span>
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
              {next.num} {next.title}
              {next.status !== 'live' && ' · 规划中'}
            </span>
          </Link>
        ) : (
          <span />
        )}
      </nav>

      <p className="page-foot">
        共 {allChapters.length} 节 · 你在第 {me.num} 节
      </p>
    </article>
  )
}
