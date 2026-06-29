import { useState } from 'react'
import { Link } from 'react-router-dom'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// Fixed concrete example: A (2×2) · B (2×2) = C (2×2)
// A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = [[19,22],[43,50]]

const IKB = '#002fa7'
const WARM = '#c75b39'
const SOFT = '#eaf0ff'
const WARM_SOFT = '#fdf0eb'
const NEUTRAL_BORDER = '#d0d5dd'

type View = 'dot' | 'col' | 'row' | 'outer'

const VIEW_LABELS: { id: View; label: string }[] = [
  { id: 'dot', label: '① 点积视角' },
  { id: 'col', label: '② 列的线性组合' },
  { id: 'row', label: '③ 行的线性组合' },
  { id: 'outer', label: '④ 秩-1 外积之和' },
]

const A: number[][] = [[1, 2], [3, 4]]
const B: number[][] = [[5, 6], [7, 8]]
const C: number[][] = [[19, 22], [43, 50]]

const DOT_TERMS: string[][] = [
  ['1·5 + 2·7 = 19', '1·6 + 2·8 = 22'],
  ['3·5 + 4·7 = 43', '3·6 + 4·8 = 50'],
]

const OUTER_TERMS: { mat: number[][]; label: string }[] = [
  { mat: [[5, 6], [15, 18]], label: 'col₀(A) ⊗ row₀(B)' },
  { mat: [[14, 16], [28, 32]], label: 'col₁(A) ⊗ row₁(B)' },
]

function SmallMatrix({
  data,
  name,
  rowHighlight,
  colHighlight,
  colorScheme = 'blue',
}: {
  data: number[][]
  name?: string
  rowHighlight?: number
  colHighlight?: number
  colorScheme?: 'blue' | 'warm' | 'neutral'
}) {
  const primary = colorScheme === 'warm' ? WARM : colorScheme === 'neutral' ? '#555' : IKB

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'monospace' }}>
      {name && (
        <span style={{ fontWeight: 700, color: primary, marginRight: 2, fontSize: 15 }}>{name}</span>
      )}
      <span style={{ fontSize: 22, color: '#888', lineHeight: 1 }}>[</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {data.map((row, i) => (
          <div key={i} style={{ display: 'flex', gap: 4 }}>
            {row.map((val, j) => {
              const isRowHl = rowHighlight === i
              const isColHl = colHighlight === j
              let bg = 'transparent'
              let color = '#1b1f24'
              let fontWeight: number = 400
              if (isRowHl) { bg = SOFT; color = IKB; fontWeight = 600 }
              else if (isColHl) { bg = WARM_SOFT; color = WARM; fontWeight = 600 }
              return (
                <span
                  key={j}
                  style={{
                    display: 'inline-block',
                    width: 28,
                    textAlign: 'center',
                    borderRadius: 3,
                    padding: '2px 0',
                    background: bg,
                    color,
                    fontWeight,
                    fontSize: 14,
                    transition: 'background 0.15s, color 0.15s',
                  }}
                >
                  {val}
                </span>
              )
            })}
          </div>
        ))}
      </div>
      <span style={{ fontSize: 22, color: '#888', lineHeight: 1 }}>]</span>
    </div>
  )
}

function DotView() {
  const [active, setActive] = useState<[number, number] | null>(null)

  return (
    <div>
      <p style={{ marginBottom: 16, color: '#444', lineHeight: 1.7 }}>
        点击 C 中任意一个格子，可以看到它由 A 的哪一行与 B 的哪一列做内积得到。
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
        <SmallMatrix data={A} name="A" rowHighlight={active !== null ? active[0] : undefined} />
        <span style={{ fontSize: 20, color: '#888' }}>·</span>
        <SmallMatrix
          data={B}
          name="B"
          colHighlight={active !== null ? active[1] : undefined}
          colorScheme="warm"
        />
        <span style={{ fontSize: 20, color: '#888' }}>=</span>

        {/* C matrix — clickable cells */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'monospace' }}>
          <span style={{ fontWeight: 700, color: '#555', marginRight: 2, fontSize: 15 }}>C</span>
          <span style={{ fontSize: 22, color: '#888', lineHeight: 1 }}>[</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {C.map((row, i) => (
              <div key={i} style={{ display: 'flex', gap: 4 }}>
                {row.map((val, j) => {
                  const isActive = active !== null && active[0] === i && active[1] === j
                  return (
                    <span
                      key={j}
                      onClick={() => setActive(isActive ? null : [i, j])}
                      style={{
                        display: 'inline-block',
                        width: 28,
                        textAlign: 'center',
                        borderRadius: 3,
                        padding: '2px 0',
                        background: isActive ? SOFT : 'transparent',
                        color: isActive ? IKB : '#1b1f24',
                        fontWeight: isActive ? 700 : 400,
                        fontSize: 14,
                        cursor: 'pointer',
                        border: `1.5px solid ${isActive ? IKB : NEUTRAL_BORDER}`,
                        transition: 'all 0.15s',
                      }}
                    >
                      {val}
                    </span>
                  )
                })}
              </div>
            ))}
          </div>
          <span style={{ fontSize: 22, color: '#888', lineHeight: 1 }}>]</span>
        </div>
      </div>

      {active !== null ? (
        <div style={{
          marginTop: 16,
          padding: '10px 16px',
          background: SOFT,
          borderLeft: `3px solid ${IKB}`,
          borderRadius: 4,
          fontFamily: 'monospace',
          fontSize: 14,
          color: IKB,
        }}>
          C[{active[0]}][{active[1]}] = {DOT_TERMS[active[0]][active[1]]}
        </div>
      ) : (
        <p style={{ marginTop: 12, color: '#999', fontSize: 13 }}>← 点击 C 中的格子查看内积展开</p>
      )}
    </div>
  )
}

function ColView() {
  // col j of C = A · col j of B = B[0][j]·col0(A) + B[1][j]·col1(A)
  return (
    <div>
      <p style={{ marginBottom: 16, color: '#444', lineHeight: 1.7 }}>
        C 的第 j 列 = A 作用在 B 的第 j 列——即 A 的各列按 B 的列分量加权求和。
      </p>
      {[0, 1].map((j) => {
        const bCol = B.map((r) => r[j])
        const cCol = C.map((r) => r[j])
        return (
          <div key={j} style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ color: '#555', fontFamily: 'monospace', fontSize: 13 }}>col {j} of C:</span>
            {[0, 1].map((k) => (
              <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'monospace' }}>
                {k > 0 && <span style={{ color: '#888' }}>+</span>}
                <span style={{ fontWeight: 700, color: WARM }}>{bCol[k]}</span>
                <span style={{ color: '#888' }}>·</span>
                <span style={{ fontSize: 18, color: '#888' }}>[</span>
                <span style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {[0, 1].map((i) => (
                    <span key={i} style={{
                      display: 'inline-block', width: 20, textAlign: 'center',
                      background: SOFT, color: IKB, fontWeight: 600,
                      borderRadius: 2, padding: '1px 0', fontSize: 13,
                    }}>
                      {A[i][k]}
                    </span>
                  ))}
                </span>
                <span style={{ fontSize: 18, color: '#888' }}>]</span>
              </span>
            ))}
            <span style={{ color: '#888' }}>=</span>
            <span style={{ fontSize: 18, color: '#888' }}>[</span>
            <span style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {cCol.map((v, i) => (
                <span key={i} style={{
                  display: 'inline-block', width: 24, textAlign: 'center',
                  background: '#f3f3f3', fontWeight: 600, borderRadius: 2, padding: '1px 0', fontSize: 13,
                }}>
                  {v}
                </span>
              ))}
            </span>
            <span style={{ fontSize: 18, color: '#888' }}>]</span>
          </div>
        )
      })}
    </div>
  )
}

function RowView() {
  // row i of C = row i of A · B = A[i][0]·row0(B) + A[i][1]·row1(B)
  return (
    <div>
      <p style={{ marginBottom: 16, color: '#444', lineHeight: 1.7 }}>
        C 的第 i 行 = A 的第 i 行作用在 B 上——即 B 的各行按 A 的行分量加权求和。
      </p>
      {[0, 1].map((i) => {
        const aRow = A[i]
        const cRow = C[i]
        return (
          <div key={i} style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ color: '#555', fontFamily: 'monospace', fontSize: 13 }}>row {i} of C:</span>
            {[0, 1].map((k) => (
              <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'monospace' }}>
                {k > 0 && <span style={{ color: '#888' }}>+</span>}
                <span style={{ fontWeight: 700, color: IKB }}>{aRow[k]}</span>
                <span style={{ color: '#888' }}>·</span>
                <span style={{ fontSize: 18, color: '#888' }}>[</span>
                <span style={{ display: 'flex', gap: 3 }}>
                  {[0, 1].map((j) => (
                    <span key={j} style={{
                      display: 'inline-block', width: 20, textAlign: 'center',
                      background: WARM_SOFT, color: WARM, fontWeight: 600,
                      borderRadius: 2, padding: '1px 0', fontSize: 13,
                    }}>
                      {B[k][j]}
                    </span>
                  ))}
                </span>
                <span style={{ fontSize: 18, color: '#888' }}>]</span>
              </span>
            ))}
            <span style={{ color: '#888' }}>=</span>
            <span style={{ fontSize: 18, color: '#888' }}>[</span>
            <span style={{ display: 'flex', gap: 3 }}>
              {cRow.map((v, j) => (
                <span key={j} style={{
                  display: 'inline-block', width: 24, textAlign: 'center',
                  background: '#f3f3f3', fontWeight: 600, borderRadius: 2, padding: '1px 0', fontSize: 13,
                }}>
                  {v}
                </span>
              ))}
            </span>
            <span style={{ fontSize: 18, color: '#888' }}>]</span>
          </div>
        )
      })}
    </div>
  )
}

function OuterView() {
  return (
    <div>
      <p style={{ marginBottom: 16, color: '#444', lineHeight: 1.7 }}>
        C = A·B 等于 A 的各列与 B 对应各行的<strong>外积</strong>（rank-1 矩阵）之和。
        每一项 col<sub>k</sub>(A) ⊗ row<sub>k</sub>(B) 的秩只有 1——
        这正是注意力对 value 向量加权求和、以及 LoRA 低秩分解的几何种子。
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        {OUTER_TERMS.map((term, k) => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {k > 0 && <span style={{ fontSize: 20, color: '#888' }}>+</span>}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{term.label}</div>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'monospace' }}>
                <span style={{ fontSize: 18, color: '#888' }}>[</span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {term.mat.map((row, i) => (
                    <div key={i} style={{ display: 'flex', gap: 4 }}>
                      {row.map((val, j) => (
                        <span key={j} style={{
                          display: 'inline-block', width: 28, textAlign: 'center',
                          background: k === 0 ? SOFT : WARM_SOFT,
                          color: k === 0 ? IKB : WARM,
                          fontWeight: 500, borderRadius: 3, padding: '2px 0', fontSize: 14,
                        }}>
                          {val}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
                <span style={{ fontSize: 18, color: '#888' }}>]</span>
              </div>
            </div>
          </div>
        ))}
        <span style={{ fontSize: 20, color: '#888' }}>=</span>
        <SmallMatrix data={C} name="C" colorScheme="neutral" />
      </div>
    </div>
  )
}

const SNIPPET = `# 四种视角，同一个乘法 (A: 2×2, B: 2×2, C = A·B)
import numpy as np
A = np.array([[1,2],[3,4]]); B = np.array([[5,6],[7,8]])

# ① 点积视角：C[i,j] = row_i(A) · col_j(B)
C1 = np.array([[A[i] @ B[:,j] for j in range(2)] for i in range(2)])

# ② 列的线性组合：col_j(C) = A @ col_j(B)
C2 = np.column_stack([A @ B[:,j] for j in range(2)])

# ③ 行的线性组合：row_i(C) = row_i(A) @ B
C3 = np.vstack([A[i] @ B for i in range(2)])

# ④ 秩-1 外积之和：C = Σ_k outer(col_k(A), row_k(B))
C4 = sum(np.outer(A[:,k], B[k]) for k in range(2))

# 四种写法结果完全相同
assert np.allclose(C1, C2) and np.allclose(C2, C3) and np.allclose(C3, C4)

# attention 中的 softmax(QKᵀ)V 本质是视角④：
# 对 value 行向量的加权求和，每个位置贡献一个 rank-1 项。
# LoRA 把 ΔW 写成 BA（两个低秩矩阵之积）也是同一个思路。`

export function MatmulViews() {
  const [view, setView] = useState<View>('dot')

  const me = findChapter('matmul-views')!
  const { prev, next } = neighbors('matmul-views')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb"><Link to="/">大纲</Link> · 第二部分 · 矩阵：一个动作</div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>矩阵乘法的四种视角<span className="zh-sub">同一个乘法，四种看法</span></h1>
        <p className="lede">
          C = A·B 只有一个答案，却可以用<strong>四种完全不同的眼光</strong>去读它：
          逐格做内积、列的线性组合、行的线性组合、秩-1 outer product 之和。
          读懂这四种视角，就能在任何论文或代码库里认出同一件事——
          不管它写成 <code>einsum</code>、<code>torch.matmul</code> 还是手写嵌套循环。
        </p>
      </header>

      {/* 切换视角 tab row */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {VIEW_LABELS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setView(id)}
            style={{
              padding: '6px 14px',
              borderRadius: 6,
              border: `1.5px solid ${view === id ? IKB : NEUTRAL_BORDER}`,
              background: view === id ? IKB : 'white',
              color: view === id ? 'white' : '#444',
              fontWeight: view === id ? 700 : 400,
              cursor: 'pointer',
              fontSize: 13,
              transition: 'all 0.15s',
            }}
          >
            {label}
          </button>
        ))}
      </div>

      {/* View content area */}
      <div style={{
        padding: '20px 24px',
        border: `1.5px solid ${NEUTRAL_BORDER}`,
        borderRadius: 8,
        background: '#fafbfd',
        marginBottom: 32,
        minHeight: 200,
      }}>
        {view === 'dot' && (
          <>
            <h3 style={{ margin: '0 0 12px', color: IKB, fontSize: 15, fontWeight: 700 }}>
              ① 点积视角 — C[i][j] = row i(A) · col j(B)
            </h3>
            <DotView />
          </>
        )}
        {view === 'col' && (
          <>
            <h3 style={{ margin: '0 0 12px', color: IKB, fontSize: 15, fontWeight: 700 }}>
              ② 列的线性组合 — col j(C) = A · col j(B)
            </h3>
            <ColView />
          </>
        )}
        {view === 'row' && (
          <>
            <h3 style={{ margin: '0 0 12px', color: IKB, fontSize: 15, fontWeight: 700 }}>
              ③ 行的线性组合 — row i(C) = row i(A) · B
            </h3>
            <RowView />
          </>
        )}
        {view === 'outer' && (
          <>
            <h3 style={{ margin: '0 0 12px', color: IKB, fontSize: 15, fontWeight: 700 }}>
              ④ 秩-1 外积之和 — C = Σ<sub>k</sub> col<sub>k</sub>(A) ⊗ row<sub>k</sub>(B)
            </h3>
            <OuterView />
          </>
        )}
      </div>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            attention 的写法千变万化——<code>QK<sup>T</sup></code> 再对 V 做加权和、
            <code>einsum('bhqd,bhkd-&gt;bhqk', Q, K)</code>、<code>torch.matmul</code>、
            手写 for 循环——本质都是这同一个乘法的不同视角。能把任意写法在四种视角间自由切换，
            就不会再被论文里换了种写法的同一公式绕晕。
          </p>
          <p>
            尤其是<strong>视角④（秩-1 outer product 之和）</strong>：
            注意力对 value 向量的加权求和，每个 key 位置贡献一个 rank-1 项；
            LoRA 把权重增量写成两个低秩矩阵之积 <code>ΔW = BA</code>，
            正是在说"只需要几个 rank-1 项就能近似一个完整变换"。这一页是后面 LoRA 章节的直接铺垫。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：四种写法，同一结果</h2>
        <CodeBlock code={SNIPPET} language="python" title="matmul_views.py" />
      </section>

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
