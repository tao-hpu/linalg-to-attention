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

// 两个 rank-1 外积项：col_k(A) ⊗ row_k(B)
const OUTER_TERMS: { mat: number[][]; aCol: number[]; bRow: number[] }[] = [
  { mat: [[5, 6], [15, 18]], aCol: [A[0][0], A[1][0]], bRow: B[0] },   // [1,3] ⊗ [5,6]
  { mat: [[14, 16], [28, 32]], aCol: [A[0][1], A[1][1]], bRow: B[1] }, // [2,4] ⊗ [7,8]
]

type Tone = 'plain' | 'blue' | 'warm' | 'active' | 'activeWarm'

function toneStyle(t: Tone): { background: string; color: string; fontWeight: number; border: string } {
  switch (t) {
    case 'blue': return { background: SOFT, color: IKB, fontWeight: 600, border: '1.5px solid transparent' }
    case 'warm': return { background: WARM_SOFT, color: WARM, fontWeight: 600, border: '1.5px solid transparent' }
    case 'active': return { background: SOFT, color: IKB, fontWeight: 700, border: `1.5px solid ${IKB}` }
    case 'activeWarm': return { background: WARM_SOFT, color: WARM, fontWeight: 700, border: `1.5px solid ${WARM}` }
    default: return { background: 'transparent', color: '#1b1f24', fontWeight: 400, border: '1.5px solid transparent' }
  }
}

const OP = (s: string) => (
  <span style={{ fontSize: 22, color: '#9aa0a6', fontFamily: 'var(--mono)' }}>{s}</span>
)

// 通用 2×2 矩阵：每格 tone 由调用方决定，可选 onCell 让格子可点。
function Matrix({ data, name, nameColor = '#555', tone, onCell }: {
  data: number[][]
  name: string
  nameColor?: string
  tone: (i: number, j: number) => Tone
  onCell?: (i: number, j: number) => void
}) {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontFamily: 'var(--mono)' }}>
      <span style={{ fontWeight: 700, color: nameColor, marginRight: 2, fontSize: 15 }}>{name}</span>
      <span style={{ fontSize: 30, color: NEUTRAL_BORDER, fontWeight: 300, lineHeight: 1 }}>[</span>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {data.map((row, i) => (
          <div key={i} style={{ display: 'flex', gap: 4 }}>
            {row.map((val, j) => {
              const ts = toneStyle(tone(i, j))
              return (
                <span
                  key={j}
                  onClick={onCell ? () => onCell(i, j) : undefined}
                  style={{
                    display: 'inline-block', width: 30, textAlign: 'center',
                    borderRadius: 3, padding: '3px 0', fontSize: 15,
                    cursor: onCell ? 'pointer' : 'default',
                    transition: 'background .15s, color .15s, border-color .15s',
                    ...ts,
                  }}
                >
                  {val}
                </span>
              )
            })}
          </div>
        ))}
      </div>
      <span style={{ fontSize: 30, color: NEUTRAL_BORDER, fontWeight: 300, lineHeight: 1 }}>]</span>
    </div>
  )
}

// 点积视角：点 C 的格子 → 高亮 A 的那一行 · B 的那一列。
function DotView() {
  const [active, setActive] = useState<[number, number] | null>(null)

  return (
    <>
      <h3 className="sec-h" style={{ color: IKB }}>① 点积视角 · C[i][j] = row i(A) · col j(B)</h3>
      <p style={{ marginTop: 0 }}>
        点击 C 中任意一个格子，看它由 <strong>A 的哪一行</strong>与 <strong>B 的哪一列</strong>做内积得到。
      </p>

      <section className="stage">
        <Matrix data={A} name="A" nameColor={IKB}
          tone={(i) => (active && i === active[0] ? 'blue' : 'plain')} />
        {OP('·')}
        <Matrix data={B} name="B" nameColor={WARM}
          tone={(_i, j) => (active && j === active[1] ? 'warm' : 'plain')} />
        {OP('=')}
        <Matrix data={C} name="C"
          tone={(i, j) => (active && i === active[0] && j === active[1] ? 'active' : 'plain')}
          onCell={(i, j) => setActive(active && active[0] === i && active[1] === j ? null : [i, j])} />
      </section>

      <div className="note">
        {active ? (
          <p style={{ fontFamily: 'var(--mono)' }}>
            <strong style={{ color: IKB }}>C[{active[0]}][{active[1]}]</strong> = {DOT_TERMS[active[0]][active[1]]}
          </p>
        ) : (
          <p style={{ color: 'var(--ink-soft)' }}>点击 C 的某个格子查看它的内积展开。</p>
        )}
      </div>
    </>
  )
}

// 列的线性组合：点 C 的某一列 → C 的第 j 列是 A 的两列按 B 第 j 列加权求和。
function ColView() {
  const [sel, setSel] = useState<number | null>(null)

  return (
    <>
      <h3 className="sec-h" style={{ color: IKB }}>② 列的线性组合 · col j(C) = A · col j(B)</h3>
      <p style={{ marginTop: 0 }}>
        点击 C 的<strong>某一列</strong>：它是 A 的两列（蓝）按 B 同一列的分量（橙，即权重）加权求和。
      </p>

      <section className="stage">
        <Matrix data={A} name="A" nameColor={IKB}
          tone={() => (sel !== null ? 'blue' : 'plain')} />
        {OP('·')}
        <Matrix data={B} name="B" nameColor={WARM}
          tone={(_i, j) => (sel !== null && j === sel ? 'warm' : 'plain')} />
        {OP('=')}
        <Matrix data={C} name="C"
          tone={(_i, j) => (sel !== null && j === sel ? 'active' : 'plain')}
          onCell={(_i, j) => setSel(sel === j ? null : j)} />
      </section>

      <div className="note">
        {sel !== null ? (
          <p style={{ fontFamily: 'var(--mono)' }}>
            <strong style={{ color: IKB }}>col {sel}(C)</strong> = {B[0][sel]}·col₀(A) + {B[1][sel]}·col₁(A)
            {' = '}{B[0][sel]}·[{A[0][0]},{A[1][0]}]ᵀ + {B[1][sel]}·[{A[0][1]},{A[1][1]}]ᵀ = [{C[0][sel]},{C[1][sel]}]ᵀ
          </p>
        ) : (
          <p style={{ color: 'var(--ink-soft)' }}>点击 C 的某一列查看它是 A 各列的哪种加权组合。</p>
        )}
      </div>
    </>
  )
}

// 行的线性组合：点 C 的某一行 → C 的第 i 行是 B 的两行按 A 第 i 行加权求和。
function RowView() {
  const [sel, setSel] = useState<number | null>(null)

  return (
    <>
      <h3 className="sec-h" style={{ color: IKB }}>③ 行的线性组合 · row i(C) = row i(A) · B</h3>
      <p style={{ marginTop: 0 }}>
        点击 C 的<strong>某一行</strong>：它是 B 的两行（橙）按 A 同一行的分量（蓝，即权重）加权求和。
      </p>

      <section className="stage">
        <Matrix data={A} name="A" nameColor={IKB}
          tone={(i) => (sel !== null && i === sel ? 'blue' : 'plain')} />
        {OP('·')}
        <Matrix data={B} name="B" nameColor={WARM}
          tone={() => (sel !== null ? 'warm' : 'plain')} />
        {OP('=')}
        <Matrix data={C} name="C"
          tone={(i) => (sel !== null && i === sel ? 'active' : 'plain')}
          onCell={(i) => setSel(sel === i ? null : i)} />
      </section>

      <div className="note">
        {sel !== null ? (
          <p style={{ fontFamily: 'var(--mono)' }}>
            <strong style={{ color: IKB }}>row {sel}(C)</strong> = {A[sel][0]}·row₀(B) + {A[sel][1]}·row₁(B)
            {' = '}{A[sel][0]}·[{B[0][0]},{B[0][1]}] + {A[sel][1]}·[{B[1][0]},{B[1][1]}] = [{C[sel][0]},{C[sel][1]}]
          </p>
        ) : (
          <p style={{ color: 'var(--ink-soft)' }}>点击 C 的某一行查看它是 B 各行的哪种加权组合。</p>
        )}
      </div>
    </>
  )
}

// 秩-1 外积之和：点某一个 rank-1 项 → 高亮它的来源（A 的列 ⊗ B 的行），并说明它逐格贡献整个 C。
function OuterView() {
  const [sel, setSel] = useState<number | null>(null)

  return (
    <>
      <h3 className="sec-h" style={{ color: IKB }}>④ 秩-1 外积之和 · C = Σ col k(A) ⊗ row k(B)</h3>
      <p style={{ marginTop: 0 }}>
        点击<strong>某一个 rank-1 项</strong>：它是 A 的一列与 B 的一行的外积，秩只有 1。
        每个 rank-1 项都<strong>逐格贡献整个 C</strong>——C 的每一格 = 两项对应格之和。
        这正是注意力对 value 加权求和、LoRA 低秩分解的几何种子。
      </p>

      <section className="stage">
        {OUTER_TERMS.map((term, k) => (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {k > 0 && OP('+')}
            <Matrix
              data={term.mat}
              name={`R${k + 1}`}
              nameColor={k === 0 ? IKB : WARM}
              tone={() => (sel === k ? (k === 0 ? 'active' : 'activeWarm') : (k === 0 ? 'blue' : 'warm'))}
              onCell={() => setSel(sel === k ? null : k)}
            />
          </div>
        ))}
        {OP('=')}
        <Matrix data={C} name="C" tone={() => 'plain'} />
      </section>

      <div className="note">
        {sel !== null ? (
          <p style={{ fontFamily: 'var(--mono)' }}>
            <strong style={{ color: sel === 0 ? IKB : WARM }}>R{sel + 1}</strong> = col{sel}(A) ⊗ row{sel}(B)
            {' = '}[{OUTER_TERMS[sel].aCol.join(',')}]ᵀ ⊗ [{OUTER_TERMS[sel].bRow.join(',')}]
            <span style={{ fontFamily: 'var(--sans)', color: 'var(--ink-soft)' }}>
              {' '}—— 它和另一项逐格相加，得到完整的 C。
            </span>
          </p>
        ) : (
          <p style={{ color: 'var(--ink-soft)' }}>点击 R1 或 R2 查看这个 rank-1 项由 A 的哪一列、B 的哪一行外积而来。</p>
        )}
      </div>
    </>
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

      {/* 切换视角：复用站点 .chip 胶囊，is-on 即 IKB 高亮，自带 hover 与响应式 */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', margin: '40px 0 4px' }}>
        {VIEW_LABELS.map(({ id, label }) => (
          <button
            key={id}
            className={`chip${view === id ? ' is-on' : ''}`}
            onClick={() => setView(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {view === 'dot' && <DotView />}
      {view === 'col' && <ColView />}
      {view === 'row' && <RowView />}
      {view === 'outer' && <OuterView />}

      <section className="verdict verdict--eq">
        <p>
          <strong>四种视角算的是同一个乘积，只是看法不同。</strong>
          A·B 的答案唯一——点积、列的线性组合、行的线性组合、秩-1 外积之和，
          只是从四个角度<em>读</em>同一个 C。哪种顺手用哪种，换一种写法也认得出是同一件事。
        </p>
      </section>

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
