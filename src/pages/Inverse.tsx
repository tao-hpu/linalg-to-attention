import { useState } from 'react'
import { Link } from 'react-router-dom'
import { multiply, format, nearlyEqual, type Mat2, IDENTITY } from '../linalg'
import { TransformPanel } from '../TransformPanel'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── 品牌色 ──────────────────────────────────────────────────────────────────
const IKB    = '#002fa7'
const DANGER = '#c0392b'
const DET_EPS = 1e-6

// ── 预设 ────────────────────────────────────────────────────────────────────
interface Preset { name: string; a: number; b: number; c: number; d: number; danger?: boolean }

const PRESETS: Preset[] = [
  { name: '可逆示例',      a: 2,    b: 0.5,  c: 0,   d: 1.5 },
  { name: '旋转 45°',     a: 0.7,  b: -0.7, c: 0.7, d: 0.7 },
  { name: '切变 k=1.5',   a: 1,    b: 1.5,  c: 0,   d: 1   },
  { name: '压扁 (det=0)', a: 1,    b: 0,    c: 0,   d: 0,  danger: true },
]

// ── 滑块行 ──────────────────────────────────────────────────────────────────
interface SliderRow { key: string; val: number; set: (v: number) => void; label: string }

// ── 数字格式化 ───────────────────────────────────────────────────────────────
function fmtNum(n: number): string {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── 矩阵展示块（复用 Ch02 样式）──────────────────────────────────────────────
function MatrixReadout({ M, name }: { M: Mat2; name: string }) {
  const [r1, r2] = format(M)
  return (
    <div className="matrix">
      <span className="matrix-name">{name}</span>
      <span className="bracket">[</span>
      <span className="matrix-rows"><span>{r1}</span><span>{r2}</span></span>
      <span className="bracket">]</span>
    </div>
  )
}

// ── 奇异时的占位面板 ──────────────────────────────────────────────────────────
function SingularPanel() {
  return (
    <figure className="panel">
      <div style={{
        width: 260,
        height: 260,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#fff8f7',
        border: `2px dashed ${DANGER}`,
        borderRadius: 4,
        gap: '0.75rem',
      }}>
        <span style={{ fontSize: '2.8rem', lineHeight: 1 }}>∅</span>
        <span style={{ color: DANGER, fontWeight: 700, fontSize: '1rem' }}>逆不存在</span>
        <span style={{
          color: '#666',
          fontSize: '0.82rem',
          textAlign: 'center',
          lineHeight: 1.6,
          padding: '0 1.5rem',
        }}>
          det = 0，平面已被压扁成一条线<br />
          丢失的维度无法还原
        </span>
      </div>
      <figcaption>
        <span className="panel-label" style={{ color: DANGER }}>M⁻¹ 不存在</span>
        <span className="panel-sub">singular matrix</span>
      </figcaption>
    </figure>
  )
}

// ── 代码片段 ─────────────────────────────────────────────────────────────────
const SNIPPET = `\
// TypeScript — Mat2 = [a, b, c, d]（行主序，即 [[a,b],[c,d]]）

function det(M: Mat2): number {
  const [a, b, c, d] = M
  return a * d - b * c         // 面积缩放比；= 0 ⟺ singular（奇异）
}

function inv(M: Mat2): Mat2 {
  const d = det(M)
  if (Math.abs(d) < 1e-9) throw new Error('det ≈ 0，逆不存在 (singular)')
  const [a, b, c, dd] = M
  return [dd / d, -b / d, -c / d, a / d]  // 1/det · [[d,−b],[−c,a]]
}

const M: Mat2   = [2, 0.5, 0, 1.5]
const Minv      = inv(M)              // M⁻¹
const I_check   = multiply(Minv, M)  // M⁻¹·M ≈ [[1,0],[0,1]]
// nearlyEqual(I_check, IDENTITY) → true ✓

// 奇异情形 — det = 0，不可逆
const Msing: Mat2 = [1, 0, 0, 0]    // 投影到 x 轴
// inv(Msing) → throws 'det ≈ 0，逆不存在 (singular)'`

// ── 主组件 ───────────────────────────────────────────────────────────────────
export function Inverse() {
  const [a, setA] = useState(2)
  const [b, setB] = useState(0.5)
  const [c, setC] = useState(0)
  const [d, setD] = useState(1.5)

  // 当前矩阵 M
  const M: Mat2 = [a, b, c, d]
  const det = a * d - b * c
  const singular = Math.abs(det) < DET_EPS

  // 逆矩阵：仅在 det ≠ 0 时计算
  const inv: Mat2 | null = singular
    ? null
    : [d / det, -b / det, -c / det, a / det]

  // 验证合成结果 M⁻¹·M ≈ I
  const composed: Mat2 | null = inv ? multiply(inv, M) : null
  const isIdentity: boolean = composed !== null && nearlyEqual(composed, IDENTITY)

  const me = findChapter('inverse')!
  const { prev, next } = neighbors('inverse')

  const sliders: SliderRow[] = [
    { key: 'a', val: a, set: setA, label: 'a  （行 1 列 1）' },
    { key: 'b', val: b, set: setB, label: 'b  （行 1 列 2）' },
    { key: 'c', val: c, set: setC, label: 'c  （行 2 列 1）' },
    { key: 'd', val: d, set: setD, label: 'd  （行 2 列 2）' },
  ]

  return (
    <article className="page">

      {/* ── 头部 ─────────────────────────────────────────────────────────── */}
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第三部分 · 方阵的秘密
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          逆矩阵
          <span className="zh-sub">变换能"撤销"吗？</span>
        </h1>
        <p className="lede">
          inverse M⁻¹ 是那个<strong>把 M 的效果完全撤销</strong>的变换——先做 M、再做 M⁻¹，
          等于什么都没做：<code>M⁻¹M = I</code>（identity）。
          关键约束：inverse <strong>只在 det ≠ 0 时存在</strong>。
          一旦 M 把整个平面压扁成一条线（det = 0，rank 下降），
          那条被"吞掉"的维度再也找不回来——所以没有任何矩阵能把它还原。
        </p>
      </header>

      {/* ── 控制区 ───────────────────────────────────────────────────────── */}
      <section className="controls">

        {/* 滑块 */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">矩阵 M</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>拖动调整各元素</span>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.6rem 2rem',
            paddingTop: '0.4rem',
          }}>
            {sliders.map(({ key, val, set, label }) => (
              <div key={key}>
                <div style={{ fontSize: '0.82rem', color: '#555', marginBottom: '0.2rem' }}>
                  {label}
                </div>
                <label className="slider-row">
                  <input
                    type="range" min={-3} max={3} step={0.1}
                    value={val}
                    onChange={(e) => set(Number(e.target.value))}
                  />
                  <span className="param-val">{fmtNum(val)}</span>
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* 预设 */}
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>快速切换</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map((p) => (
              <button
                key={p.name}
                onClick={() => { setA(p.a); setB(p.b); setC(p.c); setD(p.d) }}
                style={{
                  padding: '0.35rem 0.9rem',
                  border: `1.5px solid ${p.danger ? DANGER : IKB}`,
                  borderRadius: 4,
                  background: 'white',
                  color: p.danger ? DANGER : IKB,
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                }}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── 变换舞台 ─────────────────────────────────────────────────────── */}
      <section className="stage">
        <TransformPanel
          M={IDENTITY}
          label="原图"
          sublabel="identity，变换前"
        />
        <div className="arrow-sep">→ M →</div>
        <TransformPanel
          M={M}
          label="施加 M"
          sublabel="变换后"
          active
        />
        <div className="arrow-sep" style={{ color: singular ? DANGER : undefined }}>
          → M⁻¹ →
        </div>
        {singular
          ? <SingularPanel />
          : <TransformPanel
              M={composed ?? IDENTITY}
              label="再施加 M⁻¹"
              sublabel={isIdentity ? '= I，完美还原 ✓' : '≈ I'}
              active
            />
        }
      </section>

      {/* ── 数值展示 ─────────────────────────────────────────────────────── */}
      <section className="readouts">

        {/* M */}
        <MatrixReadout M={M} name="M" />

        {/* det */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '0.25rem',
          padding: '0.6rem 1.2rem',
          border: `1.5px solid ${singular ? DANGER : '#e6e8ea'}`,
          borderRadius: 6,
          background: singular ? '#fff8f7' : '#fff',
        }}>
          <span style={{ fontSize: '0.78rem', color: '#888', letterSpacing: '0.03em' }}>
            det(M)
          </span>
          <span style={{
            fontSize: '1.8rem',
            fontWeight: 700,
            color: singular ? DANGER : IKB,
            fontVariantNumeric: 'tabular-nums',
            lineHeight: 1,
          }}>
            {fmtNum(det)}
          </span>
          {singular && (
            <span style={{ fontSize: '0.75rem', color: DANGER, fontWeight: 600 }}>
              奇异 (singular)
            </span>
          )}
        </div>

        {/* M⁻¹ 或"逆不存在" */}
        {inv !== null
          ? <MatrixReadout M={inv} name="M⁻¹" />
          : (
            <div className="matrix" style={{ border: `1.5px solid ${DANGER}` }}>
              <span className="matrix-name" style={{ color: DANGER }}>M⁻¹</span>
              <span style={{
                color: DANGER,
                fontWeight: 700,
                fontSize: '0.95rem',
                padding: '0.25rem 0.5rem',
              }}>
                逆不存在
              </span>
            </div>
          )
        }
      </section>

      {/* ── 结论框 ───────────────────────────────────────────────────────── */}
      <section className={`verdict ${singular ? 'verdict--neq' : 'verdict--eq'}`}>
        {singular ? (
          <p>
            <strong style={{ color: DANGER }}>det = 0，逆矩阵不存在（singular matrix）。</strong>{' '}
            M 把整个平面压扁成了一条线——一整个维度的信息被永久抹掉。
            不管事后用什么矩阵作用在压扁的结果上，都无法还原那些消失的向量。
            这就是 singular 矩阵的本质：<strong>不可逆，rank-deficient（秩亏缺）</strong>。
            试试「压扁 (det=0)」预设，观察 M 把 F 字压成一条线段。
          </p>
        ) : (
          <p>
            <strong>det ≠ 0，M 是 invertible（可逆）的。</strong>{' '}
            M⁻¹·M {isIdentity ? '=' : '≈'} I——右边面板的 F 字完美还原到原始位置。
            公式{' '}<code>M⁻¹ = 1/det · [[d, −b], [−c, a]]</code>{' '}
            把 det 当分母：det 越接近 0，逆矩阵的元素绝对值就越大（变换越"极端"），
            直到 det = 0 时分母为零，inverse 彻底消失。
          </p>
        )}
      </section>

      {/* ── LLM 桥接 ─────────────────────────────────────────────────────── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            Transformer 里有些操作是<strong>有意不可逆</strong>的——
            ReLU 把负值置零、降维投影（<code>d_model → d_k</code>）把高维空间压入低维、
            softmax 把分数归一化为概率——它们的 det ≈ 0，信息一旦被"吞掉"就无法还原。
            这不是 bug，而是模型在<strong>主动做选择</strong>：压缩、聚焦、忽略无关信息。
          </p>
          <p>
            反例：RoPE 旋转位置编码是<strong>正交变换</strong>（第 17 节），
            det = ±1，完全 invertible——位置信息被旋转进去，也可以被精确旋转出来。
            可逆 ⟺ det ≠ 0 ⟺ 满秩（连回第 12、13 节）。
          </p>
        </div>
      </section>

      {/* ── 代码 ─────────────────────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：det、inv、奇异情形</h2>
        <CodeBlock code={SNIPPET} language="typescript" title="inverse.ts" />
      </section>

      {/* ── 翻页 ─────────────────────────────────────────────────────────── */}
      <nav className="pager">
        {prev
          ? (
            <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
              <span className="pager-dir">← 上一章</span>
              <span className="pager-title">{prev.num} {prev.title}</span>
            </Link>
          )
          : <span />}
        {next
          ? (
            <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
              <span className="pager-dir">下一章 →</span>
              <span className="pager-title">
                {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
              </span>
            </Link>
          )
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
