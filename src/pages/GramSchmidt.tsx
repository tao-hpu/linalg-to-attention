import { useState, type CSSProperties } from 'react'
import { VectorCanvas, type CanvasVector } from '../components/VectorCanvas'
import { ChRef } from '../components/ChRef'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { sub, dot, norm, normalize, projectOnto, fmt } from '../vec'

const IKB = '#002fa7'
const RUST = '#c75b39'
const GRAY = '#9aa1a9'
const TEAL = '#00b3a4'

const STEP_LABELS = ['① 投影', '② 相减', '③ 归一'] as const

function stepDesc(s: number): string {
  if (s === 1) {
    return '从 a₂ 减去 proj，得 u₂（蓝绿色）。灰色虚线表示「减掉的那段」。u₂ ⊥ u₁。'
  }
  if (s === 2) {
    return '归一化：e₁ = u₁/‖u₁‖，e₂ = u₂/‖u₂‖。直角符号确认彼此正交。'
  }
  return '将 a₂ 投影到 u₁ 方向，得 proj（灰色虚线向量）。投影长度 r₁₂ = a₂·e₁。'
}

interface Preset {
  label: string
  a1: { x: number; y: number }
  a2: { x: number; y: number }
}

const PRESETS: Preset[] = [
  { label: '近乎平行（数值难）', a1: { x: 3, y: 0.5 }, a2: { x: 2.5, y: 1 } },
  { label: '垂直（已正交）',     a1: { x: 3, y: 0 },   a2: { x: 0, y: 2.5 } },
]

const SNIPPET = `import numpy as np

# 输入：两个列向量（构成矩阵 A = [a1 | a2]）
a1 = np.array([3.0, 1.0])
a2 = np.array([2.0, 3.0])

# Step 1 — u1 = a1；e1 = 归一化
u1 = a1.copy()
e1 = u1 / np.linalg.norm(u1)

# Step 2 — 从 a2 减掉它在 e1 上的分量；u2 ⊥ e1
u2 = a2 - np.dot(a2, e1) * e1

# Step 3 — 归一化
e2 = u2 / np.linalg.norm(u2)

# 组装 Q（列 = orthonormal 基）和 R（上三角）
Q = np.column_stack([e1, e2])
R = np.array([[np.dot(a1, e1), np.dot(a2, e1)],
              [0.0,             np.dot(a2, e2)]])

# 验证
print(np.allclose(Q @ R, np.column_stack([a1, a2])))  # True  (QR = A)
print(np.allclose(Q.T @ Q, np.eye(2)))                # True  (Qᵀ Q = I)
print(f"e1·e2 = {np.dot(e1, e2):.2e}")               # ≈ 0`

const btnStyle = (active: boolean): CSSProperties => ({
  background: active ? IKB : 'transparent',
  color: active ? '#fff' : '#1b1f24',
  border: `1.5px solid ${active ? IKB : '#ccc'}`,
  borderRadius: 3,
  padding: '4px 12px',
  marginRight: 6,
  cursor: 'pointer',
  fontFamily: 'inherit',
  fontSize: 'inherit',
})

const presetBtnStyle: CSSProperties = {
  background: 'transparent',
  color: '#1b1f24',
  border: '1.5px solid #ccc',
  borderRadius: 3,
  padding: '4px 10px',
  marginRight: 8,
  cursor: 'pointer',
  fontFamily: 'inherit',
  fontSize: 'inherit',
}

export function GramSchmidt() {
  const [a1, setA1] = useState({ x: 3, y: 1 })
  const [a2, setA2] = useState({ x: 1.5, y: 3 })
  const [step, setStep] = useState(0)

  // ── Gram-Schmidt math ────────────────────────────────────────────────
  const u1 = a1                           // u1 = a1
  const e1 = normalize(u1)               // e1 = u1 / |u1|
  const proj = projectOnto(a2, u1)       // vector projection of a2 onto u1
  const u2 = sub(a2, proj)              // u2 = a2 − proj  (⊥ u1)
  const e2 = normalize(u2)              // e2 = u2 / |u2|

  // QR entries (R is upper-triangular)
  const r11 = norm(u1)                  // ‖a1‖
  const r12 = dot(a2, e1)              // a2·e1  (scalar projection)
  const r22 = norm(u2)                 // ‖u2‖

  // Orthogonality check
  const dotE1E2 = dot(e1, e2)
  const isOrtho = Math.abs(dotE1E2) < 0.01
  const valid = r11 > 1e-6 && r22 > 1e-6

  // ── Event handlers ───────────────────────────────────────────────────
  const onDrag = (id: string, x: number, y: number) => {
    if (id === 'a1') setA1({ x, y })
    else if (id === 'a2') setA2({ x, y })
  }

  // ── Build vector list based on step ─────────────────────────────────
  const projVis = norm(proj) > 1e-6
  const u2Vis = norm(u2) > 1e-6

  const vectors: CanvasVector[] = []
  vectors.push({ id: 'a1', x: a1.x, y: a1.y, color: RUST, label: 'a₁', draggable: true })
  vectors.push({ id: 'a2', x: a2.x, y: a2.y, color: IKB,  label: 'a₂', draggable: true })
  if (projVis) {
    vectors.push({ id: 'proj', x: proj.x, y: proj.y, color: GRAY, label: 'proj', dashed: true, width: 2 })
  }
  if (step >= 1 && u2Vis) {
    vectors.push({ id: 'u2', x: u2.x, y: u2.y, color: TEAL, label: 'u₂', width: 3 })
  }
  if (step >= 2 && r11 > 1e-6) {
    vectors.push({ id: 'e1', x: e1.x, y: e1.y, color: RUST, label: 'e₁', width: 4 })
  }
  if (step >= 2 && u2Vis) {
    vectors.push({ id: 'e2', x: e2.x, y: e2.y, color: IKB, label: 'e₂', width: 4 })
  }

  return (
    <ChapterShell
      slug="gram-schmidt"
      part="第四部分 · 正交、回归与投影"
      lede={
        <>
          Gram-Schmidt 解决的是一个工程问题：手头有一组独立向量，
          想把它们换成互相垂直、长度为 1 的{' '}
          <strong>orthonormal 基</strong>——怎么一步步制造？
          方法是：保留第一个方向；对每个后续向量，
          <strong>减掉它在已有基方向上的投影</strong>，剩下的部分天然垂直；
          再 normalize 成单位长度。把所有正交基列成矩阵就是{' '}
          <code>Q</code>，原坐标变换到新基的系数是{' '}
          <code>R</code>（上三角）——合起来就是{' '}
          <strong>QR 分解</strong>：<code>A = QR</code>。
          拖动下面两个向量，切换步骤逐步观察整个过程。
        </>
      }
    >
      {/* ── Step toggle + presets ──────────────────────────────────────── */}
      <section className="controls">
        <div className="control">
          <span style={{ fontWeight: 600, marginRight: 8 }}>显示步骤：</span>
          {STEP_LABELS.map((label, i) => (
            <button key={label} onClick={() => setStep(i)} style={btnStyle(step === i)}>
              {label}
            </button>
          ))}
        </div>
        <div className="control">
          <span style={{ fontWeight: 600, marginRight: 8 }}>预设：</span>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => { setA1(p.a1); setA2(p.a2); setStep(0) }}
              style={presetBtnStyle}
            >
              {p.label}
            </button>
          ))}
        </div>
      </section>

      {/* ── Canvas + readout panel ─────────────────────────────────────── */}
      <section className="vstage">
        <VectorCanvas
          vectors={vectors}
          onDrag={onDrag}
          size={380}
          range={5}
          overlay={({ sx, sy }) => {
            // Right-angle marker at origin between e1 and e2 (step 2 only)
            if (step < 2 || !valid) return null
            const small = 14
            const ox = sx(0), oy = sy(0)
            const e1sx = sx(e1.x) - ox, e1sy = sy(e1.y) - oy
            const e2sx = sx(e2.x) - ox, e2sy = sy(e2.y) - oy
            const len1 = Math.hypot(e1sx, e1sy), len2 = Math.hypot(e2sx, e2sy)
            if (len1 < 1 || len2 < 1) return null
            const n1x = (e1sx / len1) * small, n1y = (e1sy / len1) * small
            const n2x = (e2sx / len2) * small, n2y = (e2sy / len2) * small
            return (
              <polyline
                points={`${ox + n1x},${oy + n1y} ${ox + n1x + n2x},${oy + n1y + n2y} ${ox + n2x},${oy + n2y}`}
                fill="none"
                stroke={GRAY}
                strokeWidth={1.5}
              />
            )
          }}
        >
          {({ sx, sy }) => (
            <>
              {/* Guide line along u1 direction (step 0): shows the axis we project onto */}
              {step === 0 && r11 > 1e-6 && (
                <line
                  x1={sx(-e1.x * 5)} y1={sy(-e1.y * 5)}
                  x2={sx(e1.x * 5)}  y2={sy(e1.y * 5)}
                  stroke={RUST} strokeWidth={1} strokeDasharray="2 6" opacity={0.25}
                />
              )}
              {/* Dashed line: from a2 tip to proj tip (step 1+), shows a2 = proj + u2 */}
              {step >= 1 && projVis && u2Vis && (
                <line
                  x1={sx(a2.x)} y1={sy(a2.y)}
                  x2={sx(proj.x)} y2={sy(proj.y)}
                  stroke={GRAY} strokeWidth={2} strokeDasharray="5 5"
                />
              )}
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <p style={{ fontSize: '0.82rem', color: GRAY, margin: '0 0 10px', lineHeight: 1.5 }}>
            {stepDesc(step)}
          </p>

          <div className="vrow"><span>a₁（rust）</span><code>({fmt(a1.x)}, {fmt(a1.y)})</code></div>
          <div className="vrow"><span>a₂（IKB）</span><code>({fmt(a2.x)}, {fmt(a2.y)})</code></div>

          <div style={{ borderTop: '1px solid #eee', margin: '8px 0' }} />

          <div className="vrow">
            <span>u₂ = a₂ − {fmt(r12)}·e₁</span>
            <code>({fmt(u2.x)}, {fmt(u2.y)})</code>
          </div>
          <div className="vrow"><span>e₁</span><code>({fmt(e1.x)}, {fmt(e1.y)})</code></div>
          <div className="vrow"><span>e₂</span><code>({fmt(e2.x)}, {fmt(e2.y)})</code></div>

          <div
            className="vrow"
            style={{ color: isOrtho ? '#0a7d52' : RUST, fontWeight: 600 }}
          >
            <span>e₁·e₂</span>
            <code style={{ color: 'inherit' }}>
              {fmt(dotE1E2)}&nbsp;{isOrtho ? '✓' : '(非零 — 拖动看效果)'}
            </code>
          </div>

          <div style={{ borderTop: '1px solid #eee', margin: '8px 0' }} />

          {/* QR matrices */}
          <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <div className="matrix">
              <span className="matrix-name">Q</span>
              <span className="bracket">[</span>
              <span className="matrix-rows">
                <span>{fmt(e1.x, 3)}  {fmt(e2.x, 3)}</span>
                <span>{fmt(e1.y, 3)}  {fmt(e2.y, 3)}</span>
              </span>
              <span className="bracket">]</span>
            </div>
            <div className="matrix">
              <span className="matrix-name">R</span>
              <span className="bracket">[</span>
              <span className="matrix-rows">
                <span>{fmt(r11)}  {fmt(r12)}</span>
                <span>0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{fmt(r22)}</span>
              </span>
              <span className="bracket">]</span>
            </div>
          </div>

          {!valid && (
            <p className="vhint" style={{ color: RUST, marginTop: 8 }}>
              向量近乎零或线性相关（a₁ ∥ a₂）——请拖动让两者线性无关。
            </p>
          )}

          <p className="vhint">
            拖动 a₁、a₂ 实时更新所有步骤；切换上方按钮逐步观察。
            「近乎平行」时注意 r₂₂ 很小——数值上就容易出问题。
          </p>
        </div>
      </section>

      {/* ── Explanatory note ───────────────────────────────────────────── */}
      <section className="note">
        <p>
          Gram-Schmidt 的核心洞察：对任意线性无关的一组向量，
          总能把它们「洗成」orthonormal 的——关键是
          <strong>每步只减掉已有基方向的分量</strong>，
          剩余的残差天然与所有已有基垂直。
          归一化不影响方向和正交性，只是统一长度。
        </p>
        <p>
          产物 <code>Q</code>（orthonormal 列）和 <code>R</code>（上三角）满足{' '}
          <code>A = QR</code>、<code>Q<sup>T</sup>Q = I</code>。
          R 是上三角，因为第 k 列的 e<sub>k</sub> 只依赖前 k 个原始向量——
          构造是单向推进的。
          QR 分解在数值上比直接求 (A<sup>T</sup>A)<sup>−1</sup> 更稳定：
          最小二乘的标准数值解法正是 QR，而非正规方程。
        </p>
      </section>

      {/* ── Bridge to LLM ─────────────────────────────────────────────── */}
      <Bridge>
        <p>
          正交基 = 一组互不干扰、数值稳定的坐标轴。
          QR 是最小二乘和很多分解的数值主力：
          对 Q 做操作远比通过 A 更干净，不会因矩阵病态而崩溃。
        </p>
        <p>
          注意力里的<strong>多头（multi-head）</strong>，
          以及各种「想让表征解耦」的设计，本质上都在追求{' '}
          <strong>「彼此正交、各管一摊」</strong>的基——
          每个头学一个相对独立的子空间。
          不完全是 Gram-Schmidt 的产物，但想法同根：
          方向不互相干扰，信息读写就更准。
          连回<ChRef slug="orthogonal-rotation" />正交矩阵、<ChRef slug="orthogonal-projection" />正交投影，接向<ChRef slug="svd" /> SVD。
        </p>
      </Bridge>

      {/* ── Code block ─────────────────────────────────────────────────── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：Gram-Schmidt 与 QR 分解</h2>
        <CodeBlock code={SNIPPET} language="python" title="gram_schmidt.py" />
      </section>
    </ChapterShell>
  )
}
