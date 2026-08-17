import { useState } from 'react'
import type { CSSProperties } from 'react'
import { apply, multiply, transforms, format, type Mat2, type Vec2 } from '../linalg'
import { ChRef } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'
import { TransformPanel } from '../TransformPanel'
import { CodeBlock } from '../components/CodeBlock'

const RUST = '#c75b39'
const IKB = '#002fa7'

type Mode = 'rotate' | 'reflect'

/**
 * Rotation:   Q = [[cosθ, −sinθ], [sinθ, cosθ]]          det = +1
 * Reflection: Q = [[cos2θ, sin2θ], [sin2θ, −cos2θ]]      det = −1
 * (reflection across the line at angle θ° from x-axis)
 */
function buildQ(mode: Mode, theta: number): Mat2 {
  if (mode === 'rotate') return transforms.rotate(theta)
  const a = (theta * 2 * Math.PI) / 180
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [c, s, s, -c]
}

function transposeM(M: Mat2): Mat2 {
  // Row-major [a,b,c,d] → [[a,b],[c,d]]; transpose → [[a,c],[b,d]] → [a,c,b,d]
  return [M[0], M[2], M[1], M[3]]
}

function detM(M: Mat2): number {
  return M[0] * M[3] - M[1] * M[2]
}

function vlen(v: Vec2): number {
  return Math.hypot(v[0], v[1])
}

function fmt(n: number): string {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ---- Q matrix display with colored columns ----
function QMatrix({ Q }: { Q: Mat2 }) {
  return (
    <div className="matrix">
      <span className="matrix-name">Q</span>
      <span className="bracket">[</span>
      <span className="matrix-rows">
        <span>
          <span style={{ color: RUST, fontWeight: 700 }}>{fmt(Q[0])}</span>
          {'  '}
          <span style={{ color: IKB, fontWeight: 700 }}>{fmt(Q[1])}</span>
        </span>
        <span>
          <span style={{ color: RUST, fontWeight: 700 }}>{fmt(Q[2])}</span>
          {'  '}
          <span style={{ color: IKB, fontWeight: 700 }}>{fmt(Q[3])}</span>
        </span>
      </span>
      <span className="bracket">]</span>
    </div>
  )
}

// ---- QᵀQ readout showing it equals I ----
function QtQReadout({ Q }: { Q: Mat2 }) {
  const Qt = transposeM(Q)
  const R = multiply(Qt, Q)
  const [r1, r2] = format(R)
  const isI =
    Math.abs(R[0] - 1) < 1e-4 &&
    Math.abs(R[3] - 1) < 1e-4 &&
    Math.abs(R[1]) < 1e-4 &&
    Math.abs(R[2]) < 1e-4
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
      <span style={{ fontFamily: 'monospace', fontSize: '0.9rem', color: '#555' }}>QᵀQ =</span>
      <div className="matrix">
        <span className="bracket">[</span>
        <span className="matrix-rows">
          <span>{r1}</span>
          <span>{r2}</span>
        </span>
        <span className="bracket">]</span>
      </div>
      {isI && (
        <span style={{ color: '#1a7f4e', fontWeight: 700, fontSize: '1rem' }}>= I ✓</span>
      )}
    </div>
  )
}

// ---- Unit-circle + vectors SVG ----
const SVG_S = 148
const SVG_C = SVG_S / 2
const SVG_U = 52 // 1 unit in px

function toSx(x: number) { return SVG_C + x * SVG_U }
function toSy(y: number) { return SVG_C - y * SVG_U }

function SvgArrow({ to, color, dashed = false }: { to: Vec2; color: string; dashed?: boolean }) {
  const x1 = SVG_C, y1 = SVG_C
  const x2 = toSx(to[0]), y2 = toSy(to[1])
  if (Math.hypot(x2 - x1, y2 - y1) < 2) return null
  const ang = Math.atan2(y1 - y2, x2 - x1)
  const ah = 7, aw = 4
  const bx1 = x2 - ah * Math.cos(ang) - aw * Math.sin(ang)
  const by1 = y2 + ah * Math.sin(ang) - aw * Math.cos(ang)
  const bx2 = x2 - ah * Math.cos(ang) + aw * Math.sin(ang)
  const by2 = y2 + ah * Math.sin(ang) + aw * Math.cos(ang)
  return (
    <g>
      <line
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke={color} strokeWidth={dashed ? 1.5 : 2.5}
        strokeDasharray={dashed ? '4,2' : undefined}
      />
      <polygon points={`${x2},${y2} ${bx1},${by1} ${bx2},${by2}`} fill={color} />
    </g>
  )
}

function LengthCircleSVG({ Q }: { Q: Mat2 }) {
  const v1: Vec2 = [1, 0]
  const v2: Vec2 = [0.6, 0.8]
  const qv1 = apply(Q, v1)
  const qv2 = apply(Q, v2)
  const labelStyle: CSSProperties = { pointerEvents: 'none', userSelect: 'none' }
  return (
    <svg width={SVG_S} height={SVG_S} viewBox={`0 0 ${SVG_S} ${SVG_S}`} style={{ display: 'block' }}>
      {/* axes */}
      <line x1={4} y1={SVG_C} x2={SVG_S - 4} y2={SVG_C} stroke="#e6e8ea" strokeWidth={1} />
      <line x1={SVG_C} y1={4} x2={SVG_C} y2={SVG_S - 4} stroke="#e6e8ea" strokeWidth={1} />
      {/* unit circle */}
      <circle cx={SVG_C} cy={SVG_C} r={SVG_U} fill="none" stroke="#e6e8ea" strokeWidth={1.5} />
      {/* original vectors (dashed) */}
      <SvgArrow to={v1} color="#c0c8d0" dashed />
      <SvgArrow to={v2} color="#c0c8d0" dashed />
      {/* transformed vectors */}
      <SvgArrow to={qv1} color={RUST} />
      <SvgArrow to={qv2} color={IKB} />
      {/* labels */}
      <text x={toSx(qv1[0]) + 5} y={toSy(qv1[1]) + 4}
        fill={RUST} fontSize={10} fontFamily="monospace" fontWeight="bold" style={labelStyle}>
        Qv₁
      </text>
      <text x={toSx(qv2[0]) + 5} y={toSy(qv2[1]) - 3}
        fill={IKB} fontSize={10} fontFamily="monospace" fontWeight="bold" style={labelStyle}>
        Qv₂
      </text>
    </svg>
  )
}

const SNIPPET = `import numpy as np

# Rotation matrix Q(θ)
def rotation(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t),  np.cos(t)]])

Q = rotation(45)

# 1. QᵀQ ≈ I  — orthogonal: columns are orthonormal
print(np.allclose(Q.T @ Q, np.eye(2)))    # True

# 2. det(Q) = 1  — pure rotation (no flip)
print(np.linalg.det(Q))                    # 1.0

# 3. Lengths preserved: ‖Qv‖ = ‖v‖ for any v
v = np.array([3.0, 4.0])
print(np.linalg.norm(v))                   # 5.0
print(np.linalg.norm(Q @ v))               # 5.0  ← identical`

type Preset = { label: string; mode: Mode; theta: number }

const PRESETS: Preset[] = [
  { label: '旋转 30°', mode: 'rotate', theta: 30 },
  { label: '旋转 90°', mode: 'rotate', theta: 90 },
  { label: '镜像 x 轴', mode: 'reflect', theta: 0 },
]

export function OrthogonalRotation() {
  const [theta, setTheta] = useState(45)
  const [mode, setMode] = useState<Mode>('rotate')

  const Q = buildQ(mode, theta)
  const d = Math.round(detM(Q) * 1e6) / 1e6
  const col1: Vec2 = [Q[0], Q[2]]
  const col2: Vec2 = [Q[1], Q[3]]
  const col1Len = vlen(col1)
  const col2Len = vlen(col2)
  const colDot = col1[0] * col2[0] + col1[1] * col2[1]

  const v1: Vec2 = [1, 0]
  const v2: Vec2 = [0.6, 0.8]
  const qv1 = apply(Q, v1)
  const qv2 = apply(Q, v2)

  const applyPreset = (p: Preset) => { setMode(p.mode); setTheta(p.theta) }
  const isActive = (p: Preset) => p.mode === mode && p.theta === theta

  const btnBase: CSSProperties = {
    padding: '0.35rem 0.9rem',
    border: `1.5px solid ${IKB}`,
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontFamily: 'inherit',
    transition: 'background 0.15s, color 0.15s',
  }

  return (
      <ChapterShell
        slug="orthogonal-rotation"
        part="第四部分 · 正交、回归与投影"
        sub="不改变任何长度的变换长什么样？"
        lede={
          <>
        旋转一个图形，形状完好、距离不变——这种 rigid motion 背后是
        <strong> orthogonal matrix</strong> Q。它的两列两两正交且均为单位向量
        （orthonormal），由此推出 <code>QᵀQ = I</code>，即
        <code> Q⁻¹ = Qᵀ</code>。det = +1 是 rotation，det = −1 是 reflection。
        拖动 θ 滑块，亲眼看 F 字形只转不拉、length 永远 1.00。
          </>
        }
      >

      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => applyPreset(p)}
                style={{
                  ...btnBase,
                  background: isActive(p) ? IKB : 'white',
                  color: isActive(p) ? 'white' : IKB,
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="control">
          <label className="slider-row">
            <span style={{ minWidth: '1.2rem', fontFamily: 'monospace', fontSize: '0.9rem', color: '#555' }}>
              θ
            </span>
            <input
              type="range" min={-180} max={180} step={1} value={theta}
              onChange={(e) => setTheta(Number(e.target.value))}
            />
            <span className="param-val">{theta}°</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.6rem', cursor: 'pointer', fontSize: '0.9rem' }}>
            <input
              type="checkbox"
              checked={mode === 'reflect'}
              onChange={(e) => setMode(e.target.checked ? 'reflect' : 'rotate')}
              style={{ accentColor: IKB, width: 15, height: 15 }}
            />
            <span>reflection 模式（det = −1）——镜像，不是旋转</span>
          </label>
          <p style={{ margin: '0.5rem 0 0', fontSize: '0.8rem', color: '#888', lineHeight: 1.6 }}>
            {mode === 'rotate'
              ? 'θ = 逆时针旋转的角度。'
              : 'θ = 镜像轴与 x 轴的夹角（Q 的元素里是 2θ）。拖动 θ 看镜面转起来，F 字跟着换边。'}
          </p>
        </div>
      </section>

      <section className="stage">
        <TransformPanel M={[1, 0, 0, 1]} label="原图" sublabel="恒等" />
        <div className="arrow-sep">→</div>
        <TransformPanel
          M={Q}
          label={mode === 'rotate' ? `rotation ${theta}°` : `reflection ${theta}°`}
          sublabel="Q 作用后"
          active
        />
      </section>

      <section className="readouts">
        <QMatrix Q={Q} />

        <div style={{ marginTop: '1rem' }}>
          <QtQReadout Q={Q} />
        </div>

        <div style={{
          display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginTop: '0.8rem',
          fontFamily: 'monospace', fontSize: '0.88rem', lineHeight: 2,
        }}>
          <div>
            <span style={{ color: '#888' }}>det(Q) = </span>
            <strong style={{ color: Math.abs(d - 1) < 1e-3 ? IKB : RUST }}>{fmt(d)}</strong>
            <span style={{ marginLeft: '0.35rem', color: '#888', fontSize: '0.8rem' }}>
              {Math.abs(d - 1) < 1e-3 ? '— rotation' : '— reflection'}
            </span>
          </div>
          <div>
            <span style={{ color: RUST }}>|col₁| = </span>
            <strong style={{ color: RUST }}>{fmt(col1Len)}</strong>
          </div>
          <div>
            <span style={{ color: IKB }}>|col₂| = </span>
            <strong style={{ color: IKB }}>{fmt(col2Len)}</strong>
          </div>
          <div>
            <span style={{ color: '#888' }}>col₁·col₂ = </span>
            <strong>{fmt(colDot)}</strong>
            <span style={{ marginLeft: '0.3rem', color: '#1a7f4e', fontSize: '0.8rem' }}>⊥ ✓</span>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'flex-start', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div>
            <p style={{ margin: '0 0 0.4rem', fontSize: '0.82rem', color: '#555' }}>
              单位圆上的向量经 Q 后仍在圆上——length 不变
            </p>
            <div style={{ border: '1.5px solid #e6e8ea', borderRadius: 8, overflow: 'hidden', display: 'inline-block' }}>
              <LengthCircleSVG Q={Q} />
            </div>
            <p style={{ margin: '0.3rem 0 0', fontSize: '0.78rem', color: '#aaa' }}>
              虚线 = 变换前 &nbsp;|&nbsp; 实线 = Q 作用后
            </p>
          </div>
          <div style={{ fontFamily: 'monospace', fontSize: '0.86rem', lineHeight: 2 }}>
            <div>v₁ = (1.00, 0.00)&emsp;‖v₁‖ = 1.00</div>
            <div>
              Qv₁ = ({fmt(qv1[0])}, {fmt(qv1[1])})&emsp;
              <strong style={{ color: '#1a7f4e' }}>‖Qv₁‖ = 1.00 ✓</strong>
            </div>
            <div style={{ marginTop: '0.4rem' }}>v₂ = (0.60, 0.80)&emsp;‖v₂‖ = 1.00</div>
            <div>
              Qv₂ = ({fmt(qv2[0])}, {fmt(qv2[1])})&emsp;
              <strong style={{ color: '#1a7f4e' }}>‖Qv₂‖ = 1.00 ✓</strong>
            </div>
          </div>
        </div>
      </section>

      <section className={`verdict ${mode === 'rotate' ? 'verdict--eq' : 'verdict--neq'}`}>
        {mode === 'rotate' ? (
          <p>
            <strong>Rotation：det = +1，F 字形只转不翻。</strong>
            两列始终单位长度且互相垂直——orthonormal。任意向量经 Q 后 length 不变、夹角不变：
            这是 orthogonal matrix 的核心承诺——rigid motion，世界刚性地旋转。
          </p>
        ) : (
          <p>
            <strong>Reflection：det = −1，F 字形被翻转。</strong>
            同样满足 QᵀQ = I、两列 orthonormal——reflection 也是 orthogonal matrix，保长保角。
            但方向感（chirality）反了，F 的"手性"翻了个面。
            det 的正负是区分 rotation 与 reflection 的唯一数字线索。
          </p>
        )}
      </section>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            <strong>RoPE（旋转位置编码）</strong>：给 query 和 key 各自乘上按 token 位置 m
            索引的旋转矩阵 <code>R(mθ)</code>。由于 rotation 保内积，可推导出
            <code> (R(m)q)·(R(n)k) = q·R(n−m)k</code>——点积只依赖<em>相对位置</em> n−m，
            绝对编号自然消掉。这正是 orthogonal 变换"保长保角"的直接应用，也解释了为什么
            RoPE 在长文本外推上比可学习位置编码更稳定：<code>Q⁻¹ = Qᵀ</code> 意味着数值上无损可逆。
          </p>
          <p>
            连回<ChRef slug="spectral" />谱分解里的正交矩阵 Q；<ChRef slug="gram-schmidt" />格拉姆-施密特会教你如何
            <em>构造</em>一组 orthonormal basis——那里的输出正是这里 Q 的列。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：验证 QᵀQ = I 与长度不变</h2>
        <CodeBlock code={SNIPPET} language="python" title="orthogonal_rotation.py" />
      </section>

      </ChapterShell>
  )
}
