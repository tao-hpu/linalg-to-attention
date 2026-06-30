import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import {
  sub, projectOnto, scalarProjection, norm, angleBetween, degrees, fmt,
} from '../vec'

const IKB = '#002fa7'
const TEAL = '#00b3a4'
const PROJ = '#e08a00'   // 投影向量的高亮色（沿 b 的分量）
const DROP = '#9aa1a9'   // 残差 / 误差虚线

const SNIPPET = `import { sub, scalarProjection, projectOnto, norm } from './vec'

// 「投影」= a 在 b 方向上占多少 —— 把 a 拆成沿 b + 垂直 b 两部分
const a = { x: 2.5, y: 3 }
const b = { x: 4, y: 1 }

// 标量投影：a 在 b 方向上的「长度」(带符号) = |a|·cosθ
const s = scalarProjection(a, b)     // = dot(a, b) / |b|

// 向量投影：把那段长度还原成一个真正沿 b 的向量
const proj = projectOnto(a, b)       // = b · (dot(a, b) / dot(b, b))

// 残差：a 减掉投影，剩下的部分必然垂直于 b
const residual = sub(a, proj)        // = a − proj_b a
// dot(residual, b) ≈ 0  ← b 完全「看不见」这部分信息`

export function Projection() {
  const [a, setA] = useState({ x: 2.5, y: 3 })
  const [b, setB] = useState({ x: 4, y: 1 })

  const proj = projectOnto(a, b)
  const residual = sub(a, proj)
  const scalar = scalarProjection(a, b)
  const theta = degrees(angleBetween(a, b))
  const resLen = norm(residual)

  const onDrag = (id: string, x: number, y: number) => {
    if (id === 'a') setA({ x, y })
    else if (id === 'b') setB({ x, y })
  }

  return (
    <ChapterShell
      slug="projection"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          投影回答一个很具体的问题：<strong>站在 b 的方向上看，a 到底"占了多少"？</strong>
          答案是把 a 的影子打在 b 这条线上——影子的长度就是 a 沿 b 的分量，
          而 a 减掉这个影子剩下的部分，<strong>正好垂直于 b</strong>，是 b 怎么也看不见的。
          拖动下面的两个箭头，看影子和那条垂直的"误差线"怎么变。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[
            { id: 'proj', x: proj.x, y: proj.y, color: PROJ, label: 'proj_b a', width: 5 },
            { id: 'b', x: b.x, y: b.y, color: TEAL, label: 'b', draggable: true },
            { id: 'a', x: a.x, y: a.y, color: IKB, label: 'a', draggable: true },
          ]}
          onDrag={onDrag}
          size={380}
          range={5}
          overlay={({ sx, sy }) => {
            // 投影端点处的直角标记，强调 a−proj ⊥ b
            if (norm(proj) < 1e-6 || resLen < 1e-6) return null
            const px = sx(proj.x), py = sy(proj.y)
            const ax = sx(a.x), ay = sy(a.y)
            const ox = sx(0), oy = sy(0)
            const len1 = Math.hypot(ax - px, ay - py)
            const len2 = Math.hypot(ox - px, oy - py)
            if (len1 < 1e-6 || len2 < 1e-6) return null
            const m = 12
            const u1x = ((ax - px) / len1) * m, u1y = ((ay - py) / len1) * m
            const u2x = ((ox - px) / len2) * m, u2y = ((oy - py) / len2) * m
            return (
              <polyline
                points={`${px + u1x},${py + u1y} ${px + u1x + u2x},${py + u1y + u2y} ${px + u2x},${py + u2y}`}
                fill="none" stroke={DROP} strokeWidth={1.5}
              />
            )
          }}
        >
          {({ sx, sy }) => (
            <>
              {/* b 所在的整条方向线（投影落在它上面） */}
              <line
                x1={sx(-b.x * 2)} y1={sy(-b.y * 2)}
                x2={sx(b.x * 2)} y2={sy(b.y * 2)}
                stroke={TEAL} strokeWidth={1} strokeDasharray="2 5" opacity={0.5}
              />
              {/* 残差 / 误差：从 a 的端点垂直落到投影的端点 */}
              <line
                x1={sx(a.x)} y1={sy(a.y)}
                x2={sx(proj.x)} y2={sy(proj.y)}
                stroke={DROP} strokeWidth={2} strokeDasharray="5 5"
              />
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>a（IKB）</span><code>({fmt(a.x)}, {fmt(a.y)})</code></div>
          <div className="vrow"><span>b（方向）</span><code>({fmt(b.x)}, {fmt(b.y)})</code></div>
          <div className="vrow"><span>夹角 θ</span><code>{fmt(theta, 1)}°</code></div>
          <div className="vrow"><span>标量投影 |a|cosθ</span><code>{fmt(scalar)}</code></div>
          <div className="vrow"><span>投影向量 proj_b a</span><code>({fmt(proj.x)}, {fmt(proj.y)})</code></div>
          <div className="vrow"><span>残差 a−proj 长度</span><code>{fmt(resLen)}</code></div>
          <p className="vhint">
            橙色粗箭头是 a 落在 b 上的影子；灰色虚线是 a 端点垂直落下的"误差"。
            θ 超过 90° 时标量投影会变成负数，影子也会倒向 b 的反方向。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          投影把 a 干净地拆成<strong>两块互不重叠的部分</strong>：
          <strong>沿 b 的分量（proj）</strong> + <strong>垂直于 b 的分量（residual）</strong>。
          沿 b 的那一块，就是从 b 的角度能"读出"的、关于 a 的<strong>全部信息</strong>；
          垂直的那一块，b 完全看不见——投影做的，正是"只保留 b 看得见的，丢掉其余"。
          这也解释了为什么残差永远垂直于 b：如果它在 b 上还有任何分量，那就说明影子还没打满，没"读干净"。
        </p>
      </section>

      <Bridge>
        <p>
          这正是后面注意力"读出 value"和"回归 = 投影"的种子。注意力用一组权重去<strong>线性组合 value 向量</strong>，
          本质就是把信息投影、读出——只保留"当前 query 看得见"的那部分。
        </p>
        <p>
          第 19 节会专门讲<strong>最小二乘回归 = 把目标投影到特征张成的空间</strong>：
          回归的几何本质就是这个投影——预测值是目标在特征空间里的影子，
          而残差（误差）必然垂直于所有特征，和这里 a−proj ⊥ b 是同一件事。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：标量投影、向量投影与残差</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="projection.ts" />
      </section>
    </ChapterShell>
  )
}
