import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { sub, dot, norm, projectOnto, fmt } from '../vec'

const IKB   = '#002fa7'   // a 方向 / subspace 线
const RUST  = '#c75b39'   // b 向量
const PROJ  = '#e08a00'   // 投影向量（沿 span(a)）
const RESID = '#9aa1a9'   // residual 虚线

const SNIPPET = `import { dot, scale, sub, norm } from './vec'

const a = { x: 3, y: 1 }   // 子空间方向（span(a) 是一条过原点的线）
const b = { x: 1, y: 4 }   // 待投影的向量

// 正交投影：proj_a(b) = (a·b / a·a) · a
const scalar = dot(a, b) / dot(a, a)
const proj   = scale(a, scalar)       // b 在 span(a) 上的最近点

// residual = b − proj  ← 被最小化的误差
const residual = sub(b, proj)
console.log(norm(residual))           // ‖b−proj‖ = b 到子空间的最短距离

// 核心：residual ⊥ a（least squares 成立的原因）
console.log(dot(residual, a))         // ≈ 0  ✓

// Projection matrix  P = a aᵀ / aᵀa（idempotent：P²=P）
// P = [[a.x², a.x·a.y],
//      [a.y·a.x, a.y²]] / dot(a, a)
// P @ b = proj，  P @ proj = proj（再投影不变）`

export function OrthogonalProjection() {
  const [a, setA] = useState({ x: 3, y: 1 })
  const [b, setB] = useState({ x: 1, y: 3 })

  const proj     = projectOnto(b, a)
  const residual = sub(b, proj)
  const residLen = norm(residual)
  const dotCheck = dot(residual, a)

  const onDrag = (id: string, x: number, y: number) => {
    if (id === 'a') setA({ x, y })
    else if (id === 'b') setB({ x, y })
  }

  return (
    <ChapterShell
      slug="orthogonal-projection"
      part="第四部分 · 正交、回归与投影"
      lede={
        <>
          正交投影回答一个更深的问题：<strong>在子空间里，距离 b 最近的点在哪里？</strong>
          答案就是把 b 垂直"拍"到子空间上得到的 proj——
          剩下的误差（residual = b − proj）必然<strong>垂直于整个子空间</strong>。
          这个垂直性正是 least squares 能工作的全部秘密：
          最优近似的误差必须与所有可以表示的方向正交。
          拖动下面的箭头，观察直角标记如何始终成立。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[
            { id: 'proj', x: proj.x, y: proj.y, color: PROJ, label: 'proj_a b', width: 5 },
            { id: 'a', x: a.x, y: a.y, color: IKB, label: 'a', draggable: true },
            { id: 'b', x: b.x, y: b.y, color: RUST, label: 'b', draggable: true },
          ]}
          onDrag={onDrag}
          size={380}
          range={5}
          overlay={({ sx, sy }) => {
            if (norm(residual) < 1e-6 || norm(a) < 1e-6) return null
            const px = sx(proj.x), py = sy(proj.y)
            const btx = sx(b.x), bty = sy(b.y)
            const ox = sx(0), oy = sy(0)
            // direction: proj → b tip (residual direction in SVG coords)
            const rLen = Math.hypot(btx - px, bty - py)
            // direction: proj → origin (along the line, since proj lies on span(a))
            const oLen = Math.hypot(ox - px, oy - py)
            if (rLen < 1e-6 || oLen < 5) return null
            const m = 10
            const u1x = ((btx - px) / rLen) * m
            const u1y = ((bty - py) / rLen) * m
            const u2x = ((ox - px) / oLen) * m
            const u2y = ((oy - py) / oLen) * m
            return (
              <polyline
                points={`${px + u1x},${py + u1y} ${px + u1x + u2x},${py + u1y + u2y} ${px + u2x},${py + u2y}`}
                fill="none" stroke={RESID} strokeWidth={1.5}
              />
            )
          }}
        >
          {({ sx, sy }) => (
            <>
              {/* span(a) 的完整方向线，双向延伸到画布边缘 */}
              <line
                x1={sx(-a.x * 10)} y1={sy(-a.y * 10)}
                x2={sx(a.x * 10)} y2={sy(a.y * 10)}
                stroke={IKB} strokeWidth={1} strokeDasharray="3 6" opacity={0.3}
              />
              {/* residual：从 proj 端点到 b 端点的虚线段 */}
              <line
                x1={sx(proj.x)} y1={sy(proj.y)}
                x2={sx(b.x)} y2={sy(b.y)}
                stroke={RESID} strokeWidth={2} strokeDasharray="5 5"
              />
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>b（rust）</span><code>({fmt(b.x)}, {fmt(b.y)})</code></div>
          <div className="vrow"><span>a（方向）</span><code>({fmt(a.x)}, {fmt(a.y)})</code></div>
          <div className="vrow"><span>proj_a b</span><code>({fmt(proj.x)}, {fmt(proj.y)})</code></div>
          <div className="vrow"><span>‖b − proj‖（最短距离）</span><code>{fmt(residLen)}</code></div>
          <div className="vrow"><span>(b−proj)·a</span><code>{fmt(dotCheck, 6)}</code></div>
          <div className="vrow"><span>误差 ⊥ 子空间</span><code>✓</code></div>
          <p className="vhint">
            橙色粗箭头是 b 投影到 span(a) 上的最近点（proj）；灰色虚线是 residual；
            直角标记说明 residual ⊥ subspace。
            ‖b−proj‖ 是 b 到这条直线的最短距离——
            直线上不存在比 proj 更近的点，这就是 least squares 的几何。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          正交投影把 b 干净地拆成两块：
          <strong>子空间里的影子（proj）</strong>和
          <strong>垂直于子空间的 residual</strong>。
          residual 的长度 ‖b−proj‖ 就是 b 到 span(a) 的最短距离——
          直线上任何其他点离 b 都更远。
          这就是 least squares 的几何本质：
          <strong>最优近似 = 正交投影，最优误差 ⊥ 子空间</strong>。
          如果 residual 在子空间上还留着任何分量，那就说明"影子还没打满"——
          把那个分量补进去还能让误差更小，说明原来的近似还不是最优的。
          拖动 b 和 a：(b−proj)·a 始终 ≈ 0，直角标记始终成立。
        </p>
      </section>

      <Bridge>
        <p>
          正交投影 = 「把一个向量在某个子空间里能表示的部分挑出来，剩下的丢掉」。
          <strong>多头注意力</strong>让每个 head 用自己的 W_Q/W_K/W_V
          在一个更小的 subspace 里独立投影、各管一摊；
          PCA 降维、最小二乘<strong>回归 = 投影</strong>（下一节 20 专门讲）全是同一套几何。
        </p>
        <p>
          <code>P²=P</code>（idempotent）：projection matrix 是幂等的——
          把一个已经在 subspace 里的向量再投影一次，结果不变。
          这个性质说明<strong>「投完就够了」</strong>：
          多头注意力里每个 head 的子空间投影，
          叠多少层线性变换，subspace 内的信息都不会再被二次拉走。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：正交投影、residual 与 P²=P</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="orthogonal-projection.ts" />
      </section>
    </ChapterShell>
  )
}
