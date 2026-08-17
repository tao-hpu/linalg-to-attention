import { useState, type ReactNode } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChRef } from '../components/ChRef'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import {
  dot, norm, angleBetween, degrees, cosineSimilarity,
  scalarProjection, projectOnto, fmt,
} from '../vec'

const IKB = '#002fa7'
const TEAL = '#00b3a4'
const ACCENT = '#b23a00' // 投影高亮（橙）
const INK = '#5b6168'

const SNIPPET = `// 内积只有一行：把对应分量相乘再相加
const dot = (a, b) => a.x * b.x + a.y * b.y

// 它和「长度 × 夹角」是同一件事：
//   a·b = |a| · |b| · cosθ
const norm = (a) => Math.hypot(a.x, a.y)
const angleBetween = (a, b) =>
  Math.acos(dot(a, b) / (norm(a) * norm(b)))   // 0..π

// 去掉长度只看方向，就是「余弦相似度」——纯粹的"像不像"
const cosineSimilarity = (a, b) => dot(a, b) / (norm(a) * norm(b))

// 一个数，三种含义：
//   dot > 0  →  方向相近，越大越相关
//   dot ≈ 0  →  几乎垂直，互不相关
//   dot < 0  →  方向相反`

interface Verdict {
  text: string
  color: string
  bg: string
}

function verdictFor(cos: number): Verdict {
  if (cos > 0.15) return { text: '内积 > 0：方向相近，越大越相关', color: '#0a7d52', bg: '#e7f6ee' }
  if (cos < -0.15) return { text: '内积 < 0：方向相反', color: '#b23a00', bg: '#fbece3' }
  return { text: '内积 ≈ 0：几乎垂直，互不相关', color: INK, bg: '#eef0f2' }
}

export function DotProduct() {
  const [a, setA] = useState({ x: 3, y: 1 })
  const [b, setB] = useState({ x: 1, y: 2.5 })

  const onDrag = (id: string, x: number, y: number) => {
    if (id === 'a') setA({ x, y })
    else setB({ x, y })
  }

  const d = dot(a, b)
  const na = norm(a)
  const nb = norm(b)
  const theta = angleBetween(a, b)
  const cos = cosineSimilarity(a, b)
  const sProj = scalarProjection(a, b) // |a|cosθ（带符号的投影长度）
  const foot = projectOnto(a, b)       // a 投影到 b 上的落脚点
  const verdict = verdictFor(cos)

  return (
    <ChapterShell
      slug="dot-product"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          两个向量"<strong>像不像</strong>"，能不能用<strong>一个数</strong>说清楚？能——这个数叫
          <strong>内积</strong>（点积，<code>a·b</code>）。把对应分量相乘再相加就得到它，
          而它恰好等于 <code>|a|·|b|·cosθ</code>：<strong>方向越接近，这个数越大</strong>。
          下面两个箭头都能<strong>拖动</strong>，盯着面板里的 <code>a·b</code> 和夹角一起变。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[
            { id: 'a', x: a.x, y: a.y, color: IKB, label: 'a', draggable: true },
            { id: 'b', x: b.x, y: b.y, color: TEAL, label: 'b', draggable: true },
          ]}
          onDrag={onDrag}
          size={380}
          range={5}
          snap={0.5}
        >
          {({ sx, sy }) => {
            const minLen = Math.min(na, nb)
            const elems = [] as ReactNode[]

            // —— 投影：a 在 b 上的落脚点 + 从 a 端点垂下来的虚线 ——
            if (nb > 1e-6 && na > 1e-6) {
              // 高亮的投影线段（原点 → 落脚点），画在 b 之上一层粗描边
              elems.push(
                <line key="proj" x1={sx(0)} y1={sy(0)} x2={sx(foot.x)} y2={sy(foot.y)}
                  stroke={ACCENT} strokeWidth={6} strokeOpacity={0.32} strokeLinecap="round" />,
              )
              // 从 a 的端点垂直落到落脚点的虚线
              elems.push(
                <line key="drop" x1={sx(a.x)} y1={sy(a.y)} x2={sx(foot.x)} y2={sy(foot.y)}
                  stroke={ACCENT} strokeWidth={1.5} strokeDasharray="4 4" strokeOpacity={0.8} />,
              )
              // 落脚点的小圆点
              elems.push(
                <circle key="foot" cx={sx(foot.x)} cy={sy(foot.y)} r={3.5} fill={ACCENT} />,
              )
            }

            // —— 夹角圆弧（靠近原点）——
            if (na > 1e-6 && nb > 1e-6) {
              const R = Math.min(1.1, minLen * 0.55) // 数学单位下的半径
              const angA = Math.atan2(a.y, a.x)
              const angB = Math.atan2(b.y, b.x)
              let delta = angB - angA
              while (delta <= -Math.PI) delta += 2 * Math.PI
              while (delta > Math.PI) delta -= 2 * Math.PI
              // 从 +CCW（数学）方向较"靠前"的那根开始扫，保证走内角
              const [s, e] = delta >= 0 ? [angA, angB] : [angB, angA]
              const p1x = sx(R * Math.cos(s)), p1y = sy(R * Math.sin(s))
              const p2x = sx(R * Math.cos(e)), p2y = sy(R * Math.sin(e))
              const rpx = R * (sx(1) - sx(0))
              // y 轴翻转：数学逆时针在屏幕上是顺时针 → sweep-flag = 1
              elems.push(
                <path key="arc" d={`M ${p1x} ${p1y} A ${rpx} ${rpx} 0 0 1 ${p2x} ${p2y}`}
                  fill="none" stroke={INK} strokeWidth={1.6} />,
              )
              // 角度标签放在角平分线方向
              const midAng = s + (delta >= 0 ? delta : -delta) / 2
              const lr = R + 0.45
              elems.push(
                <text key="arc-t" x={sx(lr * Math.cos(midAng))} y={sy(lr * Math.sin(midAng))}
                  fill={INK} fontSize={13} textAnchor="middle" dominantBaseline="middle">
                  {fmt(degrees(theta), 0)}°
                </text>,
              )
            }

            return <>{elems}</>
          }}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>a</span><code>({fmt(a.x)}, {fmt(a.y)})</code></div>
          <div className="vrow"><span>b</span><code>({fmt(b.x)}, {fmt(b.y)})</code></div>
          <div className="vrow"><span>内积 a·b</span><code>{fmt(a.x)}×{fmt(b.x)} + {fmt(a.y)}×{fmt(b.y)} = {fmt(d)}</code></div>
          <div className="vrow"><span>长度 |a|, |b|</span><code>{fmt(na)} , {fmt(nb)}</code></div>
          <div className="vrow"><span>夹角 θ</span><code>{fmt(degrees(theta), 1)}°</code></div>
          <div className="vrow"><span>cosθ（相似度）</span><code>{fmt(cos)}</code></div>
          <div className="vrow"><span>a 在 b 上投影</span><code>{fmt(sProj)}</code></div>
          <div className="vrow"><span>验证恒等式</span><code>|b|×投影 = {fmt(nb)}×{fmt(sProj)} = {fmt(d)}</code></div>

          <div
            style={{
              marginTop: 12, padding: '10px 12px', borderRadius: 8,
              background: verdict.bg, color: verdict.color,
              fontWeight: 600, fontSize: 14, lineHeight: 1.4,
              borderLeft: `3px solid ${verdict.color}`,
            }}
          >
            {verdict.text}
          </div>

          <p className="vhint">
            拖动蓝色 a 与青色 b。橙色粗线是 a 落在 b 上的<strong>投影</strong>，
            橙色虚线是从 a 端点垂下来的高度；灰色圆弧是它们的夹角。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          为什么 <code>a·b</code> 能当"相似度"？把它拆成 <code>|b| × (a 在 b 上的投影长度)</code>：
          a 越是顺着 b 的方向，投影越长，内积越大；当 a 垂直于 b 时投影为 0，内积也为 0；
          a 掉头反向时投影为负，内积变负。<strong>面板里那行"验证恒等式"两边永远相等</strong>，
          你拖出任意角度都成立——这就是内积的几何灵魂。
        </p>
      </section>

      <Bridge>
        <p>
          <strong>注意力的第一步，就是内积。</strong>每个 token 都带一个 query 向量 <code>q</code> 和
          一个 key 向量 <code>k</code>，注意力分数 <code>score = q·k</code>——内积越大，代表这两个 token
          "越相关"。这些分数再过一道 <code>softmax</code>，就变成了真正的注意力权重。
        </p>
        <p>
          所以"<strong>内积 = 相似度</strong>"就是整个注意力机制的心脏。
          （顺带预告：实际用的是"缩放点积" <code>q·k/√d</code>，为什么要除以 <code>√d</code>，<ChRef slug="norms" />会讲。）
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：一个数，衡量"像不像"</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="dot.ts" />
      </section>
    </ChapterShell>
  )
}
