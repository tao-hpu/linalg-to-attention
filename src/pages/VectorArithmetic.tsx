import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { add, scale, norm, fmt } from '../vec'

const IKB = '#002fa7'    // 向量 a
const TEAL = '#00b3a4'   // 向量 b
const PLUM = '#7a1fa2'   // 和 a+b / 缩放 k·a

const SNIPPET = `// 向量的加减与缩放：逐分量做就行，不难
const add   = (a, b) => ({ x: a.x + b.x, y: a.y + b.y })
const sub   = (a, b) => ({ x: a.x - b.x, y: a.y - b.y })
const scale = (a, k) => ({ x: a.x * k, y: a.y * k })

// 几何直觉：
//   a + b  →  把 b "接"到 a 的尖端（首尾相接），和就是从原点到终点的箭头
//   k · a  →  把 a 拉长/压短 k 倍，k<0 时反向

// 真正神奇的是：在词向量空间里，这套算术是「有语义」的——
//   king − man + woman ≈ queen
// 减掉"男性"、加上"女性"，方向就从"国王"滑向了"女王"。
// 向量加减，捕捉的是「类比」。`

export function VectorArithmetic() {
  const [a, setA] = useState({ x: 3, y: 1 })
  const [b, setB] = useState({ x: 1, y: 2.5 })
  const [k, setK] = useState(1.5)

  const sum = add(a, b)
  const ka = scale(a, k)
  const sumLen = norm(sum)

  return (
    <ChapterShell
      slug="vector-arithmetic"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          有了向量这串数字，下一步就是<strong>对它做运算</strong>。
          加法、减法、缩放——三件最朴素的事，却撑起了后面所有花样。
          加法是"首尾相接"，缩放是"拉长压短"。
          下面两个箭头都能<strong>拖动</strong>，滑块还能<strong>缩放</strong> a，
          边拖边看"为什么加法长这样"。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[
            { id: 'a', x: a.x, y: a.y, color: IKB, label: 'a', draggable: true },
            { id: 'b', x: b.x, y: b.y, color: TEAL, label: 'b', draggable: true },
            // k·a：用虚线+第三色画出缩放结果，和 a 同向（或反向）
            { id: 'ka', x: ka.x, y: ka.y, color: PLUM, label: 'k·a', dashed: true, width: 2 },
            // a+b：实线第三色，从原点到终点
            { id: 'sum', x: sum.x, y: sum.y, color: PLUM, label: 'a+b', width: 3 },
          ]}
          onDrag={(id, x, y) => {
            if (id === 'a') setA({ x, y })
            else if (id === 'b') setB({ x, y })
          }}
          size={380}
          range={5}
        >
          {({ sx, sy }) => (
            <>
              {/* 首尾相接：把 b 平移到 a 的尖端，虚线画出这条"搬过去的 b" */}
              <line
                x1={sx(a.x)} y1={sy(a.y)}
                x2={sx(sum.x)} y2={sy(sum.y)}
                stroke={TEAL} strokeWidth={2} strokeDasharray="5 5" opacity={0.7}
              />
              {/* 平行四边形的另外一条边：把 a 平移到 b 的尖端 */}
              <line
                x1={sx(b.x)} y1={sy(b.y)}
                x2={sx(sum.x)} y2={sy(sum.y)}
                stroke={IKB} strokeWidth={2} strokeDasharray="5 5" opacity={0.4}
              />
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>a 坐标</span><code>({fmt(a.x)}, {fmt(a.y)})</code></div>
          <div className="vrow"><span>b 坐标</span><code>({fmt(b.x)}, {fmt(b.y)})</code></div>
          <div className="vrow"><span>a + b 坐标</span><code>({fmt(sum.x)}, {fmt(sum.y)})</code></div>
          <div className="vrow"><span>|a + b|</span><code>{fmt(sumLen)}</code></div>

          <div style={{ margin: '14px 0 4px', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 13, color: '#5b6168', whiteSpace: 'nowrap' }}>k = {fmt(k, 1)}</span>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.1}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              style={{ flex: 1, accentColor: PLUM }}
            />
          </div>
          <div className="vrow"><span>k · a 坐标</span><code>({fmt(ka.x)}, {fmt(ka.y)})</code></div>

          <p className="vhint">
            蓝色 = a，青色 = b。紫色实线 a+b 是"首尾相接"的终点；
            两条虚线补成平行四边形。紫色虚线 k·a 是把 a 缩放后的样子，
            拉动滑块看它伸缩——k 为负时会反向。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          <strong>语义算术</strong>：在词向量空间里，加减法是<strong>有意义</strong>的。
          最经典的例子是 <code>king − man + woman ≈ queen</code>——
          从"国王"减去"男性"这个方向、再加上"女性"，落点恰好在"女王"附近。
          向量加减捕捉的，正是人类说的"<strong>类比</strong>"：A 之于 B，如同 C 之于谁？
          这也是第 01、02 节里 <code>embedding</code> 之所以值得做成向量的真正回报——
          一旦词成了向量，方向和位移就开始携带语义。
        </p>
      </section>

      <Bridge>
        <p>
          Transformer 里有一条贯穿始终的主干，叫<strong>残差流（residual stream）</strong>。
          它的本质，就是一路在做<strong>向量加法</strong>：每一层把自己算出的新信息，
          <code>加</code>回主干向量上——<code>x = x + 层的输出</code>。
        </p>
        <p>
          所以信息不是被"替换"，而是被"累积"。看懂了这一节的 <code>a + b</code> 首尾相接，
          你其实就看懂了残差流为什么能<strong>层层叠加、不断累积信息</strong>，
          而梯度又为什么能顺着这条加法主干一路传回去。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：加、减、缩放与语义类比</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="vec-arithmetic.ts" />
      </section>
    </ChapterShell>
  )
}
