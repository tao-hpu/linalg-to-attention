import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { norm, fmt } from '../vec'

const IKB = '#002fa7'
const TEAL = '#00b3a4'

const SNIPPET = `// 一个 2D 向量就是两个数：沿 x 走多少、沿 y 走多少
const a = { x: 3, y: 2 }

// 它等于「基向量」的线性组合：
//   a = 3·î + 2·ĵ      （î=(1,0)，ĵ=(0,1)）

// 长度（范数）来自勾股定理：
const length = Math.hypot(a.x, a.y)   // √(3² + 2²) ≈ 3.61

// 一个词向量只是把这件事搬到几百上千维：
//   "猫" → [0.21, -0.07, 0.88, ... ]   （几百个数）
// 维度太多没法画，但「一串数 = 空间里的一个箭头」这件事不变`

export function Ch01Vectors() {
  const [p, setP] = useState({ x: 3, y: 2 })
  const len = norm(p)

  return (
    <ChapterShell
      slug="vectors"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          向量就是<strong>一串有序的数字</strong>——但别把它只看成数字，
          把它看成空间里的<strong>一个箭头</strong>：从原点出发，沿每个坐标轴各走一段。
          一个词之所以能被计算机"理解"，正是因为它被表示成了这样一串数字。
          下面这个箭头你可以<strong>直接拖动</strong>，看它的坐标和长度怎么变。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[{ id: 'a', x: p.x, y: p.y, color: IKB, label: 'a', draggable: true }]}
          onDrag={(_id, x, y) => setP({ x, y })}
          size={380}
          range={5}
        >
          {({ sx, sy }) => (
            <>
              {/* x、y 分量的虚线分解 */}
              <line x1={sx(0)} y1={sy(0)} x2={sx(p.x)} y2={sy(0)}
                stroke={TEAL} strokeWidth={2} strokeDasharray="4 4" />
              <line x1={sx(p.x)} y1={sy(0)} x2={sx(p.x)} y2={sy(p.y)}
                stroke={TEAL} strokeWidth={2} strokeDasharray="4 4" />
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>坐标</span><code>a = ({fmt(p.x)}, {fmt(p.y)})</code></div>
          <div className="vrow"><span>分量分解</span><code>{fmt(p.x)}·î + {fmt(p.y)}·ĵ</code></div>
          <div className="vrow"><span>长度 |a|</span><code>√({fmt(p.x)}² + {fmt(p.y)}²) = {fmt(len)}</code></div>
          <p className="vhint">拖动蓝色箭头的端点。绿色虚线是它在 x、y 轴上的分量。</p>
        </div>
      </section>

      <section className="note">
        <p>
          关键转变:<strong>别读这串数字,看这个箭头</strong>。`(3, 2)` 不是"三和二",
          是"往右 3、往上 2 的那个方向和距离"。一旦接受了这个视角,后面所有概念
          ——相似度、投影、矩阵变换——都会变成你<strong>看得见的几何动作</strong>,而不是符号操作。
        </p>
      </section>

      <Bridge>
        <p>
          大语言模型干的第一件事,就是把每个词(token)变成一个向量,叫 <code>embedding</code>。
          "猫"可能是 <code>[0.21, -0.07, 0.88, …]</code> 这样几百上千个数。
        </p>
        <p>
          维度太高没法画,但<strong>本质和这个 2D 箭头一模一样</strong>:一串数 = 高维空间里的一个点/箭头。
          意思相近的词,箭头方向也相近——这正是下一节"语义算术"和后面"内积=相似度"的全部地基。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码:向量、分量与长度</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="vector.ts" />
      </section>
    </ChapterShell>
  )
}
