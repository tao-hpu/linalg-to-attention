import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { norm, normalize, scale, cosineSimilarity, fmt } from '../vec'

const IKB = '#002fa7'
const TEAL = '#00b3a4'
const RUST = '#b23a00'

const SNIPPET = `// 范数 = 长度，来自勾股定理（L2 范数）
const norm = (a) => Math.hypot(a.x, a.y)   // √(x² + y²)

// 归一化 = 除以自己的长度，把向量缩到长度 1，只留方向
const normalize = (a) => {
  const n = norm(a)
  return { x: a.x / n, y: a.y / n }
}

const a = { x: 3, y: 2 }
const u = normalize(a)
norm(u)            // === 1.00（永远落在单位圆上）

// 余弦相似度 = 先各自归一化，再做内积
const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b))

// 关键性质：缩放不改变方向，所以 cosine 不变
cosine(a, scale(a, 5))   // === 1，长度变了，方向没变`

export function Norms() {
  const [a, setA] = useState({ x: 3, y: 2 })
  const [s, setS] = useState(1.5)

  const len = norm(a)
  const u = normalize(a)
  const scaled = scale(a, s)
  const cos = cosineSimilarity(a, scaled)

  return (
    <ChapterShell
      slug="norms"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          一个向量有<strong>方向</strong>，也有<strong>长度</strong>。长度就是它的<strong>范数</strong>——
          最常用的 L2 范数不过是勾股定理 <code>√(x² + y²)</code>。
          很多时候我们只关心"它指向哪"，不关心"它有多长"，这时就要<strong>归一化</strong>：
          把向量缩到长度 1，只保留方向。下面拖动蓝色箭头，看它的单位向量 â 始终钉在<strong>单位圆</strong>上。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas
          vectors={[
            { id: 'sa', x: scaled.x, y: scaled.y, color: RUST, label: 's·a', dashed: true, width: 2 },
            { id: 'a', x: a.x, y: a.y, color: IKB, label: 'a', draggable: true },
            { id: 'u', x: u.x, y: u.y, color: TEAL, label: 'â', width: 3 },
          ]}
          onDrag={(_id, x, y) => setA({ x, y })}
          size={380}
          range={5}
        >
          {({ sx, sy, unit }) => (
            <>
              {/* 单位圆：半径 = 1 个数学单位 */}
              <circle cx={sx(0)} cy={sy(0)} r={unit}
                fill="none" stroke={TEAL} strokeWidth={1.5}
                strokeDasharray="3 4" opacity={0.7} />
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow"><span>坐标</span><code>a = ({fmt(a.x)}, {fmt(a.y)})</code></div>
          <div className="vrow"><span>L2 范数 |a|</span><code>√({fmt(a.x)}² + {fmt(a.y)}²) = {fmt(len)}</code></div>
          <div className="vrow"><span>归一化 â</span><code>({fmt(u.x)}, {fmt(u.y)})</code></div>
          <div className="vrow"><span>验证 |â|</span><code>{fmt(norm(u))}</code></div>

          <hr style={{ border: 'none', borderTop: '1px solid #e3e6ea', margin: '14px 0' }} />

          <div className="vrow">
            <span>缩放 s = {fmt(s, 1)}</span>
            <input
              type="range" min={0.3} max={3} step={0.1} value={s}
              onChange={(e) => setS(Number(e.target.value))}
              style={{ flex: 1, accentColor: IKB, cursor: 'pointer' }}
            />
          </div>
          <div className="vrow"><span>|s·a|</span><code>{fmt(norm(scaled))}</code></div>
          <div className="vrow"><span>cos(a, s·a)</span><code style={{ color: TEAL }}>{fmt(cos)}</code></div>

          <p className="vhint">
            拖动蓝色箭头 a，绿色单位向量 â 永远落在虚线圆上。
            拉动滑块:橙色虚线箭头 <code>s·a</code> 沿同一方向伸缩，长度 <code>|s·a|</code> 在变，
            但 <code>cosine</code> 始终是 1.00——它只看方向，不看长度。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          归一化 = 把向量缩到长度 1，<strong>只保留方向</strong>。为什么要这么做？
          因为要"公平地"比较两个向量像不像，得先各自归一化再比，否则<strong>长的向量天然占便宜</strong>:
          一个又长又偏的箭头，内积也可能比一个又短又准的箭头大。先除掉长度，剩下的纯粹是方向之争——
          这正是上面 <code>cos(a, s·a) ≡ 1</code> 想说的：把 a 拉长 3 倍，方向没动，相似度就不该变。
        </p>
      </section>

      <Bridge>
        <p>
          这就是 <code>cosine 相似度</code>的来历——<strong>先归一化，再做内积</strong>。
          它把第 04 节的"内积 = 相似度"修正成了"只看方向的相似度"。
        </p>
        <p>
          还有一处更隐蔽的长度校正:Transformer 的"缩放点积注意力"把分数除以 <code>√d</code>。
          维度 d 越高，两个随机向量的内积<strong>波动幅度</strong>天然越大
          （期望约为 0，但标准差按 <code>√d</code> 增长），
          除以 <code>√d</code> 就是在抵消这件事，让 <code>softmax</code> 不至于一上来就饱和。
          把第 04 节的内积、和这一节的长度校正合起来，就是 <code>q·k/√d</code> 的全部含义。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码:范数、归一化与 cosine</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="norms.ts" />
      </section>
    </ChapterShell>
  )
}
