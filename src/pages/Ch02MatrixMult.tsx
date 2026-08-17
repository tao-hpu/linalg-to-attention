import { useState } from 'react'
import { Link } from 'react-router-dom'
import { multiply, format, nearlyEqual, transforms, type Mat2 } from '../linalg'
import { TransformPanel } from '../TransformPanel'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

type Kind = 'rotate' | 'scaleX' | 'shear' | 'projectX'

const KINDS: { kind: Kind; name: string; unit: string; min: number; max: number; step: number; def: number; hasParam: boolean }[] = [
  { kind: 'rotate', name: '旋转', unit: '°', min: -180, max: 180, step: 1, def: 50, hasParam: true },
  { kind: 'scaleX', name: '横向拉伸', unit: '×', min: 0.2, max: 3, step: 0.1, def: 2.2, hasParam: true },
  { kind: 'shear', name: '切变', unit: 'k', min: -2, max: 2, step: 0.1, def: 1, hasParam: true },
  { kind: 'projectX', name: '投影到 x 轴', unit: '', min: 0, max: 0, step: 1, def: 0, hasParam: false },
]

function buildMatrix(kind: Kind, param: number): Mat2 {
  switch (kind) {
    case 'rotate': return transforms.rotate(param)
    case 'scaleX': return transforms.scaleX(param)
    case 'shear': return transforms.shear(param)
    case 'projectX': return transforms.projectX()
  }
}

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

function TransformControl({ slot, kind, param, onKind, onParam }: {
  slot: string
  kind: Kind
  param: number
  onKind: (k: Kind) => void
  onParam: (n: number) => void
}) {
  const meta = KINDS.find((k) => k.kind === kind)!
  return (
    <div className="control">
      <div className="control-head">
        <span className="slot-tag">{slot}</span>
        <select value={kind} onChange={(e) => onKind(e.target.value as Kind)}>
          {KINDS.map((k) => <option key={k.kind} value={k.kind}>{k.name}</option>)}
        </select>
      </div>
      {meta.hasParam ? (
        <label className="slider-row">
          <input type="range" min={meta.min} max={meta.max} step={meta.step}
            value={param} onChange={(e) => onParam(Number(e.target.value))} />
          <span className="param-val">{param}{meta.unit}</span>
        </label>
      ) : (
        <p className="no-param">无参数 · 把整个平面压成一条线</p>
      )}
    </div>
  )
}

const SNIPPET = `// 行主序 2×2：[[a,b],[c,d]] -> [a,b,c,d]
function multiply(A: Mat2, B: Mat2): Mat2 {
  const [a, b, c, d] = A
  const [e, f, g, h] = B
  return [
    a * e + b * g,  a * f + b * h,
    c * e + d * g,  c * f + d * h,
  ]
}

const AB = multiply(A, B)  // 先做 B，再做 A
const BA = multiply(B, A)  // 先做 A，再做 B
// 一般情况下 AB ≠ BA
// 练习：取 A = 旋转 90°，B = 横向拉伸 2×，手算证明二者不等`

export function Ch02MatrixMult() {
  const [aKind, setAKind] = useState<Kind>('rotate')
  const [aParam, setAParam] = useState(50)
  const [bKind, setBKind] = useState<Kind>('scaleX')
  const [bParam, setBParam] = useState(2.2)

  const A = buildMatrix(aKind, aParam)
  const B = buildMatrix(bKind, bParam)
  const AB = multiply(A, B) // 先 B 后 A
  const BA = multiply(B, A) // 先 A 后 B
  const same = nearlyEqual(AB, BA)

  const setKind = (setK: (k: Kind) => void, setP: (n: number) => void) => (k: Kind) => {
    setK(k)
    setP(KINDS.find((x) => x.kind === k)!.def)
  }

  const aName = KINDS.find((k) => k.kind === aKind)!.name
  const bName = KINDS.find((k) => k.kind === bKind)!.name

  const me = findChapter('matrix-mult')!
  const { prev, next } = neighbors('matrix-mult')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb"><Link to="/">大纲</Link> · 第二部分 · 矩阵：一个动作</div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>矩阵乘法的几何<span className="zh-sub">为什么换了顺序结果就不一样？</span></h1>
        <p className="lede">
          矩阵不是一堆数字，而是一个<strong>动作</strong>——旋转、拉伸、切变、投影。
          两个矩阵相乘 <code>AB</code>，意思是<strong>先做 B、再做 A</strong>。
          既然是按顺序做的动作，换顺序当然可能得到完全不同的结果。下面亲手试一下。
        </p>
      </header>

      <section className="controls">
        <TransformControl slot="A" kind={aKind} param={aParam}
          onKind={setKind(setAKind, setAParam)} onParam={setAParam} />
        <TransformControl slot="B" kind={bKind} param={bParam}
          onKind={setKind(setBKind, setBParam)} onParam={setBParam} />
      </section>

      <section className="stage">
        <TransformPanel M={[1, 0, 0, 1]} label="原图" sublabel="还没动作之前" />
        <div className="arrow-sep">→</div>
        <TransformPanel M={AB} label="先 B 后 A" sublabel="= A·B" active />
        <TransformPanel M={BA} label="先 A 后 B" sublabel="= B·A" active />
      </section>

      <section className="readouts">
        <MatrixReadout M={A} name="A" />
        <MatrixReadout M={B} name="B" />
        <MatrixReadout M={AB} name="AB" />
        <MatrixReadout M={BA} name="BA" />
      </section>

      <section className={`verdict ${same ? 'verdict--eq' : 'verdict--neq'}`}>
        {same ? (
          <p><strong>这一组里 AB = BA。</strong> 你恰好选到了两个能交换的动作。
            换一组试试——大多数动作是<strong>不能</strong>交换的。</p>
        ) : (
          <p><strong>AB ≠ BA。</strong> 同样是「{aName}」和「{bName}」这两个动作，
            只是先后不同，<strong>F 落到了完全不同的位置</strong>。这就是矩阵乘法不可交换的全部秘密：
            它是动作的<em>复合</em>，先洗澡再穿衣，和先穿衣再洗澡，不是一回事。</p>
        )}
      </section>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            Transformer 的每一层里，一个词向量要先后乘上好几个矩阵：
            <code>W_Q</code>、<code>W_K</code>、<code>W_V</code> 投影出 Query / Key / Value，
            注意力加权后再乘 <code>W_O</code>，然后进 MLP 的 <code>W_1</code>、<code>W_2</code>……
          </p>
          <p>
            这些<strong>全是矩阵乘法，全都讲顺序</strong>。把 <code>W_Q</code> 和 <code>W_K</code> 的作用顺序对调，
            算出来的注意力分数就变了——和你刚才把 A、B 对调、F 跑到别处，是<strong>同一件事</strong>。
            看懂这一页，你就看懂了「线性层为什么不能随便换序」。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：AB 和 BA 怎么算的</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="multiply.ts" />
      </section>

      <nav className="pager">
        {prev
          ? <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
              <span className="pager-dir">← 上一节</span>
              <span className="pager-title">{prev.num} {prev.title}</span>
            </Link>
          : <span />}
        {next
          ? <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
              <span className="pager-dir">下一节 →</span>
              <span className="pager-title">{next.num} {next.title}{next.status !== 'live' && ' · 规划中'}</span>
            </Link>
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
