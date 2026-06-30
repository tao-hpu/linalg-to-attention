import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { apply, type Mat2, type Vec2 } from '../linalg'
import { TransformPanel } from '../TransformPanel'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// F 字形顶点（与 TransformPanel 一致，平移到原点附近）
const F_RAW: Vec2[] = [
  [0, 0], [0, 3], [1.8, 3], [1.8, 2.4], [0.6, 2.4],
  [0.6, 1.8], [1.5, 1.8], [1.5, 1.2], [0.6, 1.2], [0.6, 0],
]
const F_SHAPE: Vec2[] = F_RAW.map(([x, y]): Vec2 => [x - 0.9, y - 1.5])

// 画布参数
const SIZE = 360
const RANGE = 3
const UNIT = SIZE / 2 / RANGE  // 1 个数学单位 = 60px
const CX = SIZE / 2
const CY = SIZE / 2

// 坐标互转（y 轴朝上 → SVG y 朝下）
const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT
const toMx = (px: number) => (px - CX) / UNIT
const toMy = (py: number) => (CY - py) / UNIT

// 吸附到 0.5 格，并限制在 [-2.5, 2.5]
const snapClamp = (v: number): number =>
  Math.round(Math.min(2.5, Math.max(-2.5, v)) * 2) / 2

// 品牌色
const RUST = '#c75b39'   // î，第一列
const IKB = '#002fa7'    // ĵ，第二列

// 预设
const PRESETS: Array<{ name: string; i: Vec2; j: Vec2 }> = [
  { name: '恒等', i: [1, 0], j: [0, 1] },
  { name: '旋转 90°', i: [0, 1], j: [-1, 0] },
  { name: '切变', i: [1, 0], j: [1, 1] },
]

// 变形后的格线（对所有格点施加变换 M）
function buildGridLines(M: Mat2) {
  const lines = []
  for (let k = -RANGE; k <= RANGE; k++) {
    const isAxis = k === 0
    const stroke = isAxis ? '#9aa5b0' : '#e6e8ea'
    const sw = isAxis ? 1.5 : 1
    // 竖线 x=k
    const v1 = apply(M, [k, -RANGE])
    const v2 = apply(M, [k, RANGE])
    // 横线 y=k
    const h1 = apply(M, [-RANGE, k])
    const h2 = apply(M, [RANGE, k])
    lines.push(
      <line key={`v${k}`}
        x1={toSx(v1[0])} y1={toSy(v1[1])}
        x2={toSx(v2[0])} y2={toSy(v2[1])}
        stroke={stroke} strokeWidth={sw}
      />,
      <line key={`h${k}`}
        x1={toSx(h1[0])} y1={toSy(h1[1])}
        x2={toSx(h2[0])} y2={toSy(h2[1])}
        stroke={stroke} strokeWidth={sw}
      />
    )
  }
  return lines
}

// 带箭头的向量线（从原点到 v）
function BasisArrow({ v, color }: { v: Vec2; color: string }) {
  const [x, y] = v
  if (Math.hypot(x, y) < 1e-6) return null
  const tipX = toSx(x)
  const tipY = toSy(y)
  const orgX = toSx(0)
  const orgY = toSy(0)
  const ang = Math.atan2(orgY - tipY, tipX - orgX)
  const ah = 10
  const aw = 5
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)
  return (
    <g>
      <line
        x1={orgX} y1={orgY} x2={tipX} y2={tipY}
        stroke={color} strokeWidth={2.5} strokeLinecap="round"
      />
      <polygon
        points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`}
        fill={color}
      />
    </g>
  )
}

// 可拖动的基向量 SVG 画布
function BasisCanvas({
  iHat,
  jHat,
  onIHat,
  onJHat,
}: {
  iHat: Vec2
  jHat: Vec2
  onIHat: (v: Vec2) => void
  onJHat: (v: Vec2) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  // useRef 避免拖动时频繁 re-render
  const dragging = useRef<'i' | 'j' | null>(null)

  const M: Mat2 = [iHat[0], jHat[0], iHat[1], jHat[1]]
  const fPts = F_SHAPE.map(p => apply(M, p))
  const fPath = fPts.map(([x, y]) => `${toSx(x)},${toSy(y)}`).join(' ')

  // 手柄标签偏移：如果靠近右边缘则放到左侧
  const labelOffsetX = (v: Vec2) => (toSx(v[0]) > SIZE - 30 ? -28 : 13)

  return (
    <svg
      ref={svgRef}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      style={{ display: 'block', touchAction: 'none', cursor: 'default' }}
      onPointerDown={(e) => {
        const handle = (e.target as Element).getAttribute('data-handle')
        if (handle === 'i' || handle === 'j') {
          e.currentTarget.setPointerCapture(e.pointerId)
          dragging.current = handle
        }
      }}
      onPointerMove={(e) => {
        if (!dragging.current || !svgRef.current) return
        const rect = svgRef.current.getBoundingClientRect()
        // 移动端 SVG 会被 CSS 等比缩放（rect.width < SIZE），先把屏幕像素换算回
        // viewBox 内坐标，否则除以写死的 UNIT 会让拖动偏移。
        const sclX = SIZE / rect.width
        const sclY = SIZE / rect.height
        const mx = snapClamp(toMx((e.clientX - rect.left) * sclX))
        const my = snapClamp(toMy((e.clientY - rect.top) * sclY))
        if (dragging.current === 'i') onIHat([mx, my])
        else onJHat([mx, my])
      }}
      onPointerUp={() => { dragging.current = null }}
      onPointerCancel={() => { dragging.current = null }}
    >
      <defs>
        <clipPath id="mat-canvas-clip">
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
      </defs>

      {/* 变形格线 */}
      <g clipPath="url(#mat-canvas-clip)">
        {buildGridLines(M)}
      </g>

      {/* 变形的 F 字形 */}
      <polygon
        clipPath="url(#mat-canvas-clip)"
        points={fPath}
        fill="rgba(0,47,167,0.10)"
        stroke={IKB}
        strokeWidth={1.5}
        strokeLinejoin="round"
      />

      {/* 基向量箭头 */}
      <g clipPath="url(#mat-canvas-clip)">
        <BasisArrow v={iHat} color={RUST} />
        <BasisArrow v={jHat} color={IKB} />
      </g>

      {/* î 手柄 */}
      <circle
        cx={toSx(iHat[0])}
        cy={toSy(iHat[1])}
        r={9}
        fill={RUST}
        stroke="white"
        strokeWidth={2.5}
        data-handle="i"
        style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(iHat[0]) + labelOffsetX(iHat)}
        y={toSy(iHat[1]) + 5}
        fill={RUST}
        fontSize={14}
        fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        î
      </text>

      {/* ĵ 手柄 */}
      <circle
        cx={toSx(jHat[0])}
        cy={toSy(jHat[1])}
        r={9}
        fill={IKB}
        stroke="white"
        strokeWidth={2.5}
        data-handle="j"
        style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(jHat[0]) + labelOffsetX(jHat)}
        y={toSy(jHat[1]) + 5}
        fill={IKB}
        fontSize={14}
        fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        ĵ
      </text>
    </svg>
  )
}

// 带颜色编码列的矩阵展示
function ColoredMatrix({ iHat, jHat }: { iHat: Vec2; jHat: Vec2 }) {
  const fmt = (n: number): string => {
    const r = Math.round(n * 100) / 100
    return (Object.is(r, -0) ? 0 : r).toFixed(2)
  }
  return (
    <div className="matrix">
      <span className="bracket">[</span>
      <span className="matrix-rows">
        <span>
          <span style={{ color: RUST, fontWeight: 700 }}>{fmt(iHat[0])}</span>
          {'  '}
          <span style={{ color: IKB, fontWeight: 700 }}>{fmt(jHat[0])}</span>
        </span>
        <span>
          <span style={{ color: RUST, fontWeight: 700 }}>{fmt(iHat[1])}</span>
          {'  '}
          <span style={{ color: IKB, fontWeight: 700 }}>{fmt(jHat[1])}</span>
        </span>
      </span>
      <span className="bracket">]</span>
    </div>
  )
}

const SNIPPET = `// 「矩阵」只是把「î 落点」和「ĵ 落点」两列拼在一起
// Row-major Mat2 = [a, b, c, d]  →  [[a, b], [c, d]]
//   列 1 (a, c) = î 落点 = iHat   （第一列，rust 色）
//   列 2 (b, d) = ĵ 落点 = jHat   （第二列，blue 色）

const M: Mat2 = [iHat[0], jHat[0], iHat[1], jHat[1]]
//               ^a        ^b        ^c        ^d

// 任意向量 v = x·î + y·ĵ，线性性保证：
//   M·v = x·(M·î) + y·(M·ĵ)
//       = x·col₁(M) + y·col₂(M)

function apply(M: Mat2, v: Vec2): Vec2 {
  const [a, b, c, d] = M
  const [x, y] = v
  return [a * x + b * y, c * x + d * y]
  //      ───────────── x乘第一列 + y乘第二列
}

// PyTorch 等价写法：
// output = input @ W.T   （W shape = [out_features, in_features]）
// W 的每一列 = 对应输入基方向的落点`

export function MatrixAsTransform() {
  const [iHat, setIHat] = useState<Vec2>([1, 0])
  const [jHat, setJHat] = useState<Vec2>([0, 1])

  const M: Mat2 = [iHat[0], jHat[0], iHat[1], jHat[1]]

  const me = findChapter('matrix-as-transform')!
  const { prev, next } = neighbors('matrix-as-transform')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第二部分 · 矩阵：一个动作
        </div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>
          矩阵是变换
          <span className="zh-sub">矩阵到底"做"了什么？</span>
        </h1>
        <p className="lede">
          2×2 矩阵不是一张枯燥的"数表"——它是一次把整个平面
          <strong>拉伸、旋转或切变</strong>的动作。
          关键在于：矩阵的
          <span style={{ color: RUST }}>第一列</span>就是 î=(1,0) 的落点，
          <span style={{ color: IKB }}>第二列</span>就是 ĵ=(0,1) 的落点。
          知道这两个落点，<strong>平面上每一个向量的去向就全定了</strong>——因为任何向量都是
          î 和 ĵ 的线性组合。下面亲手拖动两根箭头，看"数表"怎么从中长出来。
        </p>
      </header>

      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>点击快速切换</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map(p => (
              <button
                key={p.name}
                onClick={() => { setIHat(p.i); setJHat(p.j) }}
                style={{
                  padding: '0.35rem 0.9rem',
                  border: `1.5px solid ${IKB}`,
                  borderRadius: 4,
                  background: 'white',
                  color: IKB,
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

      <section
        className="stage"
        style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}
      >
        <div style={{
          border: '1.5px solid #e6e8ea',
          borderRadius: 8,
          overflow: 'hidden',
          background: '#fff',
          boxShadow: '0 1px 6px rgba(0,0,0,0.06)',
        }}>
          <BasisCanvas iHat={iHat} jHat={jHat} onIHat={setIHat} onJHat={setJHat} />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0 }}>
          拖动{' '}
          <span style={{ color: RUST, fontWeight: 700 }}>● î</span>（列 1）和{' '}
          <span style={{ color: IKB, fontWeight: 700 }}>● ĵ</span>（列 2）塑造变换，格线和 F 字实时随之变形
        </p>
      </section>

      <section className="readouts">
        <ColoredMatrix iHat={iHat} jHat={jHat} />
        <div style={{
          display: 'flex',
          gap: '1.5rem',
          flexWrap: 'wrap',
          marginTop: '1rem',
          justifyContent: 'center',
        }}>
          <TransformPanel M={M} label="你的变换" sublabel="实时更新" active />
          <TransformPanel M={[1, 0, 0, 1]} label="恒等（对比）" sublabel="î ĵ 未动" />
        </div>
      </section>

      <section className="verdict verdict--eq">
        <p>
          <strong>「列 = 基向量的落点」——这就是全部秘密。</strong>
          那四个数 a、b、c、d，排列为
          <span style={{ color: RUST }}> 第一列 (a, c) = î 的落点</span>、
          <span style={{ color: IKB }}> 第二列 (b, d) = ĵ 的落点</span>。
          任意向量 <em>v = x·î + y·ĵ</em>，变换后恰好是{' '}
          <em>x·col₁ + y·col₂</em>——正是 <code>apply(M, v)</code> 一行算完的事。
          "数表"不过是把两个落点并排拼在了一起。
        </p>
      </section>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            一个 <code>nn.Linear(in_features, out_features)</code>，权重矩阵{' '}
            <code>W</code>（shape: [out, in]）就是一次这样的线性变换：
            把输入空间的每个基方向送到输出空间的某处。
            <code>W</code> 的每一列告诉你一个输入基方向的落点——和你刚才拖 î、ĵ 是同一件事，只是维度更高。
          </p>
          <p>
            Transformer 里的 Q/K/V 投影 <code>W_Q</code>、<code>W_K</code>、<code>W_V</code>，
            每一个都是这样的变换，分别把词向量送到"查询空间"、"键空间"、"值空间"。
            看懂这一页 = 看懂"线性层在几何上做了什么"。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：从基向量落点构造矩阵</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="matrix-as-transform.ts" />
      </section>

      <nav className="pager">
        {prev ? (
          <Link
            className="pager-link prev"
            to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}
          >
            <span className="pager-dir">← 上一章</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link
            className="pager-link next"
            to={next.status === 'live' ? `/ch/${next.slug}` : '/'}
          >
            <span className="pager-dir">下一章 →</span>
            <span className="pager-title">
              {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
            </span>
          </Link>
        ) : (
          <span />
        )}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
