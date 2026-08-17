import { useState, useRef } from 'react'
import { apply, type Mat2, type Vec2 } from '../linalg'
import { ChRef } from '../components/ChRef'
import { ChapterShell } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'

// ── Canvas constants ──────────────────────────────────────────────────
const SIZE = 360
const RANGE = 3
const UNIT = SIZE / 2 / RANGE   // 60 px per math unit
const CX = SIZE / 2
const CY = SIZE / 2

const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT
const toMx = (px: number) => (px - CX) / UNIT
const toMy = (py: number) => (CY - py) / UNIT

// Snap to nearest 0.5, clamp to [-2.5, 2.5]
const snapClamp = (v: number): number =>
  Math.round(Math.min(2.5, Math.max(-2.5, v)) * 2) / 2

// ── Brand colors ──────────────────────────────────────────────────────
const RUST     = '#c75b39'   // î  (column 1)
const IKB      = '#002fa7'   // ĵ  (column 2)
const FLIP_RED = '#c0392b'   // flipped / singular

// |det| below this → treat as singular
const SINGULAR_EPS = 0.15

// ── Presets ───────────────────────────────────────────────────────────
const PRESETS: Array<{ name: string; i: Vec2; j: Vec2 }> = [
  { name: '恒等',    i: [1, 0], j: [0,  1] },  // det = 1
  { name: '旋转 90°', i: [0, 1], j: [-1, 0] }, // det = 1
  { name: '拉伸 2×', i: [2, 0], j: [0,  1] },  // det = 2
  { name: '压扁',    i: [1, 0], j: [2,  0] },  // det = 0
]

// ── Code snippet ──────────────────────────────────────────────────────
const SNIPPET = `// det(M) = a·d − b·c，M = [a, b, c, d]（行主序 2×2）
// 几何含义：î 和 ĵ 张成的平行四边形的「带符号面积」

function det(M: Mat2): number {
  const [a, b, c, d] = M
  return a * d - b * c           // ad − bc
}

// 带符号面积 = det(M)，真实面积 = |det(M)|
// det > 0 → 朝向不变（右手系 → 右手系）
// det < 0 → 朝向翻转（平面被镜像）
// det = 0 → singular：rank < 2，变换不可逆

const area = Math.abs(det(M))    // 平行四边形面积
const singular = det(M) === 0   // 信息被销毁，不可逆`

// ── Deformed grid ─────────────────────────────────────────────────────
function buildGridLines(M: Mat2) {
  const lines = []
  for (let k = -RANGE; k <= RANGE; k++) {
    const isAxis = k === 0
    const stroke = isAxis ? '#9aa5b0' : '#e6e8ea'
    const sw = isAxis ? 1.5 : 1
    const v1 = apply(M, [k, -RANGE])
    const v2 = apply(M, [k,  RANGE])
    const h1 = apply(M, [-RANGE, k])
    const h2 = apply(M, [ RANGE, k])
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
      />,
    )
  }
  return lines
}

// ── Arrow from origin to v ────────────────────────────────────────────
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

// ── Draggable basis-vector canvas ─────────────────────────────────────
function DetCanvas({
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
  const dragging = useRef<'i' | 'j' | null>(null)

  const M: Mat2 = [iHat[0], jHat[0], iHat[1], jHat[1]]

  // det = ad − bc  (a=iHat[0], b=jHat[0], c=iHat[1], d=jHat[1])
  const det = iHat[0] * jHat[1] - jHat[0] * iHat[1]
  const absDet = Math.abs(det)
  const isSingular = absDet < SINGULAR_EPS
  const isFlipped  = det < -SINGULAR_EPS

  // Parallelogram: origin → î → î+ĵ → ĵ
  const p0x = toSx(0),                          p0y = toSy(0)
  const p1x = toSx(iHat[0]),                    p1y = toSy(iHat[1])
  const p2x = toSx(iHat[0] + jHat[0]),          p2y = toSy(iHat[1] + jHat[1])
  const p3x = toSx(jHat[0]),                    p3y = toSy(jHat[1])
  const paraPoints = `${p0x},${p0y} ${p1x},${p1y} ${p2x},${p2y} ${p3x},${p3y}`

  const paraFill = isSingular
    ? 'rgba(192,57,43,0.15)'
    : isFlipped
      ? 'rgba(192,57,43,0.20)'
      : 'rgba(0,47,167,0.13)'
  const paraStroke = isSingular || isFlipped ? FLIP_RED : IKB

  // Handle label: push left if near the right edge
  const labelOffX = (v: Vec2) => (toSx(v[0]) > SIZE - 30 ? -28 : 13)

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
        if (rect.width === 0 || rect.height === 0) return
        // 窄屏下 .page svg { max-width:100% } 会把 SVG 等比缩小，
        // 先把屏幕像素换算回 viewBox 内坐标，否则除以写死的 UNIT 会让拖动偏移。
        const px = (e.clientX - rect.left) * (SIZE / rect.width)
        const py = (e.clientY - rect.top) * (SIZE / rect.height)
        const mx = snapClamp(toMx(px))
        const my = snapClamp(toMy(py))
        if (dragging.current === 'i') onIHat([mx, my])
        else onJHat([mx, my])
      }}
      onPointerUp={() => { dragging.current = null }}
      onPointerCancel={() => { dragging.current = null }}
    >
      <defs>
        <clipPath id="det-canvas-clip">
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
      </defs>

      {/* Deformed grid — shows how the whole plane transforms */}
      <g clipPath="url(#det-canvas-clip)">
        {buildGridLines(M)}
      </g>

      {/* Parallelogram spanned by î and ĵ: area = |det| */}
      <polygon
        clipPath="url(#det-canvas-clip)"
        points={paraPoints}
        fill={paraFill}
        stroke={paraStroke}
        strokeWidth={2}
        strokeLinejoin="round"
      />

      {/* Basis vector arrows */}
      <g clipPath="url(#det-canvas-clip)">
        <BasisArrow v={iHat} color={RUST} />
        <BasisArrow v={jHat} color={IKB}  />
      </g>

      {/* î handle */}
      <circle
        cx={toSx(iHat[0])} cy={toSy(iHat[1])}
        r={9} fill={RUST} stroke="white" strokeWidth={2.5}
        data-handle="i" style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(iHat[0]) + labelOffX(iHat)}
        y={toSy(iHat[1]) + 5}
        fill={RUST} fontSize={14} fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >î</text>

      {/* ĵ handle */}
      <circle
        cx={toSx(jHat[0])} cy={toSy(jHat[1])}
        r={9} fill={IKB} stroke="white" strokeWidth={2.5}
        data-handle="j" style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(jHat[0]) + labelOffX(jHat)}
        y={toSy(jHat[1]) + 5}
        fill={IKB} fontSize={14} fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >ĵ</text>
    </svg>
  )
}

// ── Number formatter (2 dp, fix -0) ──────────────────────────────────
function fmt(n: number): string {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── Matrix with column-colored entries ────────────────────────────────
function ColoredMatrix({ iHat, jHat }: { iHat: Vec2; jHat: Vec2 }) {
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

// ── Page component ────────────────────────────────────────────────────
export function Determinant() {
  const [iHat, setIHat] = useState<Vec2>([1, 0])
  const [jHat, setJHat] = useState<Vec2>([0, 1])

  // M = [[a, b], [c, d]]  (row-major)
  const a = iHat[0], b = jHat[0], c = iHat[1], d = jHat[1]
  const det = a * d - b * c
  const absDet = Math.abs(det)
  const isSingular = absDet < SINGULAR_EPS
  const isFlipped  = det < -SINGULAR_EPS

  const verdictClass = isSingular || isFlipped ? 'verdict--neq' : 'verdict--eq'

  const areaDesc = absDet >= 1
    ? `放大 ${absDet.toFixed(2)} 倍`
    : `缩小到 ${absDet.toFixed(2)} 倍`

  const verdictContent = isSingular ? (
    <p>
      <strong>压扁成一条线 ——</strong>{' '}
      det ≈ 0，整个平面塌缩进更低维的子空间。
      变换 <strong>singular</strong>（奇异），not invertible（不可逆），信息永久丢失。
      你没法从输出里还原输入。
    </p>
  ) : isFlipped ? (
    <p>
      <strong>翻转了！朝向反了。</strong>{' '}
      det = {fmt(det)}（负数），面积变为{absDet.toFixed(2)} 倍，
      但平面的 <strong>orientation（朝向）</strong>反了——
      就像把纸翻到背面，右手系变成了左手系。变换可逆，但带镜像翻转。
    </p>
  ) : (
    <p>
      <strong>保持朝向，面积{areaDesc}。</strong>{' '}
      det = {fmt(det)}（正数），orientation 不变，变换 invertible（可逆）。
    </p>
  )

  return (
      <ChapterShell
        slug="determinant"
        part="第三部分 · 方阵的秘密"
        sub="一个变换会不会把信息「压扁」？"
        lede={
          <>
        determinant（行列式）只有一个含义：这个变换把面积
        <strong>缩放了多少倍</strong>？单位正方形（面积 1）经过变换，
        落成一个平行四边形——那个平行四边形的面积恰好等于{' '}
        <code>|det|</code>。符号告诉你 orientation：
        det &gt; 0 朝向不变，det &lt; 0 平面被
        <strong>翻转（mirrored）</strong>，det = 0 整个平面
        <strong>塌缩进一条线——信息丢了，not invertible</strong>。
        拖动下面两根箭头，亲眼看行列式怎么从形状里长出来。
          </>
        }
      >

      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>点击快速切换</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            {PRESETS.map((p) => (
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
          <DetCanvas iHat={iHat} jHat={jHat} onIHat={setIHat} onJHat={setJHat} />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0 }}>
          拖动{' '}
          <span style={{ color: RUST, fontWeight: 700 }}>● î</span>（列 1）和{' '}
          <span style={{ color: IKB, fontWeight: 700 }}>● ĵ</span>（列 2）——
          色块面积 = |det|，实时更新
        </p>
      </section>

      <section className="readouts">
        <ColoredMatrix iHat={iHat} jHat={jHat} />
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem 1.25rem',
          background: isSingular || isFlipped ? 'rgba(192,57,43,0.06)' : 'rgba(0,47,167,0.04)',
          borderLeft: `3px solid ${isSingular || isFlipped ? FLIP_RED : IKB}`,
          borderRadius: '0 4px 4px 0',
          fontFamily: 'monospace',
          fontSize: '0.95rem',
          lineHeight: 1.7,
        }}>
          <div>
            {'det = '}
            <span style={{ color: RUST }}>{fmt(a)}</span>
            {'·'}
            <span style={{ color: IKB }}>{fmt(d)}</span>
            {' − '}
            <span style={{ color: IKB }}>{fmt(b)}</span>
            {'·'}
            <span style={{ color: RUST }}>{fmt(c)}</span>
            {' = '}
            <strong style={{ color: isSingular || isFlipped ? FLIP_RED : IKB }}>
              {fmt(det)}
            </strong>
          </div>
          <div style={{ marginTop: '0.25rem', color: '#555', fontSize: '0.88rem' }}>
            {'面积 = |det| = '}
            <strong>{absDet.toFixed(2)}</strong>
          </div>
        </div>
      </section>

      <section className={`verdict ${verdictClass}`}>
        {verdictContent}
      </section>

      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            det = 0 ⟺ 变换把高维空间<strong>压进低维子空间</strong>、
            信息不可逆地丢失——这是「信息瓶颈」的几何根源，
            也是「为什么有些层不可逆」的真正答案。
            一个 <code>nn.Linear</code> 的权重矩阵如果 rank 不满（奇异），
            它就在销毁信息，你永远拿不回输入。
          </p>
          <p>
            后面三节都从这里长出来：<ChRef slug="rank" />「矩阵的秩」讲变换
            <em>留下了几维</em>；<ChRef slug="inverse" />「逆矩阵」讲 det ≠ 0 时才能还原；
            <ChRef slug="svd" /> SVD 把奇异值和行列式联系起来
            （<code>|det| = 所有奇异值之积</code>——奇异值恒非负，符号另由朝向决定，
            正好对应本节的 det &lt; 0）。行列式是整条线索的关键岔路口。
          </p>
        </div>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：det = ad − bc</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="determinant.ts" />
      </section>

      </ChapterShell>
  )
}
