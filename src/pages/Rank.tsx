import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { type Vec2 } from '../linalg'
import { CodeBlock } from '../components/CodeBlock'
import { neighbors, allChapters, findChapter } from '../chapters'

// ── Canvas geometry ──────────────────────────────────────────────────────────
const SIZE = 360
const RANGE = 3
const UNIT = SIZE / 2 / RANGE   // 1 math unit = 60 px
const CX = SIZE / 2
const CY = SIZE / 2

const toSx = (x: number) => CX + x * UNIT
const toSy = (y: number) => CY - y * UNIT
const toMx = (px: number) => (px - CX) / UNIT
const toMy = (py: number) => (CY - py) / UNIT

const snapClamp = (v: number): number =>
  Math.round(Math.min(2.5, Math.max(-2.5, v)) * 2) / 2

// ── Brand colors ─────────────────────────────────────────────────────────────
const RUST  = '#c75b39'
const IKB   = '#002fa7'
const GRID  = '#e6e8ea'
const WARN  = '#c0392b'
const GREEN = '#1a8a4a'
const EPS   = 1e-6

const fmt = (n: number): string => {
  const r = Math.round(n * 100) / 100
  return (Object.is(r, -0) ? 0 : r).toFixed(2)
}

// ── Math helpers ─────────────────────────────────────────────────────────────
function colDet(c1: Vec2, c2: Vec2): number {
  return c1[0] * c2[1] - c1[1] * c2[0]
}

function computeRank(c1: Vec2, c2: Vec2): 0 | 1 | 2 {
  const z1 = Math.hypot(c1[0], c1[1]) < EPS
  const z2 = Math.hypot(c2[0], c2[1]) < EPS
  if (z1 && z2) return 0
  if (Math.abs(colDet(c1, c2)) < EPS) return 1
  return 2
}

// ── Grid lines (static, no transform) ────────────────────────────────────────
function GridLines() {
  const lines = []
  for (let k = -RANGE; k <= RANGE; k++) {
    const isAxis = k === 0
    const stroke = isAxis ? '#9aa5b0' : GRID
    const sw = isAxis ? 1.5 : 1
    lines.push(
      <line key={`h${k}`} x1={0} y1={toSy(k)} x2={SIZE} y2={toSy(k)}
        stroke={stroke} strokeWidth={sw} />,
      <line key={`v${k}`} x1={toSx(k)} y1={0} x2={toSx(k)} y2={SIZE}
        stroke={stroke} strokeWidth={sw} />,
    )
  }
  return <>{lines}</>
}

// ── Arrow from origin to v ────────────────────────────────────────────────────
function Arrow({ v, color }: { v: Vec2; color: string }) {
  const [x, y] = v
  if (Math.hypot(x, y) < EPS) return null
  const tipX = toSx(x), tipY = toSy(y)
  const orgX = toSx(0), orgY = toSy(0)
  const ang = Math.atan2(orgY - tipY, tipX - orgX)
  const ah = 10, aw = 5
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)
  return (
    <g>
      <line x1={orgX} y1={orgY} x2={tipX} y2={tipY}
        stroke={color} strokeWidth={2.5} strokeLinecap="round" />
      <polygon points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`} fill={color} />
    </g>
  )
}

// ── Rank badge overlay (SVG) ──────────────────────────────────────────────────
function RankBadge({ rank }: { rank: 0 | 1 | 2 }) {
  const color = rank === 2 ? GREEN : WARN
  return (
    <g style={{ pointerEvents: 'none' }}>
      <rect x={SIZE - 84} y={8} width={74} height={26} rx={13} fill={color} />
      <text x={SIZE - 47} y={26} textAnchor="middle"
        fill="white" fontSize={13} fontWeight="bold"
        style={{ userSelect: 'none' }}>
        秩 = {rank}
      </text>
    </g>
  )
}

// ── Interactive canvas ────────────────────────────────────────────────────────
function RankCanvas({
  c1, c2, onC1, onC2,
}: {
  c1: Vec2; c2: Vec2
  onC1: (v: Vec2) => void
  onC2: (v: Vec2) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const dragging = useRef<'c1' | 'c2' | null>(null)

  const rank = computeRank(c1, c2)

  // Parallelogram span (rank 2): origin → c1 → c1+c2 → c2
  const px0 = toSx(0),              py0 = toSy(0)
  const px1 = toSx(c1[0]),          py1 = toSy(c1[1])
  const px2 = toSx(c1[0] + c2[0]),  py2 = toSy(c1[1] + c2[1])
  const px3 = toSx(c2[0]),          py3 = toSy(c2[1])
  const paraPoints = `${px0},${py0} ${px1},${py1} ${px2},${py2} ${px3},${py3}`

  // Rank-1 line: extend the nonzero direction ±5 units
  const dir: Vec2 = Math.hypot(c1[0], c1[1]) > EPS ? c1 : c2
  const dLen = Math.hypot(dir[0], dir[1])
  const ndx = dLen > EPS ? dir[0] / dLen : 1
  const ndy = dLen > EPS ? dir[1] / dLen : 0
  const T = 5
  const lx1 = toSx(-T * ndx), ly1 = toSy(-T * ndy)
  const lx2 = toSx( T * ndx), ly2 = toSy( T * ndy)

  // Label offset: push left when near right edge
  const labelOffX = (v: Vec2) => (toSx(v[0]) > SIZE - 30 ? -28 : 13)

  return (
    <svg
      ref={svgRef}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      style={{ display: 'block', touchAction: 'none', cursor: 'default' }}
      onPointerDown={(e) => {
        const h = (e.target as Element).getAttribute('data-handle')
        if (h === 'c1' || h === 'c2') {
          e.currentTarget.setPointerCapture(e.pointerId)
          dragging.current = h
        }
      }}
      onPointerMove={(e) => {
        if (!dragging.current || !svgRef.current) return
        const rect = svgRef.current.getBoundingClientRect()
        if (rect.width === 0 || rect.height === 0) return
        // 窄屏下 SVG 被 CSS 等比缩小，先换算回 viewBox 内坐标再除以 UNIT。
        const px = (e.clientX - rect.left) * (SIZE / rect.width)
        const py = (e.clientY - rect.top) * (SIZE / rect.height)
        const mx = snapClamp(toMx(px))
        const my = snapClamp(toMy(py))
        if (dragging.current === 'c1') onC1([mx, my])
        else onC2([mx, my])
      }}
      onPointerUp={() => { dragging.current = null }}
      onPointerCancel={() => { dragging.current = null }}
    >
      <defs>
        <clipPath id="rank-canvas-clip">
          <rect x={0} y={0} width={SIZE} height={SIZE} />
        </clipPath>
      </defs>

      {/* Static grid */}
      <g clipPath="url(#rank-canvas-clip)">
        <GridLines />
      </g>

      {/* Span visualization */}
      <g clipPath="url(#rank-canvas-clip)">
        {rank === 2 && (
          <polygon
            points={paraPoints}
            fill="rgba(0,47,167,0.11)"
            stroke={IKB}
            strokeWidth={1}
            strokeDasharray="5,3"
          />
        )}
        {rank === 1 && (
          <line
            x1={lx1} y1={ly1} x2={lx2} y2={ly2}
            stroke={WARN} strokeWidth={2} strokeDasharray="6,4" opacity={0.85}
          />
        )}
      </g>

      {/* Column arrows */}
      <g clipPath="url(#rank-canvas-clip)">
        <Arrow v={c1} color={RUST} />
        <Arrow v={c2} color={IKB} />
      </g>

      {/* Rank badge — always on top */}
      <RankBadge rank={rank} />

      {/* c1 drag handle */}
      <circle
        cx={toSx(c1[0])} cy={toSy(c1[1])} r={9}
        fill={RUST} stroke="white" strokeWidth={2.5}
        data-handle="c1"
        style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(c1[0]) + labelOffX(c1)} y={toSy(c1[1]) + 5}
        fill={RUST} fontSize={12} fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}>
        c₁
      </text>

      {/* c2 drag handle */}
      <circle
        cx={toSx(c2[0])} cy={toSy(c2[1])} r={9}
        fill={IKB} stroke="white" strokeWidth={2.5}
        data-handle="c2"
        style={{ cursor: 'grab' }}
      />
      <text
        x={toSx(c2[0]) + labelOffX(c2)} y={toSy(c2[1]) + 5}
        fill={IKB} fontSize={12} fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}>
        c₂
      </text>
    </svg>
  )
}

// ── Static outer-product panel ────────────────────────────────────────────────
function OuterProductPanel() {
  return (
    <section style={{
      margin: '2rem 0',
      padding: '1.25rem 1.5rem',
      border: `1.5px solid ${IKB}22`,
      borderLeft: `4px solid ${IKB}`,
      borderRadius: 6,
      background: '#f7f8fc',
    }}>
      <h2 className="sec-h" style={{ marginTop: 0 }}>rank-1 矩阵 = outer product</h2>
      <p style={{ fontSize: '0.92rem', color: '#444', marginBottom: '1rem' }}>
        任何 rank-1 矩阵都能拆成两个向量的外积 <code>u·vᵀ</code>——
        一个<strong>列方向</strong>（column space 的唯一基）乘一个<strong>行模式</strong>（row space 的唯一基）：
      </p>
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0.9rem',
        flexWrap: 'wrap', fontFamily: 'monospace', fontSize: '0.95rem',
      }}>
        {/* Left: 2×2 rank-1 matrix */}
        <div className="matrix">
          <span className="bracket">[</span>
          <span className="matrix-rows">
            <span>
              <span style={{ color: RUST, fontWeight: 700 }}>2</span>
              {'  '}
              <span style={{ color: IKB, fontWeight: 700 }}>4</span>
            </span>
            <span>
              <span style={{ color: RUST, fontWeight: 700 }}>1</span>
              {'  '}
              <span style={{ color: IKB, fontWeight: 700 }}>2</span>
            </span>
          </span>
          <span className="bracket">]</span>
        </div>
        <span style={{ fontSize: '1.1rem', color: '#555', fontFamily: 'inherit' }}>=</span>
        {/* u column vector */}
        <div className="matrix">
          <span className="bracket">[</span>
          <span className="matrix-rows">
            <span><span style={{ color: RUST, fontWeight: 700 }}>2</span></span>
            <span><span style={{ color: RUST, fontWeight: 700 }}>1</span></span>
          </span>
          <span className="bracket">]</span>
        </div>
        <span style={{ fontSize: '1.1rem', color: '#555', fontFamily: 'inherit' }}>·</span>
        {/* vᵀ row vector */}
        <div className="matrix">
          <span className="bracket">[</span>
          <span className="matrix-rows">
            <span>
              <span style={{ fontWeight: 700 }}>1</span>
              {'  '}
              <span style={{ color: IKB, fontWeight: 700 }}>2</span>
            </span>
          </span>
          <span className="bracket">]</span>
        </div>
      </div>
      <p style={{ marginTop: '0.9rem', fontSize: '0.9rem', color: '#555', lineHeight: 1.6 }}>
        验算：<span style={{ color: RUST, fontWeight: 600 }}>u = [2, 1]ᵀ</span>，
        <span style={{ color: IKB, fontWeight: 600 }}> vᵀ = [1, 2]</span>。
        {' '}u·vᵀ 的 (i,j) 元素 = uᵢ·vⱼ，所以第 1 列 = 1×[2,1]ᵀ，第 2 列 = 2×[2,1]ᵀ——
        正好平行，rank = 1。<br />
        这是 LoRA 核心思路的种子：<code>ΔW = B·A</code>，rank r，
        让大矩阵的改变量只活在一个低维子空间里。
      </p>
    </section>
  )
}

// ── Code snippet ──────────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

# 秩 = 独立方向数；det = 0  ⟺  降秩
M_rank2 = np.array([[1, 0], [0, 1]])    # det=1  → rank 2
M_rank1 = np.array([[2, 4], [1, 2]])    # det=0  → rank 1
print(np.linalg.matrix_rank(M_rank1))   # 1

# rank-1 矩阵恰好等于外积 u·vᵀ
u = np.array([2, 1])   # 列方向（column space 的基）
v = np.array([1, 2])   # 行模式（row space 的基）
print(np.outer(u, v))
# [[2 4]
#  [1 2]]  ← 与 M_rank1 完全相同

# LoRA: ΔW = B @ A，参数量从 d² 降到 2·r·d
d, r = 4096, 8               # 模型维度 d，低秩 r
B = np.random.randn(d, r)    # shape (d, r)
A = np.random.randn(r, d)    # shape (r, d)
delta_W = B @ A              # shape (d, d)，但 rank(ΔW) ≤ r = 8
print(f"全参数: {d*d:,}  LoRA 参数: {r*(d+d):,}  压缩比: {d*d//(r*(d+d))}×")`

// ── Main export ───────────────────────────────────────────────────────────────
export function Rank() {
  const [c1, setC1] = useState<Vec2>([1, 0])
  const [c2, setC2] = useState<Vec2>([0, 1])

  const rank = computeRank(c1, c2)
  const d = colDet(c1, c2)

  const me = findChapter('rank')!
  const { prev, next } = neighbors('rank')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb">
          <Link to="/">大纲</Link> · 第三部分 · 方阵的秘密
        </div>
        <div className="kicker">第 {me.num} 节 ★ 核心</div>
        <h1>
          矩阵的秩
          <span className="zh-sub">一个变换到底「留下」了几维？</span>
        </h1>
        <p className="lede">
          矩阵的 <strong>rank（秩）</strong>= 变换之后输出空间真正用到了几个独立方向——
          也就是 column space 的维数。对 2×2 矩阵：两列<strong>线性无关</strong>（det ≠ 0）→ rank 2，
          span 铺满整个平面；两列<strong>共线</strong>（det = 0，但有一列非零）→ rank 1，
          整个平面坍缩成一条线；两列全零 → rank 0。
          <strong>rank-1 矩阵恰好是 outer product <code>uvᵀ</code></strong>——这是
          low-rank 近似和 LoRA 的种子。
        </p>
      </header>

      {/* ── Preset controls ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">预设</span>
            <span style={{ color: '#888', fontSize: '0.85rem' }}>点击快速切换</span>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', paddingTop: '0.4rem' }}>
            <button
              onClick={() => { setC1([1, 0]); setC2([0, 1]) }}
              style={{
                padding: '0.35rem 0.9rem',
                border: `1.5px solid ${IKB}`,
                borderRadius: 4, background: 'white', color: IKB,
                cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit',
              }}
            >
              独立（rank 2）
            </button>
            <button
              onClick={() => { setC1([1, 0.5]); setC2([1.5, 0.75]) }}
              style={{
                padding: '0.35rem 0.9rem',
                border: `1.5px solid ${WARN}`,
                borderRadius: 4, background: 'white', color: WARN,
                cursor: 'pointer', fontSize: '0.9rem', fontFamily: 'inherit',
              }}
            >
              让两列平行（rank 1）
            </button>
          </div>
        </div>
      </section>

      {/* ── Interactive canvas ── */}
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
          <RankCanvas c1={c1} c2={c2} onC1={setC1} onC2={setC2} />
        </div>
        <p style={{ color: '#888', fontSize: '0.82rem', margin: 0 }}>
          拖动{' '}
          <span style={{ color: RUST, fontWeight: 700 }}>● c₁</span>（第一列）和{' '}
          <span style={{ color: IKB, fontWeight: 700 }}>● c₂</span>（第二列）——
          观察 span 如何从平行四边形坍缩成一条线
        </p>
      </section>

      {/* ── Matrix / det / rank readouts ── */}
      <section className="readouts">
        <div style={{
          display: 'flex', alignItems: 'center', gap: '2rem',
          flexWrap: 'wrap', justifyContent: 'center',
        }}>
          <div className="matrix">
            <span className="matrix-name">M</span>
            <span className="bracket">[</span>
            <span className="matrix-rows">
              <span>
                <span style={{ color: RUST, fontWeight: 700 }}>{fmt(c1[0])}</span>
                {'  '}
                <span style={{ color: IKB, fontWeight: 700 }}>{fmt(c2[0])}</span>
              </span>
              <span>
                <span style={{ color: RUST, fontWeight: 700 }}>{fmt(c1[1])}</span>
                {'  '}
                <span style={{ color: IKB, fontWeight: 700 }}>{fmt(c2[1])}</span>
              </span>
            </span>
            <span className="bracket">]</span>
          </div>
          <div style={{ fontSize: '0.92rem', lineHeight: 1.8 }}>
            <div>
              det ={' '}
              <strong style={{ color: Math.abs(d) < EPS ? WARN : '#1b1f24' }}>
                {fmt(d)}
              </strong>
            </div>
            <div>
              rank ={' '}
              <strong style={{ color: rank === 2 ? GREEN : WARN }}>{rank}</strong>
            </div>
          </div>
        </div>
      </section>

      {/* ── Verdict ── */}
      <section className={`verdict ${rank === 2 ? 'verdict--eq' : 'verdict--neq'}`}>
        {rank === 2 && (
          <p>
            <strong>rank = 2：两列线性无关，column space = 整个平面。</strong>
            {' '}det ≠ 0，变换可逆，平行四边形面积 = |det| = {fmt(Math.abs(d))}。
            任意输入向量都可以唯一还原——没有信息丢失。
            把两列拖到平行，亲眼看平行四边形如何坍缩成一条线。
          </p>
        )}
        {rank === 1 && (
          <p>
            <strong style={{ color: WARN }}>rank = 1：两列共线，det = 0，整个平面坍缩成一条线（虚线）。</strong>
            {' '}column space 只剩一个方向。无数不同的输入被送到同一条线上——
            信息被<em>不可逆地</em>压缩了。这样的矩阵恰好能写成 outer product
            {' '}<code>c₁·rᵀ</code>：一个列方向 × 一个行模式。
          </p>
        )}
        {rank === 0 && (
          <p>
            <strong style={{ color: WARN }}>rank = 0：两列都是零向量，整个空间压缩成原点。</strong>
            {' '}det = 0，所有输入都变成零向量，信息彻底丢失。
          </p>
        )}
      </section>

      {/* ── Static outer-product panel ── */}
      <OuterProductPanel />

      {/* ── Bridge ── */}
      <section className="bridge">
        <div className="bridge-tag">这就是 LLM 里的什么</div>
        <div className="bridge-body">
          <p>
            一个权重矩阵 <code>W</code> 的 rank 告诉你它真正用到了多少个「独立方向」。
            全 rank 的 <code>W</code> 能把输入映射到所有方向；低 rank 的 <code>W</code>
            只在一个低维子空间里工作——它的 column space 是有限的。
          </p>
          <p>
            <strong>LoRA 的核心赌注</strong>：微调时权重的改变量 <code>ΔW</code> 其实是
            {' '}<strong>低秩的</strong>——让模型适应新任务只需要少数几个新方向。
            所以把 <code>ΔW = B·A</code>（B: <code>d×r</code>，A: <code>r×d</code>，r ≪ d），
            只训练这两个瘦矩阵，参数量从 d² 压到 2·r·d。
            第 21 节 SVD 把 rank 的结构看得更清楚；第 22 节低秩近似、第 35 节 LoRA
            都是这一页的兑现。也连回第 12 节：<strong>det = 0 ⟺ 降秩</strong>。
          </p>
        </div>
      </section>

      {/* ── Code block ── */}
      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：rank、outer product、LoRA 参数量</h2>
        <CodeBlock code={SNIPPET} language="python" title="rank_and_lora.py" />
      </section>

      {/* ── Pagination ── */}
      <nav className="pager">
        {prev ? (
          <Link className="pager-link prev" to={prev.status === 'live' ? `/ch/${prev.slug}` : '/'}>
            <span className="pager-dir">← 上一节</span>
            <span className="pager-title">{prev.num} {prev.title}</span>
          </Link>
        ) : <span />}
        {next ? (
          <Link className="pager-link next" to={next.status === 'live' ? `/ch/${next.slug}` : '/'}>
            <span className="pager-dir">下一节 →</span>
            <span className="pager-title">
              {next.num} {next.title}{next.status !== 'live' && ' · 规划中'}
            </span>
          </Link>
        ) : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}
