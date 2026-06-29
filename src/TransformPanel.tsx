import { apply, type Mat2, type Vec2 } from './linalg'

// 一个不对称的 “F” 形状（已大致以原点为中心）。
// 不对称很关键：旋转、切变、镜像、投影的效果才能一眼看出来。
const F_RAW: Vec2[] = [
  [0, 0], [0, 3], [1.8, 3], [1.8, 2.4], [0.6, 2.4],
  [0.6, 1.8], [1.5, 1.8], [1.5, 1.2], [0.6, 1.2], [0.6, 0],
]
const F_SHAPE: Vec2[] = F_RAW.map(([x, y]) => [x - 0.9, y - 1.5])

const SIZE = 260
const RANGE = 4 // 坐标范围 [-RANGE, RANGE]
const unit = SIZE / 2 / RANGE
const cx = SIZE / 2
const cy = SIZE / 2

// 数学坐标 -> 屏幕坐标（y 轴向上为正）
const sx = (x: number) => cx + x * unit
const sy = (y: number) => cy - y * unit

function gridLines() {
  const lines = []
  for (let i = -RANGE; i <= RANGE; i++) {
    const major = i === 0
    lines.push(
      <line key={`v${i}`} x1={sx(i)} y1={sy(-RANGE)} x2={sx(i)} y2={sy(RANGE)}
        className={major ? 'axis' : 'grid'} />,
      <line key={`h${i}`} x1={sx(-RANGE)} y1={sy(i)} x2={sx(RANGE)} y2={sy(i)}
        className={major ? 'axis' : 'grid'} />,
    )
  }
  return lines
}

function arrow(v: Vec2, className: string, key: string) {
  const [x, y] = v
  if (Math.hypot(x, y) < 1e-6) return null
  const tipX = sx(x), tipY = sy(y)
  const ang = Math.atan2(sy(0) - tipY, tipX - sx(0)) // 屏幕坐标系下的朝向
  const ah = 9 // 箭头长度
  const aw = 5 // 箭头半宽
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)
  return (
    <g key={key} className={className}>
      <line x1={sx(0)} y1={sy(0)} x2={tipX} y2={tipY} />
      <polygon points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`} />
    </g>
  )
}

export function TransformPanel({ M, label, sublabel, active }: {
  M: Mat2
  label: string
  sublabel?: string
  active?: boolean
}) {
  const pts = F_SHAPE.map((p) => apply(M, p))
  const path = pts.map((p) => `${sx(p[0])},${sy(p[1])}`).join(' ')

  // 变换后的基向量 = 矩阵的两列
  const iHat: Vec2 = [M[0], M[2]]
  const jHat: Vec2 = [M[1], M[3]]

  return (
    <figure className={`panel${active ? ' panel--active' : ''}`}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        <g className="grid-group">{gridLines()}</g>
        <polygon className="shape" points={path} />
        {arrow(iHat, 'ihat', 'i')}
        {arrow(jHat, 'jhat', 'j')}
      </svg>
      <figcaption>
        <span className="panel-label">{label}</span>
        {sublabel && <span className="panel-sub">{sublabel}</span>}
      </figcaption>
    </figure>
  )
}
