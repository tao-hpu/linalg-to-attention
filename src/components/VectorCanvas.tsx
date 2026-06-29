import { useRef, useState, type ReactNode } from 'react'
import type { V } from '../vec'

// 可拖拽的 2D 向量画布 —— 第一部分所有向量页面共用。
//
// 用法：
//   <VectorCanvas
//     vectors={[{ id: 'a', x: 2, y: 1, color: '#002fa7', label: 'a', draggable: true }]}
//     onDrag={(id, x, y) => setState(...)}
//   >
//     {({ sx, sy }) => (  // 可选：在数学坐标系里画额外的叠加层（投影线、平行四边形…）
//       <line x1={sx(0)} y1={sy(0)} x2={sx(2)} y2={sy(1)} stroke="#ccc" />
//     )}
//   </VectorCanvas>

export interface CanvasVector {
  id: string
  x: number
  y: number
  color: string
  label?: string
  draggable?: boolean
  dashed?: boolean
  width?: number
}

export interface CanvasHelpers {
  sx: (x: number) => number
  sy: (y: number) => number
  unit: number
}

interface Props {
  vectors: CanvasVector[]
  onDrag?: (id: string, x: number, y: number) => void
  size?: number
  range?: number
  snap?: number          // 拖拽吸附步长（默认 0.5），传 0 关闭
  children?: (h: CanvasHelpers) => ReactNode  // 叠加层（画在向量之下）
  overlay?: (h: CanvasHelpers) => ReactNode    // 叠加层（画在向量之上）
}

export function VectorCanvas({
  vectors, onDrag, size = 360, range = 5, snap = 0.5, children, overlay,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [active, setActive] = useState<string | null>(null)

  const unit = size / 2 / range
  const cx = size / 2
  const cy = size / 2
  const sx = (x: number) => cx + x * unit
  const sy = (y: number) => cy - y * unit

  const clientToMath = (clientX: number, clientY: number): V => {
    const svg = svgRef.current!
    const rect = svg.getBoundingClientRect()
    const px = ((clientX - rect.left) / rect.width) * size
    const py = ((clientY - rect.top) / rect.height) * size
    let x = (px - cx) / unit
    let y = (cy - py) / unit
    x = Math.max(-range, Math.min(range, x))
    y = Math.max(-range, Math.min(range, y))
    if (snap > 0) {
      x = Math.round(x / snap) * snap
      y = Math.round(y / snap) * snap
    }
    return { x, y }
  }

  const helpers: CanvasHelpers = { sx, sy, unit }

  const grid: ReactNode[] = []
  for (let i = -range; i <= range; i++) {
    const major = i === 0
    grid.push(
      <line key={`v${i}`} x1={sx(i)} y1={sy(-range)} x2={sx(i)} y2={sy(range)}
        className={major ? 'vc-axis' : 'vc-grid'} />,
      <line key={`h${i}`} x1={sx(-range)} y1={sy(i)} x2={sx(range)} y2={sy(i)}
        className={major ? 'vc-axis' : 'vc-grid'} />,
    )
  }

  return (
    <svg ref={svgRef} className="vector-canvas" width={size} height={size}
      viewBox={`0 0 ${size} ${size}`}>
      <g>{grid}</g>
      {children && children(helpers)}
      {vectors.map((vec) => (
        <VectorArrow key={vec.id} vec={vec} sx={sx} sy={sy} />
      ))}
      {overlay && overlay(helpers)}
      {/* 拖拽手柄画在最上层，保证可点 */}
      {vectors.filter((v) => v.draggable).map((vec) => (
        <circle
          key={`h-${vec.id}`}
          cx={sx(vec.x)} cy={sy(vec.y)} r={11}
          className={`vc-handle ${active === vec.id ? 'is-active' : ''}`}
          style={{ cursor: 'grab' }}
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture(e.pointerId)
            setActive(vec.id)
          }}
          onPointerMove={(e) => {
            if (active !== vec.id) return
            const m = clientToMath(e.clientX, e.clientY)
            onDrag?.(vec.id, m.x, m.y)
          }}
          onPointerUp={(e) => {
            (e.target as Element).releasePointerCapture(e.pointerId)
            setActive(null)
          }}
        />
      ))}
    </svg>
  )
}

function VectorArrow({ vec, sx, sy }: {
  vec: CanvasVector
  sx: (x: number) => number
  sy: (y: number) => number
}) {
  const { x, y, color, label, dashed, width = 2.5 } = vec
  if (Math.hypot(x, y) < 1e-6) {
    return <circle cx={sx(0)} cy={sy(0)} r={3} fill={color} />
  }
  const tipX = sx(x), tipY = sy(y)
  const ang = Math.atan2(sy(0) - tipY, tipX - sx(0))
  const ah = 11, aw = 6
  const back = (side: number): [number, number] => [
    tipX - ah * Math.cos(ang) - side * aw * Math.sin(ang),
    tipY + ah * Math.sin(ang) - side * aw * Math.cos(ang),
  ]
  const [bx1, by1] = back(1)
  const [bx2, by2] = back(-1)
  return (
    <g>
      <line x1={sx(0)} y1={sy(0)} x2={tipX} y2={tipY}
        stroke={color} strokeWidth={width}
        strokeDasharray={dashed ? '5 5' : undefined} strokeLinecap="round" />
      <polygon points={`${tipX},${tipY} ${bx1},${by1} ${bx2},${by2}`} fill={color} />
      {label && (
        <text x={tipX + (x >= -0.1 ? 10 : -10)} y={tipY + (y >= 0 ? -8 : 16)}
          fill={color} className="vc-label"
          textAnchor={x >= -0.1 ? 'start' : 'end'}>{label}</text>
      )}
    </g>
  )
}
