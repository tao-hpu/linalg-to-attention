// 2D 向量工具 —— 第一部分（向量）所有页面共用。
// 与 linalg.ts(矩阵) 分开：这里只关心向量本身。

export interface V { x: number; y: number }

export const v = (x: number, y: number): V => ({ x, y })
export const add = (a: V, b: V): V => ({ x: a.x + b.x, y: a.y + b.y })
export const sub = (a: V, b: V): V => ({ x: a.x - b.x, y: a.y - b.y })
export const scale = (a: V, k: number): V => ({ x: a.x * k, y: a.y * k })
export const dot = (a: V, b: V): number => a.x * b.x + a.y * b.y
export const norm = (a: V): number => Math.hypot(a.x, a.y)

export const normalize = (a: V): V => {
  const n = norm(a)
  return n < 1e-9 ? { x: 0, y: 0 } : { x: a.x / n, y: a.y / n }
}

/** a 与 b 的夹角（弧度，0..π） */
export const angleBetween = (a: V, b: V): number => {
  const d = norm(a) * norm(b)
  if (d < 1e-9) return 0
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b) / d)))
}

/** a 在 b 方向上的标量投影（带符号）：|a|cosθ */
export const scalarProjection = (a: V, b: V): number => {
  const n = norm(b)
  return n < 1e-9 ? 0 : dot(a, b) / n
}

/** a 在 b 上的向量投影 */
export const projectOnto = (a: V, b: V): V => {
  const d = dot(b, b)
  return d < 1e-9 ? { x: 0, y: 0 } : scale(b, dot(a, b) / d)
}

export const cosineSimilarity = (a: V, b: V): number => {
  const d = norm(a) * norm(b)
  return d < 1e-9 ? 0 : dot(a, b) / d
}

export const fmt = (n: number, digits = 2): string => {
  const r = Number(n.toFixed(digits))
  return (Object.is(r, -0) ? 0 : r).toFixed(digits)
}

export const degrees = (rad: number): number => (rad * 180) / Math.PI
