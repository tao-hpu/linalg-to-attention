// 最小 2×2 线性代数工具 —— 整个项目的可视化都建在这之上。
// 矩阵用行主序数组表示： [[a, b], [c, d]]  ->  Mat2 = [a, b, c, d]

export type Mat2 = [number, number, number, number]
export type Vec2 = [number, number]

export const IDENTITY: Mat2 = [1, 0, 0, 1]

/** 矩阵乘法 A·B。几何含义：先做 B 这个动作，再做 A 这个动作。 */
export function multiply(A: Mat2, B: Mat2): Mat2 {
  const [a, b, c, d] = A
  const [e, f, g, h] = B
  return [
    a * e + b * g, a * f + b * h,
    c * e + d * g, c * f + d * h,
  ]
}

/** 把矩阵 M 作用在向量 v 上。 */
export function apply(M: Mat2, v: Vec2): Vec2 {
  const [a, b, c, d] = M
  const [x, y] = v
  return [a * x + b * y, c * x + d * y]
}

const rad = (deg: number) => (deg * Math.PI) / 180

// 几种基础几何变换，每个就是一个 2×2 矩阵。
export const transforms = {
  /** 旋转 θ 度（绕原点） */
  rotate: (deg: number): Mat2 => {
    const t = rad(deg)
    return [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)]
  },
  /** 沿 x 方向拉伸 s 倍（故意不对称，才能看出不可交换） */
  scaleX: (s: number): Mat2 => [s, 0, 0, 1],
  /** 水平切变，系数 k */
  shear: (k: number): Mat2 => [1, k, 0, 1],
  /** 正交投影到 x 轴（把整个平面压成一条线） */
  projectX: (): Mat2 => [1, 0, 0, 0],
}

/** 把矩阵格式化成两行，方便在界面上显示。 */
export function format(M: Mat2): [string, string] {
  const f = (n: number) => {
    const r = Math.round(n * 100) / 100
    return (Object.is(r, -0) ? 0 : r).toFixed(2)
  }
  return [`${f(M[0])}  ${f(M[1])}`, `${f(M[2])}  ${f(M[3])}`]
}

/** 判断两个矩阵在容差内是否相等（用来判定 AB 是否真的等于 BA）。 */
export function nearlyEqual(A: Mat2, B: Mat2, eps = 1e-6): boolean {
  return A.every((v, i) => Math.abs(v - B[i]) < eps)
}
