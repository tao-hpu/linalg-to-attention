import { describe, it, expect } from 'vitest'
import * as vec from '../vec'
import { multiply, apply, transforms, nearlyEqual, IDENTITY, type Mat2 } from '../linalg'
import { computeSoftmax as softmaxT } from '../pages/Softmax'
import { softmax as ceSoftmax } from '../pages/CrossEntropy'
import { layerNorm, rmsNorm, mean, variance } from '../pages/Normalization'
import {
  computeSoftmax as sdSoftmax, renormalize, topKMask, topPMask, deterministicSample,
} from '../pages/SamplingDecoding'
import { quantChannel, runQuant } from '../pages/Quantization'
import { computeSVD } from '../pages/SVD'
import { computeEigen } from '../pages/Eigen'
import { computePCA, toDataPoints } from '../pages/PCA'

// 各页交互里那些「算出来的数字」。页面的文案会直接引用这些结果，
// 算错了整段讲解就跟着错——审计里几条最要命的问题都属于这一类，
// 所以这里全部按定义独立复算一遍，不复用被测代码的中间量。

const close = (a: number, b: number, eps = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(eps)
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0)

describe('vec.ts', () => {
  it('内积、范数、单位化', () => {
    close(vec.dot(vec.v(3, 4), vec.v(1, 2)), 11)
    close(vec.norm(vec.v(3, 4)), 5)
    const u = vec.normalize(vec.v(3, 4))
    close(vec.norm(u), 1)
    close(u.x, 0.6); close(u.y, 0.8)
  })

  it('零向量单位化不产生 NaN', () => {
    const u = vec.normalize(vec.v(0, 0))
    expect(Number.isFinite(u.x) && Number.isFinite(u.y)).toBe(true)
  })

  it('投影：a 在 b 上的投影平行于 b，残差垂直于 b', () => {
    const a = vec.v(3, 1), b = vec.v(2, 0)
    const p = vec.projectOnto(a, b)
    close(p.x, 3); close(p.y, 0)
    close(vec.dot(vec.sub(a, p), b), 0)
  })

  it('标量投影 = |a|cosθ，cosine 相似度与夹角一致', () => {
    const a = vec.v(1, 1), b = vec.v(1, 0)
    close(vec.scalarProjection(a, b), Math.SQRT1_2 * vec.norm(a))
    close(vec.cosineSimilarity(a, b), Math.SQRT1_2)
    close(vec.degrees(vec.angleBetween(a, b)), 45, 1e-9)
  })

  it('正交向量 cosine = 0，反向 = −1', () => {
    close(vec.cosineSimilarity(vec.v(1, 0), vec.v(0, 5)), 0)
    close(vec.cosineSimilarity(vec.v(1, 2), vec.v(-2, -4)), -1, 1e-12)
  })
})

describe('linalg.ts', () => {
  it('矩阵乘法不可交换，与单位阵可交换', () => {
    const A = transforms.rotate(30), B = transforms.shear(0.5)
    expect(nearlyEqual(multiply(A, B), multiply(B, A))).toBe(false)
    expect(nearlyEqual(multiply(A, IDENTITY), A)).toBe(true)
  })

  it('乘法顺序 = 先右后左作用于向量', () => {
    const A = transforms.rotate(90), B = transforms.scaleX(2)
    const v: [number, number] = [1, 1]
    const viaMatrix = apply(multiply(A, B), v)
    const viaSteps = apply(A, apply(B, v))
    close(viaMatrix[0], viaSteps[0], 1e-12)
    close(viaMatrix[1], viaSteps[1], 1e-12)
  })

  it('旋转保长；两次 90° 等于 180°', () => {
    const r90 = transforms.rotate(90)
    const p = apply(r90, [3, 4])
    close(Math.hypot(p[0], p[1]), 5, 1e-12)
    expect(nearlyEqual(multiply(r90, r90), transforms.rotate(180))).toBe(true)
  })
})

describe('softmax（第 29 节 / 交叉熵 / 采样三处实现）', () => {
  const logits = [2.0, 1.0, 0.1, -0.5]

  it('输出是概率分布：非负且和为 1', () => {
    for (const p of [softmaxT(logits, 1).probs, ceSoftmax([...logits]), sdSoftmax(logits, 1)]) {
      expect(p.every((x) => x >= 0)).toBe(true)
      close(sum(p), 1, 1e-12)
    }
  })

  it('平移不变：整体加常数结果不变（数值稳定的根据）', () => {
    const a = sdSoftmax(logits, 1)
    const b = sdSoftmax(logits.map((z) => z + 100), 1)
    a.forEach((x, i) => close(x, b[i], 1e-12))
  })

  it('三处实现在 T=1 时给出同一组概率', () => {
    const a = softmaxT(logits, 1).probs, b = ceSoftmax([...logits]), c = sdSoftmax(logits, 1)
    a.forEach((x, i) => { close(x, b[i], 1e-12); close(x, c[i], 1e-12) })
  })

  it('温度：T→小 分布变尖，T→大 趋于均匀', () => {
    const cold = sdSoftmax(logits, 0.1)
    const hot = sdSoftmax(logits, 100)
    expect(Math.max(...cold)).toBeGreaterThan(0.99)
    hot.forEach((p) => expect(Math.abs(p - 1 / logits.length)).toBeLessThan(0.02)) // 趋近 1/n
  })

  it('大 logit 不溢出（减最大值的意义）', () => {
    const p = sdSoftmax([1000, 999, 998], 1)
    expect(p.every(Number.isFinite)).toBe(true)
    close(sum(p), 1, 1e-12)
  })
})

describe('归一化（第 28 节）', () => {
  const xs = [2, 4, 4, 4, 5, 5, 7, 9]

  it('LayerNorm 后均值 0、方差 1（γ=1, β=0）', () => {
    const y = layerNorm(xs, 1, 0)
    close(mean(y), 0, 1e-9)
    // 分母是 √(σ²+ε)，ε=1e-5，所以方差只能逼近 1 到 ε 量级
    close(variance(y, mean(y)), 1, 1e-5)
  })

  it('γ、β 是仿射参数：输出 = γ·标准化 + β', () => {
    const base = layerNorm(xs, 1, 0)
    const scaled = layerNorm(xs, 2.5, -1)
    base.forEach((b, i) => close(scaled[i], 2.5 * b - 1, 1e-9))
  })

  it('RMSNorm 只除均方根，不减均值——所以 σ 一般不等于 γ', () => {
    const y = rmsNorm(xs, 1)
    const rms = Math.sqrt(sum(xs.map((x) => x * x)) / xs.length + 1e-5)   // 同样带 ε
    y.forEach((v, i) => close(v, xs[i] / rms, 1e-9))
    // 均值非零时，RMSNorm 的输出标准差不会等于 γ
    expect(Math.abs(Math.sqrt(variance(y, mean(y))) - 1)).toBeGreaterThan(0.1)
  })

  it('两者在输入已零均值时重合', () => {
    const centered = xs.map((x) => x - mean(xs))
    const ln = layerNorm(centered, 1, 0)
    const rn = rmsNorm(centered, 1)
    ln.forEach((v, i) => close(v, rn[i], 1e-6))
  })
})

describe('采样与解码（第 37 节）', () => {
  const probs = [0.66, 0.2, 0.08, 0.04, 0.02]

  it('top-k 只保留最大的 k 个', () => {
    expect(topKMask(probs, 2)).toEqual([true, true, false, false, false])
    expect(topKMask(probs, probs.length).every(Boolean)).toBe(true)
  })

  it('top-p 取累积概率刚好越过 p 的最小集合', () => {
    expect(topPMask(probs, 0.5).mask).toEqual([true, false, false, false, false])
    expect(topPMask(probs, 0.9).mask).toEqual([true, true, true, false, false])
  })

  it('top-p 至少保留一个 token', () => {
    expect(topPMask(probs, 0).mask.filter(Boolean).length).toBeGreaterThanOrEqual(1)
  })

  it('renormalize 后和为 1；全零输入不产生 NaN', () => {
    close(sum(renormalize([0.66, 0.2, 0, 0, 0])), 1, 1e-12)
    expect(renormalize([0, 0, 0]).every((x) => x === 0)).toBe(true)
  })

  it('确定性采样：前几步就必须出现不同的 token', () => {
    // 曾经用 (step % n)/n 当阈值，前 60 多次点击全落在同一个词上，
    // 「采样是随机的」这件事在页面上完全看不出来。
    const first = Array.from({ length: 5 }, (_, s) => deterministicSample(probs, s))
    expect(new Set(first).size).toBeGreaterThan(1)
  })

  it('确定性采样的长期频率收敛到真实概率', () => {
    const counts = new Array(probs.length).fill(0)
    const N = 20000
    for (let s = 0; s < N; s++) counts[deterministicSample(probs, s)]++
    counts.forEach((c, i) => expect(Math.abs(c / N - probs[i])).toBeLessThan(0.005))
  })

  it('同一 step 永远给同一结果（可复现，不依赖 Math.random）', () => {
    for (let s = 0; s < 50; s++) {
      expect(deterministicSample(probs, s)).toBe(deterministicSample(probs, s))
    }
  })
})

describe('量化（第 38 节）', () => {
  const w = [0.9, -0.4, 0.1, -0.02, 0.35, -0.77, 0.5, 0.05]

  it('对称量化：scale = max|w| / (2^(b−1) − 1)，误差不超过半个 scale', () => {
    const { scale, wHat, errors } = quantChannel(w, 127, -128)
    close(scale, Math.max(...w.map(Math.abs)) / 127, 1e-12)
    errors.forEach((e) => expect(e).toBeLessThanOrEqual(scale / 2 + 1e-12))
    wHat.forEach((v) => close(Math.round(v / scale) * scale, v, 1e-12))
  })

  it('位宽越低误差越大', () => {
    const e8 = runQuant(w, 8, false).meanErr
    const e4 = runQuant(w, 4, false).meanErr
    expect(e4).toBeGreaterThan(e8)
  })

  it('fp32 不损失', () => {
    const r = runQuant(w, 'fp32', false)
    expect(r.maxErr).toBe(0)
    expect(r.bytesPerWeight).toBe(4)
  })

  it('存在离群值时，per-channel 不劣于 per-tensor', () => {
    const outlier = [...w.slice(0, 4), 12, ...w.slice(5)]
    const perTensor = runQuant(outlier, 4, false).meanErr
    const perChannel = runQuant(outlier, 4, true).meanErr
    expect(perChannel).toBeLessThanOrEqual(perTensor)
  })

  it('全零权重不会除以 0', () => {
    const r = quantChannel([0, 0, 0], 127, -128)
    expect(r.wHat.every(Number.isFinite)).toBe(true)
  })
})

describe('SVD（第 21 节）', () => {
  const cases: Mat2[] = [[2, 1, 0, 3], [1, 0, 0, 1], [3, 0, 0, 1], [0, -2, 1, 0], [1, 2, 2, 4]]

  it('σ₁ ≥ σ₂ ≥ 0，且 σ₁σ₂ = |det M|', () => {
    for (const M of cases) {
      const { sigma1, sigma2 } = computeSVD(M)
      expect(sigma1).toBeGreaterThanOrEqual(sigma2)
      expect(sigma2).toBeGreaterThanOrEqual(-1e-12)
      close(sigma1 * sigma2, Math.abs(M[0] * M[3] - M[1] * M[2]), 1e-9)
    }
  })

  it('U、V 是正交阵（列单位、互相垂直）', () => {
    for (const M of cases) {
      const { u1, u2, v1, v2 } = computeSVD(M)
      for (const [a, b] of [[u1, u2], [v1, v2]] as const) {
        close(Math.hypot(...a), 1, 1e-9)
        close(Math.hypot(...b), 1, 1e-9)
        close(a[0] * b[0] + a[1] * b[1], 0, 1e-9)
      }
    }
  })

  it('U·Σ·Vᵀ 还原出原矩阵', () => {
    for (const M of cases) {
      const { U, Sig, Vt } = computeSVD(M)
      expect(nearlyEqual(multiply(multiply(U, Sig), Vt), M, 1e-9), JSON.stringify(M)).toBe(true)
    }
  })

  it('降秩矩阵的 σ₂ = 0', () => {
    close(computeSVD([1, 2, 2, 4]).sigma2, 0, 1e-9)
  })
})

describe('特征值（第 15 节）', () => {
  it('实特征值满足 λ₁+λ₂ = trace、λ₁λ₂ = det', () => {
    for (const M of [[2, 1, 1, 2], [3, 0, 0, 1], [1, 2, 0, 1]] as Mat2[]) {
      const r = computeEigen(M)
      expect(r.kind).toBe('real')
      if (r.kind !== 'real') return
      close(r.λ1 + r.λ2, M[0] + M[3], 1e-9)
      close(r.λ1 * r.λ2, M[0] * M[3] - M[1] * M[2], 1e-9)
    }
  })

  it('特征向量确实满足 Mv = λv', () => {
    const M: Mat2 = [2, 1, 1, 2]
    const r = computeEigen(M)
    if (r.kind !== 'real') throw new Error('应为实特征值')
    for (const [λ, e] of [[r.λ1, r.e1], [r.λ2, r.e2]] as const) {
      const Mv = apply(M, e)
      close(Mv[0], λ * e[0], 1e-9)
      close(Mv[1], λ * e[1], 1e-9)
    }
  })

  it('纯旋转没有实特征值', () => {
    expect(computeEigen(transforms.rotate(90)).kind).toBe('complex')
  })

  it('重根被标出来', () => {
    const r = computeEigen([2, 1, 0, 2])
    expect(r.kind === 'real' && r.repeated).toBe(true)
  })
})

describe('PCA（第 23 节）', () => {
  it('主轴单位正交，λ₁ ≥ λ₂ ≥ 0', () => {
    const pts = toDataPoints([[-2, -1], [-1, -0.4], [0, 0.2], [1, 0.7], [2, 1.4], [0.5, -0.3]])
    const { lam1, lam2, v1, v2 } = computePCA(pts)
    expect(lam1).toBeGreaterThanOrEqual(lam2)
    expect(lam2).toBeGreaterThanOrEqual(0)
    close(Math.hypot(...v1), 1, 1e-9)
    close(v1[0] * v2[0] + v1[1] * v2[1], 0, 1e-9)
  })

  it('λ 之和 = 总方差（迹）', () => {
    const pts = toDataPoints([[1, 3], [2, -1], [-3, 0.5], [0, 2], [4, -2]])
    const r = computePCA(pts)
    close(r.lam1 + r.lam2, r.cxx + r.cyy, 1e-9)
  })

  it('沿 x 轴拉长的点云，PC1 就是 x 轴', () => {
    const pts = toDataPoints([[-3, 0.05], [-1, -0.05], [0, 0], [1, 0.05], [3, -0.05]])
    const { v1 } = computePCA(pts)
    expect(Math.abs(v1[0])).toBeGreaterThan(0.99)
  })

  it('平移点云不改变主轴，只改变质心', () => {
    const raw: [number, number][] = [[-2, -1], [-1, -0.4], [0, 0.2], [1, 0.7], [2, 1.4]]
    const a = computePCA(toDataPoints(raw))
    const b = computePCA(toDataPoints(raw.map(([x, y]) => [x + 10, y - 7])))
    close(Math.abs(a.v1[0] * b.v1[0] + a.v1[1] * b.v1[1]), 1, 1e-9)
    close(b.mx - a.mx, 10, 1e-9)
    close(a.lam1, b.lam1, 1e-9)
  })
})
