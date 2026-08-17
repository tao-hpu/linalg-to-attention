import { describe, it, expect } from 'vitest'
import { SCORES, softmaxRow, maskedScores, weightsOf } from '../pages/CausalMask'

// 第 32 节的三种模式。页面正文直接引用这里的数字（行和塌到多少、
// 有多少信息来自未来），所以数字必须站得住。

const N = SCORES.length
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0)
const upper = <T,>(m: T[][], i: number) => m[i].filter((_, j) => j > i)

describe('分数矩阵', () => {
  it('是 n×n 的方阵', () => {
    expect(N).toBeGreaterThan(1)
    SCORES.forEach((row) => expect(row.length).toBe(N))
  })

  it('分数有足够的动态范围，softmax 出来才不是一片均匀', () => {
    const flat = SCORES.flat()
    expect(Math.max(...flat) - Math.min(...flat)).toBeGreaterThan(1.5)
  })
})

describe('因果掩码', () => {
  it('只把严格上三角置 −∞，对角线保留（每个 token 看得见自己）', () => {
    const s = maskedScores('causal')
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (j > i) expect(s[i][j]).toBe(-Infinity)
        else expect(s[i][j]).toBe(SCORES[i][j])
      }
    }
  })

  it('softmax 后未来位置恰好为 0，且行和仍是 1', () => {
    const w = weightsOf('causal')
    for (let i = 0; i < N; i++) {
      upper(w, i).forEach((v) => expect(v).toBe(0))
      expect(Math.abs(sum(w[i]) - 1)).toBeLessThan(1e-12)
    }
  })

  it('第一行只能看自己，权重必然是 1', () => {
    expect(weightsOf('causal')[0][0]).toBe(1)
  })

  it('掩码只改变候选集合，不改变可见部分的相对比例', () => {
    // 去掉若干候选后重新归一化，剩下位置两两之间的比值应当不变
    const w = weightsOf('causal')
    const none = weightsOf('none')
    for (let i = 1; i < N; i++) {
      for (let j = 1; j <= i; j++) {
        expect(Math.abs(w[i][j] / w[i][0] - none[i][j] / none[i][0])).toBeLessThan(1e-9)
      }
    }
  })
})

describe('两种错误做法', () => {
  it('不掩码时，靠前的位置有大量信息来自后文', () => {
    const w = weightsOf('none')
    expect(sum(upper(w, 0))).toBeGreaterThan(0.5)   // 第 1 个 token 主要在看未来
  })

  it('softmax 之后再抹零：行和塌下去，只有最后一行还是 1', () => {
    const w = weightsOf('after')
    for (let i = 0; i < N - 1; i++) expect(sum(w[i])).toBeLessThan(1 - 1e-6)
    expect(Math.abs(sum(w[N - 1]) - 1)).toBeLessThan(1e-12)
  })

  it('后抹零与正确掩码只差一个逐行常数——正是丢掉的那部分概率', () => {
    const after = weightsOf('after')
    const causal = weightsOf('causal')
    for (let i = 0; i < N; i++) {
      const k = sum(after[i])
      for (let j = 0; j <= i; j++) expect(Math.abs(after[i][j] / k - causal[i][j])).toBeLessThan(1e-9)
    }
  })
})

describe('softmaxRow', () => {
  it('全 −∞ 的行返回全 0，而不是 NaN', () => {
    expect(softmaxRow([-Infinity, -Infinity])).toEqual([0, 0])
  })

  it('平移不变', () => {
    const a = softmaxRow([1, 2, 3])
    const b = softmaxRow([101, 102, 103])
    a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(1e-12))
  })
})
