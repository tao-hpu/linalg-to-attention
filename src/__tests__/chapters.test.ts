import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { allChapters, parts, findChapter, neighbors } from '../chapters'

// 结构性检查。这一组不测数学，测的是「章节清单和它的所有下游还对得上吗」——
// 本项目出过的错基本都在这里：插入新章后旧章的编号引用没跟着改、
// 加了章却忘了挂路由、分享卡还印着旧的节数。

const ROOT = join(__dirname, '../..')
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8')
const pageFiles = readdirSync(join(ROOT, 'src/pages')).filter((f) => f.endsWith('.tsx'))

describe('chapters.ts 自身', () => {
  it('章号从 01 起连续、两位数、无重复', () => {
    expect(allChapters.map((c) => c.num)).toEqual(
      allChapters.map((_, i) => String(i + 1).padStart(2, '0')),
    )
  })

  it('slug 唯一且是 kebab-case', () => {
    const slugs = allChapters.map((c) => c.slug)
    expect(new Set(slugs).size).toBe(slugs.length)
    for (const s of slugs) expect(s).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/)
  })

  it('标题与 hook 都不为空', () => {
    for (const c of allChapters) {
      expect(c.title.trim(), c.slug).not.toBe('')
      expect(c.hook.trim(), c.slug).not.toBe('')
      expect(c.bridge.trim(), c.slug).not.toBe('')
    }
  })

  it('parts 展平后就是 allChapters，顺序一致', () => {
    expect(parts.flatMap((p) => p.chapters)).toEqual(allChapters)
  })

  it('neighbors 首尾开口、中间双向自洽', () => {
    expect(neighbors(allChapters[0].slug).prev).toBeUndefined()
    expect(neighbors(allChapters.at(-1)!.slug).next).toBeUndefined()
    for (let i = 1; i < allChapters.length; i++) {
      const cur = allChapters[i]
      expect(neighbors(cur.slug).prev?.slug).toBe(allChapters[i - 1].slug)
      expect(neighbors(allChapters[i - 1].slug).next?.slug).toBe(cur.slug)
    }
  })
})

describe('章节 ↔ 路由', () => {
  const app = read('src/App.tsx')

  it('每个 live 章节都有一条路由', () => {
    for (const c of allChapters) {
      if (c.status !== 'live') continue
      expect(app, `${c.slug} 缺路由`).toContain(`path="${c.slug}"`)
    }
  })

  it('路由不多于章节清单', () => {
    const routed = [...app.matchAll(/<Route path="([a-z0-9-]+)"/g)].map((m) => m[1])
    for (const slug of routed) {
      expect(findChapter(slug), `路由 ${slug} 在 chapters.ts 里查无此章`).toBeDefined()
    }
  })

  it('每个章节页都用 L(() => import(...)) 懒加载，不是提前求值的 Promise', () => {
    // 直接写 L(import(...)) 会在模块求值时把全部 chunk 一次性请求掉，lazy 白写。
    expect(app).not.toMatch(/\bL\(import\(/)
  })
})

describe('正文里的章节交叉引用', () => {
  it('<ChRef slug> 引用的章节都存在', () => {
    for (const f of pageFiles) {
      const src = read(`src/pages/${f}`)
      for (const m of src.matchAll(/<ChRef\s+slug="([a-z0-9-]+)"/g)) {
        expect(findChapter(m[1]), `${f} 引用了不存在的章节 ${m[1]}`).toBeDefined()
      }
    }
  })

  it('chNum / chLabel 引用的章节都存在', () => {
    for (const f of pageFiles) {
      const src = read(`src/pages/${f}`)
      for (const m of src.matchAll(/ch(?:Num|Label)\('([a-z0-9-]+)'/g)) {
        expect(findChapter(m[1]), `${f} 引用了不存在的章节 ${m[1]}`).toBeDefined()
      }
    }
  })

  it('页面里不再出现手写的「第 NN 节」', () => {
    // 手写编号会在插入新章时集体失效，一律走 <ChRef> / chNum。
    const offenders: string[] = []
    for (const f of pageFiles) {
      read(`src/pages/${f}`).split('\n').forEach((line, i) => {
        if (/第\s*\d{1,2}\s*节/.test(line)) offenders.push(`${f}:${i + 1}  ${line.trim().slice(0, 70)}`)
      })
    }
    expect(offenders, '改用 <ChRef slug="…" /> 或 chNum()').toEqual([])
  })
})

describe('章节数的下游产物', () => {
  it('分享卡戳记与章节数一致（不一致要跑 pnpm og）', () => {
    const stamp = JSON.parse(read('scripts/og.stamp.json'))
    expect(stamp.chapters).toBe(allChapters.length)
  })

  it('README 大纲写的节数与章节清单一致', () => {
    const m = read('README.md').match(/共 (\d+) 节/)
    expect(m, 'README 里找不到「共 N 节」').not.toBeNull()
    expect(Number(m![1])).toBe(allChapters.length)
  })
})
