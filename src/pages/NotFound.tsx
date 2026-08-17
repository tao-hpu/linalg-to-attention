import { Link, useLocation } from 'react-router-dom'
import { allChapters, chapterPath, firstLiveChapter, type Chapter } from '../chapters'

// 404：任何未命中的路径都落到这里，而不是渲染出一个只剩页脚的空壳。
// 顺手做一次「你是不是想找这个」的猜测——按拼写距离在章节清单里找最接近的几条。

/** 归一化的编辑距离：0 = 完全相同，1 = 毫无关系。 */
function distance(a: string, b: string): number {
  if (!a || !b) return 1
  const m = a.length
  const n = b.length
  let prev = Array.from({ length: n + 1 }, (_, j) => j)
  for (let i = 1; i <= m; i++) {
    const cur = [i]
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      )
    }
    prev = cur
  }
  return prev[n] / Math.max(m, n)
}

/** 从路径里猜读者想去的章节：先看包含关系，再看拼写距离。 */
function guess(pathname: string): Chapter[] {
  const raw = pathname.replace(/^\/+|\/+$/g, '').replace(/^ch\//, '').toLowerCase()
  if (!raw) return []
  return allChapters
    .filter((c) => c.status === 'live')
    .map((c) => ({
      c,
      d: c.slug.includes(raw) || raw.includes(c.slug) ? 0 : distance(raw, c.slug),
    }))
    .filter((x) => x.d < 0.55)
    .sort((a, b) => a.d - b.d)
    .slice(0, 3)
    .map((x) => x.c)
}

export function NotFound() {
  const { pathname } = useLocation()
  const suggestions = guess(pathname)
  const start = firstLiveChapter()

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb"><Link to="/">大纲</Link> · 404</div>
        <div className="kicker">页面不存在</div>
        <h1>
          没有这一节
          <span className="zh-sub">这个地址在全书 {allChapters.length} 节里找不到对应的内容</span>
        </h1>
        <p className="lede">
          你访问的是 <code>{pathname}</code>。
          可能是链接拼错了，也可能是这一节的地址变过。
          下面几条路都能把你带回正轨。
        </p>
      </header>

      {suggestions.length > 0 && (
        <section className="note">
          <p style={{ marginBottom: 12 }}>是不是想找这几节？</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {suggestions.map((c) => (
              <Link
                key={c.slug}
                to={chapterPath(c)}
                style={{
                  display: 'inline-flex', alignItems: 'baseline', gap: 8,
                  padding: '7px 14px', borderRadius: 4, textDecoration: 'none',
                  border: '1.5px solid var(--ikb)', color: 'var(--ikb)',
                  fontSize: 14, fontWeight: 600,
                }}
              >
                <span style={{ fontFamily: 'var(--mono)', opacity: 0.7 }}>{c.num}</span>
                <span>{c.title}</span>
              </Link>
            ))}
          </div>
        </section>
      )}

      <section className="note">
        <p>
          也可以直接<Link to="/">回大纲</Link>挑一节，
          或按 <kbd>⌘K</kbd> / <kbd>Ctrl</kbd>+<kbd>K</kbd> 搜索章节
          {start && <>；不知道从哪开始就从 <Link to={chapterPath(start)}>{start.num} {start.title}</Link> 起</>}。
        </p>
      </section>

      <nav className="pager">
        <Link className="pager-link prev" to="/">
          <span className="pager-dir">← 返回</span>
          <span className="pager-title">全书大纲</span>
        </Link>
        <span />
      </nav>
    </article>
  )
}
