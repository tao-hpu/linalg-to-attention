import { type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { findChapter, neighbors, allChapters } from '../chapters'

// 章节统一外壳：面包屑 + 章号 + 标题 + 副标题 + 正文 + 上一页/下一页。
// 标题/副标题/编号都从 chapters.ts 读，页面本身只写 lede 和交互正文。

export function ChapterShell({ slug, part, lede, children }: {
  slug: string
  part: string
  lede: ReactNode
  children: ReactNode
}) {
  const me = findChapter(slug)
  if (!me) return <div className="page">未找到章节：{slug}</div>
  const { prev, next } = neighbors(slug)
  const linkTo = (s: typeof prev) => (s && s.status === 'live' ? `/ch/${s.slug}` : '/')

  return (
    <article className="page">
      <header className="masthead">
        <div className="crumb"><Link to="/">大纲</Link> · {part}</div>
        <div className="kicker">第 {me.num} 节</div>
        <h1>{me.title}<span className="zh-sub">{me.hook}</span></h1>
        <p className="lede">{lede}</p>
      </header>

      {children}

      <nav className="pager">
        {prev
          ? <Link className="pager-link prev" to={linkTo(prev)}>
              <span className="pager-dir">← 上一节</span>
              <span className="pager-title">{prev.num} {prev.title}{prev.status !== 'live' && ' · 规划中'}</span>
            </Link>
          : <span />}
        {next
          ? <Link className="pager-link next" to={linkTo(next)}>
              <span className="pager-dir">下一节 →</span>
              <span className="pager-title">{next.num} {next.title}{next.status !== 'live' && ' · 规划中'}</span>
            </Link>
          : <span />}
      </nav>

      <p className="page-foot">共 {allChapters.length} 节 · 你在第 {me.num} 节</p>
    </article>
  )
}

// 通往 LLM 的桥接块，所有章节统一样式。
export function Bridge({ children }: { children: ReactNode }) {
  return (
    <section className="bridge">
      <div className="bridge-tag">这就是 LLM 里的什么</div>
      <div className="bridge-body">{children}</div>
    </section>
  )
}
