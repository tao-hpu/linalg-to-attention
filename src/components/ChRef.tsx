import { Link } from 'react-router-dom'
import { findChapter } from '../chapters'

// 正文里引用别的章节，一律走这里，不要手写「第 28 节」。
//
// 手写的编号会在插入新章时集体失效：往中间加一节，其后所有章号 +1，
// 而散在三十几个页面里的引用不会跟着动。审计时抓到过 6 处这样的错引。
// 从 chapters.ts 取，编号漂移就不可能再发生。
//
//   <ChRef slug="normalization" />          → 第 28 节
//   <ChRef slug="normalization" title />    → 第 28 节 · 归一化
//   <ChRef slug="normalization" link={false} />  → 同上，但不带链接
//
// 字符串上下文（代码片段、data 里的 label）用 chNum / chLabel。

/** 章号，如 "28"。slug 不存在时返回 "??"。 */
export function chNum(slug: string): string {
  return findChapter(slug)?.num ?? '??'
}

/** "第 28 节"；withTitle 时为 "第 28 节 · 归一化"。 */
export function chLabel(slug: string, withTitle = false): string {
  const c = findChapter(slug)
  if (!c) return `第 ?? 节`
  return withTitle ? `第 ${c.num} 节 · ${c.title}` : `第 ${c.num} 节`
}

export function ChRef({ slug, title = false, link = true }: {
  slug: string
  /** 带上章节标题（第 28 节 · 归一化）。 */
  title?: boolean
  /** 关掉链接：外层已经是 <a> 时必须关，否则 <a> 嵌套。 */
  link?: boolean
}) {
  const c = findChapter(slug)
  // slug 打错不能静默——正文里留个显眼的记号，交叉引用检查也能扫出来。
  if (!c) return <span className="chref chref--missing">第 ?? 节（未知章节 {slug}）</span>

  const text = chLabel(slug, title)
  // 规划中的章节没有可用路由，退化成纯文本。
  if (!link || c.status !== 'live') return <span className="chref">{text}</span>

  return <Link className="chref" to={`/ch/${c.slug}`}>{text}</Link>
}
