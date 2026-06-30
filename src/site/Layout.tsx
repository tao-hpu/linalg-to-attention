import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { parts, chapterPath, findChapter } from '../chapters'
import { ProgressProvider, useProgress } from './progress'
import { SearchPalette } from './SearchPalette'
import { useDocMeta } from './useDocMeta'

function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  const { pathname } = useLocation()
  const { isVisited } = useProgress()
  return (
    <nav className="sidebar" aria-label="全书导航">
      <Link to="/" className={`side-home ${pathname === '/' ? 'is-active' : ''}`} onClick={onNavigate}>
        全书大纲
      </Link>
      {parts.map((part) => (
        <div className="side-part" key={part.name}>
          <div className="side-part-name">{part.name}</div>
          <ul>
            {part.chapters.map((c) => {
              const live = c.status === 'live'
              const path = chapterPath(c)
              const active = pathname === path
              const done = live && isVisited(c.slug)
              return (
                <li key={c.slug}>
                  {live ? (
                    <NavLink to={path} onClick={onNavigate}
                      className={`side-link ${active ? 'is-active' : ''} ${done ? 'is-done' : ''}`}>
                      <span className="side-num">{done ? '✓' : c.num}</span>
                      <span className="side-title">{c.title}{c.core && <span className="side-star"> ★</span>}</span>
                    </NavLink>
                  ) : (
                    <span className="side-link is-planned" title="规划中">
                      <span className="side-num">{c.num}</span>
                      <span className="side-title">{c.title}</span>
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function Shell() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth > 900 : true)
  const contentRef = useRef<HTMLElement>(null)
  const { markVisited } = useProgress()

  useDocMeta()

  // 切换章节时回到页首（现在滚动容器是 .content，不是 window）
  useEffect(() => { contentRef.current?.scrollTo(0, 0) }, [pathname])

  // 访问某章 ≈ 读过它，记入进度（短延时，避免一闪而过的中转）
  useEffect(() => {
    if (!pathname.startsWith('/ch/')) return
    const slug = pathname.slice('/ch/'.length)
    if (!findChapter(slug)) return
    const t = window.setTimeout(() => markVisited(slug), 1200)
    return () => window.clearTimeout(t)
  }, [pathname, markVisited])

  const closeIfNarrow = () => {
    if (typeof window !== 'undefined' && window.innerWidth <= 900) setOpen(false)
  }

  return (
    <div className={`shell ${open ? 'nav-open' : 'nav-closed'}`}>
      <header className="topbar">
        <div className="topbar-left">
          <button className="nav-toggle" onClick={() => setOpen((o) => !o)}
            aria-label="切换全局导航" aria-expanded={open}>
            <span className="nav-toggle-icon">{open ? '✕' : '☰'}</span>
          </button>
          <Link to="/" className="wordmark">
            线性代数 <span className="arrow">→</span> 注意力
          </Link>
        </div>
        <nav className="topnav">
          <SearchPalette />
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>大纲</NavLink>
          <a className="star-link" href="https://github.com/tao-hpu/linalg-to-attention"
            target="_blank" rel="noreferrer">★ GitHub</a>
        </nav>
      </header>

      <div className="body">
        {open && <Sidebar onNavigate={closeIfNarrow} />}
        {open && <div className="nav-scrim" onClick={() => setOpen(false)} />}
        <main className="content" ref={contentRef}>
          <Outlet />
          <footer className="site-footer">
            <span>
              © 2026 <a href="https://fim.ai" target="_blank" rel="noreferrer">FIM Labs</a>
            </span>
            <span>线性代数 → 注意力 · 一个开源教学项目</span>
          </footer>
        </main>
      </div>
    </div>
  )
}

export function Layout() {
  return (
    <ProgressProvider>
      <Shell />
    </ProgressProvider>
  )
}
