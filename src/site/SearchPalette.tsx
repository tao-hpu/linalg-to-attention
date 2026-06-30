import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate } from 'react-router-dom'
import { allChapters, chapterPath, type Chapter } from '../chapters'

// ⌘K / Ctrl-K 全站快速跳转：按标题、为什么、bridge 模糊匹配。
function match(c: Chapter, q: string): boolean {
  const hay = `${c.num} ${c.title} ${c.hook} ${c.bridge} ${c.slug}`.toLowerCase()
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((t) => hay.includes(t))
}

export function SearchPalette() {
  const [open, setOpen] = useState(false)
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const results = useMemo(() => {
    const live = allChapters.filter((c) => c.status === 'live')
    return q.trim() ? live.filter((c) => match(c, q.trim())) : live
  }, [q])

  // 全局快捷键：⌘K / Ctrl-K 开，Esc 关
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
      } else if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (open) {
      setQ('')
      setActive(0)
      // 等 DOM 挂上再聚焦
      const t = window.setTimeout(() => inputRef.current?.focus(), 0)
      return () => window.clearTimeout(t)
    }
  }, [open])

  useEffect(() => setActive(0), [q])

  const go = (c?: Chapter) => {
    if (!c) return
    setOpen(false)
    navigate(chapterPath(c))
  }

  const onListKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((a) => Math.min(a + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      go(results[active])
    }
  }

  return (
    <>
      <button className="search-trigger" onClick={() => setOpen(true)} aria-label="搜索章节">
        <span className="search-trigger-icon">⌕</span>
        <span className="search-trigger-text">搜索章节</span>
        <kbd className="search-kbd">⌘K</kbd>
      </button>
      {open && createPortal(
        <div className="search-overlay" onClick={() => setOpen(false)}>
          <div className="search-box" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="搜索章节">
        <input
          ref={inputRef}
          className="search-input"
          placeholder="搜章节：内积、softmax、LoRA、秩…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={onListKey}
        />
        <ul className="search-results">
          {results.length === 0 && <li className="search-empty">没有匹配的章节</li>}
          {results.map((c, i) => (
            <li key={c.slug}>
              <button
                className={`search-item ${i === active ? 'is-active' : ''}`}
                onMouseEnter={() => setActive(i)}
                onClick={() => go(c)}
              >
                <span className="search-num">{c.num}</span>
                <span className="search-main">
                  <span className="search-title">
                    {c.title}
                    {c.core && <span className="search-star"> ★</span>}
                  </span>
                  <span className="search-hook">{c.hook}</span>
                </span>
                <span className="search-bridge">→ {c.bridge}</span>
              </button>
            </li>
          ))}
        </ul>
          <div className="search-foot">
            <span><kbd>↑</kbd><kbd>↓</kbd> 选择</span>
            <span><kbd>↵</kbd> 打开</span>
            <span><kbd>esc</kbd> 关闭</span>
          </div>
        </div>
      </div>,
      document.body,
      )}
    </>
  )
}
