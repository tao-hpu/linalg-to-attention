import { useState } from 'react'
import { Link } from 'react-router-dom'
import { parts, chapterPath, allChapters, firstLiveChapter, type Chapter } from '../chapters'
import { useProgress } from '../site/progress'

function ChapterRow({ c, done }: { c: Chapter; done: boolean }) {
  const live = c.status === 'live'
  const inner = (
    <>
      <span className="row-num">{done ? '✓' : c.num}</span>
      <span className="row-main">
        <span className="row-title">
          {c.title}
          {c.core && <span className="row-star" title="通往注意力的主线">★</span>}
        </span>
        <span className="row-hook">{c.hook}</span>
        {c.prereq && <span className="row-prereq">前置 · {c.prereq}</span>}
      </span>
      <span className="row-bridge">→ {c.bridge}</span>
      <span className={`row-status ${live ? 'is-live' : 'is-planned'}`}>
        {live ? (done ? '已读' : '可玩') : '规划中'}
      </span>
    </>
  )
  return live
    ? <Link to={chapterPath(c)} className={`ch-row ch-row--live ${done ? 'is-done' : ''}`}>{inner}</Link>
    : <div className="ch-row ch-row--planned">{inner}</div>
}

export function Home() {
  const { visited, isVisited, reset } = useProgress()
  const [coreOnly, setCoreOnly] = useState(false)
  const liveCount = allChapters.filter((c) => c.status === 'live').length
  const doneCount = allChapters.filter((c) => c.status === 'live' && visited.has(c.slug)).length
  const start = firstLiveChapter()
  const pct = liveCount ? Math.round((doneCount / liveCount) * 100) : 0

  return (
    <div className="home">
      <section className="hero">
        <div className="kicker">交互式教程 · 从零开始</div>
        <h1>线性代数 <span className="arrow">→</span> 注意力</h1>
        <p className="tagline">可视化地，从一个向量一路搭到 Transformer 的注意力机制。</p>
        <p className="tagline-aside">说人话：<strong>手搓大模型之前的那门预科课</strong> 😎</p>
        <p className="hero-lede">
          这不是又一份「看动画」的演示，也不是一摞「给你代码」的清单。
          每一页只回答<strong>一个「为什么」</strong>——你拖动滑块、亲手改参数，看几何当场变化，
          然后页底一句话把它焊到 LLM 上：<em>这就是注意力里的那一步</em>。
          风格对标「从零实现 GPT」，只不过我们从更底层的数学搭起。
        </p>
        <p className="outcome">
          <strong>学完你能：</strong>看懂注意力的每一步，并在之后碰 LLM 的
          <em>训练 / 微调 / 推理</em>时，知道每个环节背后的数学在做什么——而不是只会调包。
        </p>
      </section>

      <section className="how">
        <div className="how-item">
          <span className="how-n">01</span>
          <p><strong>为什么驱动</strong><br />每页一个真问题，不是定义堆砌。</p>
        </div>
        <div className="how-item">
          <span className="how-n">02</span>
          <p><strong>拖一下就懂</strong><br />几何随参数实时变，不靠想象。</p>
        </div>
        <div className="how-item">
          <span className="how-n">03</span>
          <p><strong>焊到 LLM</strong><br />每个概念都接上 Transformer 的某个零件。</p>
        </div>
      </section>

      <section className="outline">
        <div className="outline-head">
          <h2>大纲</h2>
          <span className="outline-meta">{allChapters.length} 节 · 已上线 {liveCount} · <span className="row-star">★</span> 通往注意力的主线</span>
        </div>

        <div className="progress-bar-wrap">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="progress-meta">
            <span>已读 <strong>{doneCount}</strong> / {liveCount}（{pct}%）</span>
            <span className="progress-actions">
              <button
                className={`chip ${coreOnly ? 'is-on' : ''}`}
                onClick={() => setCoreOnly((v) => !v)}
              >
                {coreOnly ? '✓ 只看主线 ★' : '只看主线 ★'}
              </button>
              {doneCount > 0 && (
                <button className="chip chip--ghost" onClick={reset}>清除进度</button>
              )}
            </span>
          </div>
        </div>

        {parts.map((part) => {
          const chapters = coreOnly ? part.chapters.filter((c) => c.core) : part.chapters
          if (chapters.length === 0) return null
          return (
            <div className="part" key={part.name}>
              <div className="part-head">
                <h3>{part.name}</h3>
                <p>{part.blurb}</p>
              </div>
              <div className="ch-list">
                {chapters.map((c) => (
                  <ChapterRow key={c.slug} c={c} done={c.status === 'live' && isVisited(c.slug)} />
                ))}
              </div>
            </div>
          )
        })}
      </section>

      {start && (
        <section className="start">
          <p>不知道从哪开始？直接玩第一个上线的章节：</p>
          <Link to={chapterPath(start)} className="start-btn">
            {start.num} · {start.title} <span className="arrow">→</span>
          </Link>
        </section>
      )}

      <section className="star-cta">
        <p>
          觉得有用？去 GitHub 点个 ★ Star，是对这个项目最实在的鼓励——
          也方便你之后回来追更后面的章节。
        </p>
        <a className="star-btn" href="https://github.com/tao-hpu/linalg-to-attention"
          target="_blank" rel="noreferrer">★ 在 GitHub 上 Star</a>
      </section>
    </div>
  )
}
