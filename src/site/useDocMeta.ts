import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { findChapter } from '../chapters'

const SITE = '线性代数 → 注意力'
const DEFAULT_DESC =
  '可视化地，从一个向量一路搭到 Transformer 的注意力机制——手搓大模型之前的那门预科课。'

function setMeta(name: string, content: string) {
  let el = document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute('name', name)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

function setProperty(property: string, content: string) {
  let el = document.querySelector<HTMLMetaElement>(`meta[property="${property}"]`)
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute('property', property)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

/** 路由切换时把 <title> 和 description / og 标签换成当前章节的——SPA 也能有正确标题与分享卡。 */
export function useDocMeta() {
  const { pathname } = useLocation()
  useEffect(() => {
    const slug = pathname.startsWith('/ch/') ? pathname.slice('/ch/'.length) : ''
    const c = slug ? findChapter(slug) : undefined

    const title = c ? `${c.num} ${c.title} · ${SITE}` : `${SITE} · 手搓大模型前的预科课`
    const desc = c ? c.hook : DEFAULT_DESC

    document.title = title
    setMeta('description', desc)
    setProperty('og:title', title)
    setProperty('og:description', desc)
    setProperty('og:type', 'website')
    setProperty('og:url', `https://l2a.fim.ai${c ? `/ch/${slug}` : ''}`)
    setMeta('twitter:title', title)
    setMeta('twitter:description', desc)
  }, [pathname])
}
