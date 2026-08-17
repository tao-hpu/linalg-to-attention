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

function setCanonical(href: string) {
  let el = document.querySelector<HTMLLinkElement>('link[rel="canonical"]')
  if (!el) {
    el = document.createElement('link')
    el.setAttribute('rel', 'canonical')
    document.head.appendChild(el)
  }
  el.setAttribute('href', href)
}

/** 路由切换时把 <title> 和 description / og / canonical 换成当前章节的——SPA 也能有正确标题与分享卡。 */
export function useDocMeta() {
  const { pathname } = useLocation()
  useEffect(() => {
    const slug = pathname.startsWith('/ch/') ? pathname.slice('/ch/'.length) : ''
    const c = slug ? findChapter(slug) : undefined
    // 既不是首页、也没匹配到章节 → 404，别让它顶着上一页的 title 和 canonical。
    const isNotFound = pathname !== '/' && !c

    const title = isNotFound
      ? `没有这一节 · ${SITE}`
      : c
        ? `${c.num} ${c.title} · ${SITE}`
        : `${SITE} · 手搓大模型前的预科课`
    const desc = c ? c.hook : DEFAULT_DESC
    const url = `https://l2a.fim.ai${c ? `/ch/${slug}` : ''}`

    document.title = title
    setMeta('description', desc)
    setProperty('og:title', title)
    setProperty('og:description', desc)
    setProperty('og:type', 'website')
    setProperty('og:url', url)
    setMeta('twitter:title', title)
    setMeta('twitter:description', desc)
    // canonical 之前只在预渲染的 HTML 里正确，客户端跳转后会一直停在进站那一页。
    setCanonical(url)
    // 404 不该被收录
    setMeta('robots', isNotFound ? 'noindex' : 'index,follow')
  }, [pathname])
}
