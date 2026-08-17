// 生成社交分享卡 public/og.png（1200×630）。
//
// 为什么要有这个脚本：原来的 og.png 是手工产物，右下角「36 节」是烤进像素里的，
// 章节加到 38 之后没人想得起来重画，分享卡就一直在对外说错数字。
// 现在卡面文字全部从 chapters.ts 读，规则和 sitemap / 预渲染 meta 一致。
//
// 用法：pnpm og
// 依赖 rsvg-convert（brew install librsvg）或 ImageMagick，二者有其一即可。
// 构建流程不依赖它：og.png 已提交进仓库，prerender.mjs 只做一致性校验。

import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { tmpdir } from 'node:os'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

// ── 从 chapters.ts 取章节数（和 prerender.mjs 用同一套解析规则）──
export function readChapterCount() {
  const src = readFileSync(join(root, 'src/chapters.ts'), 'utf8')
  return src.split('\n').filter((l) => /\bslug:\s*'/.test(l)).length
}

const IKB = '#002fa7'
const INK = '#12161c'
const GREY = '#6b7280'
const FAINT = '#aab0ba'
const GRID = '#eef0f3'

// CJK 字族：rsvg 走 fontconfig，列出常见的中文无衬线，末位兜底 sans-serif
const SANS = "'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC','Microsoft YaHei',sans-serif"
const MONO = "'SF Mono','JetBrains Mono',Menlo,Consolas,monospace"

export function buildSvg(chapterCount) {
  const W = 1200
  const H = 630
  // 淡网格：呼应站点的瑞士风底纹
  const grid = [
    ...[200, 400, 600, 800, 1000].map(
      (x) => `<line x1="${x}" y1="0" x2="${x}" y2="${H}" stroke="${GRID}" stroke-width="1"/>`,
    ),
    ...[105, 210, 315, 420, 525].map(
      (y) => `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="${GRID}" stroke-width="1"/>`,
    ),
  ].join('')

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="#ffffff"/>
  ${grid}

  <!-- 顶部短蓝条 -->
  <rect x="80" y="82" width="128" height="8" fill="${IKB}"/>

  <!-- eyebrow -->
  <text x="80" y="150" font-family="${SANS}" font-size="27" fill="${GREY}" letter-spacing="3">交互式教程 · 从零开始</text>

  <!-- 主标题 -->
  <text x="80" y="305" font-family="${SANS}" font-size="104" font-weight="700" fill="${INK}" letter-spacing="2">线性代数</text>
  <text x="565" y="305" font-family="${SANS}" font-size="104" font-weight="700" fill="${IKB}">→</text>
  <text x="700" y="305" font-family="${SANS}" font-size="104" font-weight="700" fill="${INK}" letter-spacing="2">注意力</text>

  <!-- 副标题 -->
  <text x="80" y="405" font-family="${SANS}" font-size="36" fill="${INK}">可视化地，从一个向量一路搭到 Transformer 的注意力</text>
  <text x="80" y="466" font-family="${SANS}" font-size="29" fill="${GREY}">说人话：手搓大模型之前的那门预科课。</text>

  <!-- 页脚：域名 + 章节数 -->
  <rect x="80" y="536" width="14" height="14" fill="${IKB}"/>
  <text x="106" y="550" font-family="${MONO}" font-size="27" font-weight="700" fill="${IKB}">l2a.fim.ai</text>
  <text x="${W - 80}" y="550" font-family="${SANS}" font-size="25" fill="${FAINT}" text-anchor="end">${chapterCount} 节 · 全部可玩</text>
</svg>
`
}

function render(svg, outPath) {
  const dir = mkdtempSync(join(tmpdir(), 'l2a-og-'))
  const svgPath = join(dir, 'card.svg')
  writeFileSync(svgPath, svg)
  try {
    try {
      execFileSync('rsvg-convert', ['-w', '1200', '-h', '630', svgPath, '-o', outPath])
      return 'rsvg-convert'
    } catch {
      execFileSync('magick', [
        '-background', 'white', '-density', '144', svgPath,
        '-resize', '1200x630', outPath,
      ])
      return 'ImageMagick'
    }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

// 直接执行时才渲染；被 import 时只导出工具函数（prerender.mjs 用它做校验）
if (process.argv[1] && process.argv[1].endsWith('og.mjs')) {
  const n = readChapterCount()
  const out = join(root, 'public/og.png')
  let tool
  try {
    tool = render(buildSvg(n), out)
  } catch {
    console.error(
      '[og] 渲染失败：需要 rsvg-convert 或 ImageMagick。\n' +
      '     macOS: brew install librsvg   ·   Debian: apt install librsvg2-bin',
    )
    process.exit(1)
  }
  // 戳记：prerender.mjs 在构建期用它核对卡面数字有没有落后于 chapters.ts
  writeFileSync(
    join(root, 'scripts/og.stamp.json'),
    JSON.stringify({ chapters: n, note: '由 pnpm og 写入；prerender.mjs 校验它与 chapters.ts 是否一致' }, null, 2) + '\n',
  )
  console.log(`[og] 已生成 public/og.png（1200×630，${n} 节，用 ${tool}）`)
}
