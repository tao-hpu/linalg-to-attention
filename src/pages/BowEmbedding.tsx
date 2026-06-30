import { useState } from 'react'
import { VectorCanvas } from '../components/VectorCanvas'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { CodeBlock } from '../components/CodeBlock'
import { cosineSimilarity, fmt, type V } from '../vec'

const IKB = '#002fa7'
const TEAL = '#00b3a4'
const RUST = '#b23a00'

// 手挑的 2D「embedding」坐标：动物（猫/狗/老虎）方向相近成一簇，
// 交通工具（汽车/卡车/飞机）在相反方向另成一簇。
// 于是「意思相近 → 向量相近 → cosine 大」这件事在图上一眼可见。
interface Word extends V {
  id: string
  label: string
  group: 'animal' | 'vehicle'
}

const INITIAL_VOCAB: Word[] = [
  { id: 'cat', label: '猫', x: 2.0, y: 3.2, group: 'animal' },
  { id: 'dog', label: '狗', x: 2.6, y: 2.6, group: 'animal' },
  { id: 'tiger', label: '老虎', x: 3.1, y: 3.4, group: 'animal' },
  { id: 'car', label: '汽车', x: -3.0, y: -2.2, group: 'vehicle' },
  { id: 'truck', label: '卡车', x: -2.4, y: -2.6, group: 'vehicle' },
  { id: 'plane', label: '飞机', x: -3.3, y: -1.2, group: 'vehicle' },
]

const groupColor = (g: Word['group']) => (g === 'animal' ? TEAL : RUST)

// 画布参数（与 <VectorCanvas size range> 保持一致），用于把鼠标/触摸坐标换算回数学坐标。
const SIZE = 380
const RANGE = 5
const CENTER = SIZE / 2
const UNIT = SIZE / 2 / RANGE
const SNAP = 0.1

// 把屏幕坐标换算成画布里的数学坐标。
// 按 rect.width 归一化 → 即使 SVG 被 CSS 缩放（移动端）也准确；并夹紧到可视范围、按 SNAP 吸附。
function clientToMath(svg: SVGSVGElement, clientX: number, clientY: number): V {
  const rect = svg.getBoundingClientRect()
  const px = ((clientX - rect.left) / rect.width) * SIZE
  const py = ((clientY - rect.top) / rect.height) * SIZE
  let x = (px - CENTER) / UNIT
  let y = (CENTER - py) / UNIT
  x = Math.max(-RANGE, Math.min(RANGE, x))
  y = Math.max(-RANGE, Math.min(RANGE, y))
  x = Math.round(x / SNAP) * SNAP
  y = Math.round(y / SNAP) * SNAP
  return { x, y }
}

const SNIPPET = `// 词表：6 个词，每个词有一个固定的下标
const VOCAB = ['猫', '狗', '老虎', '汽车', '卡车', '飞机']

// ① one-hot：除了自己那一位是 1，其余全是 0
const oneHot = (w: string) => VOCAB.map((x) => (x === w ? 1 : 0))
oneHot('猫')  // [1, 0, 0, 0, 0, 0]
oneHot('狗')  // [0, 1, 0, 0, 0, 0]

// ② dense embedding：从数据里学出来的稠密向量（这里手挑成 2D）
const EMB: Record<string, [number, number]> = {
  猫:   [2.0,  3.2], 狗:   [2.6,  2.6], 老虎: [3.1,  3.4],
  汽车: [-3.0, -2.2], 卡车: [-2.4, -2.6], 飞机: [-3.3, -1.2],
}

const dot = (a: number[], b: number[]) => a.reduce((s, ai, i) => s + ai * b[i], 0)
const norm = (a: number[]) => Math.hypot(...a)
const cosine = (a: number[], b: number[]) => dot(a, b) / (norm(a) * norm(b))

// 关键对比：
cosine(oneHot('猫'), oneHot('狗'))  // = 0  → one-hot 说「猫 和 狗 毫无关系」
cosine(EMB['猫'], EMB['狗'])        // ≈ 0.97 → embedding 说「猫 和 狗 很像」`

export function BowEmbedding() {
  const [selId, setSelId] = useState('cat')
  const [words, setWords] = useState<Word[]>(INITIAL_VOCAB)
  const [dragId, setDragId] = useState<string | null>(null)

  const selected = words.find((w) => w.id === selId)!
  const selIdx = words.findIndex((w) => w.id === selId)

  // 对所有其它词算 cosine，按相似度从高到低排序
  const ranked = words
    .filter((w) => w.id !== selId)
    .map((w) => ({ word: w, cos: cosineSimilarity(selected, w) }))
    .sort((a, b) => b.cos - a.cos)

  const nearest = ranked[0]

  const moveWord = (id: string, x: number, y: number) =>
    setWords((ws) => ws.map((w) => (w.id === id ? { ...w, x, y } : w)))

  const resetWords = () => setWords(INITIAL_VOCAB)
  const moved = words.some((w, i) => w.x !== INITIAL_VOCAB[i].x || w.y !== INITIAL_VOCAB[i].y)

  return (
    <ChapterShell
      slug="bow-to-embedding"
      part="第一部分 · 向量：意义的载体"
      lede={
        <>
          在 NLP 课上你已经把文本变成过向量：one-hot、词袋（bag-of-words）、TF-IDF。
          它们都能用，但有个躲不开的毛病——<strong>任意两个不同的词都「互相垂直」</strong>，
          cosine 永远是 0，模型没法表达「猫 和 狗 相似」。
          这一节我们换成<strong>稠密 embedding</strong>：把每个词放进一个连续空间里，
          让<strong>意思相近的词，向量也相近</strong>。下面的词点都能<strong>拖动</strong>——
          亲手把一个词拖进/拖出某个簇，看 cosine 相似度实时翻转。
        </>
      }
    >
      <section className="vstage">
        <VectorCanvas vectors={[]} size={SIZE} range={RANGE}>
          {({ sx, sy }) => (
            <>
              {/* 选中词作为一个「箭头」：从原点出发的方向 */}
              <line
                x1={sx(0)} y1={sy(0)} x2={sx(selected.x)} y2={sy(selected.y)}
                stroke={IKB} strokeWidth={2.5} strokeLinecap="round"
              />
              {/* 选中词 → 最近邻 的虚线连接：cosine 最大的那一对 */}
              <line
                x1={sx(selected.x)} y1={sy(selected.y)}
                x2={sx(nearest.word.x)} y2={sy(nearest.word.y)}
                stroke={TEAL} strokeWidth={2} strokeDasharray="5 5"
              />
              {/* 所有词：可拖拽的圆点 + 标签，颜色按语义簇区分 */}
              {words.map((w) => {
                const isSel = w.id === selId
                const isNear = w.id === nearest.word.id
                const base = groupColor(w.group)
                return (
                  <g key={w.id} style={{ cursor: dragId === w.id ? 'grabbing' : 'grab' }}>
                    <circle
                      cx={sx(w.x)} cy={sy(w.y)}
                      r={isSel ? 8 : 6}
                      fill={isSel ? IKB : base}
                      stroke={isNear ? TEAL : '#ffffff'}
                      strokeWidth={isNear ? 3 : 2}
                      style={{ pointerEvents: 'none' }}
                    />
                    <text
                      x={sx(w.x) + 11} y={sy(w.y) - 9}
                      fill={isSel ? IKB : '#1b1f24'}
                      fontSize={isSel ? 15 : 13}
                      fontWeight={isSel ? 700 : 500}
                      style={{ pointerEvents: 'none', userSelect: 'none' }}
                    >
                      {w.label}
                    </text>
                    {/* 透明的大命中区，方便鼠标/手指抓取 */}
                    <circle
                      cx={sx(w.x)} cy={sy(w.y)} r={16}
                      fill="transparent"
                      style={{ touchAction: 'none' }}
                      onPointerDown={(e) => {
                        e.currentTarget.setPointerCapture(e.pointerId)
                        setSelId(w.id)
                        setDragId(w.id)
                      }}
                      onPointerMove={(e) => {
                        if (dragId !== w.id) return
                        const svg = e.currentTarget.ownerSVGElement
                        if (!svg) return
                        const m = clientToMath(svg, e.clientX, e.clientY)
                        moveWord(w.id, m.x, m.y)
                      }}
                      onPointerUp={(e) => {
                        e.currentTarget.releasePointerCapture(e.pointerId)
                        setDragId(null)
                      }}
                    />
                  </g>
                )
              })}
            </>
          )}
        </VectorCanvas>

        <div className="vpanel">
          <div className="vrow">
            <span>选中词</span>
            <code>
              <select
                value={selId}
                onChange={(e) => setSelId(e.target.value)}
                style={{
                  font: 'inherit', color: IKB, fontWeight: 700,
                  border: 'none', background: 'transparent', cursor: 'pointer',
                }}
              >
                {words.map((w) => (
                  <option key={w.id} value={w.id}>{w.label}</option>
                ))}
              </select>
            </code>
          </div>

          <div className="vrow">
            <span>稠密向量</span>
            <code>{selected.label} = ({fmt(selected.x)}, {fmt(selected.y)})</code>
          </div>

          <div className="vrow">
            <span>最近邻</span>
            <code style={{ color: TEAL, fontWeight: 700 }}>
              {nearest.word.label} · cosine = {fmt(nearest.cos)}
            </code>
          </div>

          <div className="vrow" style={{ alignItems: 'flex-start' }}>
            <span>全部 cosine</span>
            <code style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {ranked.map((r, i) => (
                <span key={r.word.id} style={{ color: i === 0 ? TEAL : undefined }}>
                  {r.word.label}：{fmt(r.cos)}{i === 0 ? '  ← 最近' : ''}
                </span>
              ))}
            </code>
          </div>

          <div className="vrow">
            <span>one-hot</span>
            <code>
              [{words.map((_, i) => (i === selIdx ? '1' : '0')).join(', ')}]
            </code>
          </div>

          <div className="vrow">
            <span>one-hot cosine</span>
            <code>与任何其它词 = 0</code>
          </div>

          {moved && (
            <div className="vrow" style={{ justifyContent: 'flex-end' }}>
              <button
                type="button"
                onClick={resetWords}
                style={{
                  font: 'inherit', fontSize: 13, fontWeight: 600,
                  color: IKB, background: 'transparent',
                  border: `1px solid ${IKB}`, borderRadius: 6,
                  padding: '4px 12px', cursor: 'pointer',
                }}
              >
                ↺ 复位
              </button>
            </div>
          )}

          <p className="vhint">
            <strong>直接拖动任意一个词点</strong>，右边的 cosine 会实时重算、最近邻排名随之翻转。
            试着把<strong>「猫」拖出动物簇、推向汽车那一带</strong>，看它的近邻怎么从「狗 / 老虎」变成「汽车 / 卡车」——
            这就是「意思相近 → 向量相近」。蓝线是选中词从原点出发的方向，绿色虚线连到它当前的最近邻。
          </p>
        </div>
      </section>

      <section className="note">
        <p>
          看出区别了吗？<strong>one-hot 把「猫」写成 <code>[1,0,0,0,0,0]</code>，
          「狗」写成 <code>[0,1,0,0,0,0]</code></strong>——两个向量没有任何重叠的非零位，
          点积是 0，所以 cosine 是 0。在 one-hot 的世界里，
          <strong>每两个不同的词都正交、都「同样不相关」</strong>：猫和狗的距离，跟猫和飞机的距离一模一样。
          它根本没有「相似」这个概念。
        </p>
        <p>
          稠密 embedding 把维度从「词表大小」压到几个连续轴上，
          于是猫、狗、老虎挤在同一个方向（cosine ≈ 1），
          汽车、卡车、飞机挤在另一个方向，两簇之间 cosine 变成负数。
          <strong>「意思相近 → 向量相近」第一次变得可计算</strong>——这就是从词袋走向词向量的全部意义。
        </p>
      </section>

      <Bridge>
        <p>
          这就是 LLM 的 <code>embedding</code> 层。真实的 embedding 不是 2D，而是几百到上千维，
          而且坐标不是手挑的，是<strong>从海量文本里训练出来的</strong>。
        </p>
        <p>
          但「意思相近 → 向量相近」这件事，和你眼前这张 2D 图<strong>一模一样</strong>。
          正因为相似度被编码进了几何位置，后面的<code>内积 = 相似度</code>、
          乃至<code>注意力（attention）</code>用一个词去「查询」其它词，才全都成立。
        </p>
      </Bridge>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：one-hot 两两正交，embedding 才有「相似」</h2>
        <CodeBlock code={SNIPPET} language="tsx" title="bow-to-embedding.ts" />
      </section>
    </ChapterShell>
  )
}
