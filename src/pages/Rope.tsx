import { useState } from 'react'
import { ChapterShell, Bridge } from '../components/ChapterShell'
import { VectorCanvas } from '../components/VectorCanvas'
import { CodeBlock } from '../components/CodeBlock'
import { dot, norm, angleBetween, degrees, fmt, type V } from '../vec'

const IKB = '#002fa7'
const RUST = '#c75b39'
const GREY = '#9aa1a9'

// ── 固定基向量 q₀、k₀（长度相近、初始有夹角） ────────────────────────────────
// |q₀| ≈ 2.09，|k₀| ≈ 2.13，初始夹角 ≈ 32°。位置编码不动它们的长度，只转相位。
const Q0: V = { x: 2.0, y: 0.6 }
const K0: V = { x: 1.4, y: 1.6 }

const D2R = Math.PI / 180

// 2D 旋转：R(α)·p，α 为弧度。这正是第 17 节的旋转矩阵 [[cosα,−sinα],[sinα,cosα]]。
function rot(p: V, aRad: number): V {
  const c = Math.cos(aRad)
  const s = Math.sin(aRad)
  return { x: p.x * c - p.y * s, y: p.x * s + p.y * c }
}

// 位置 mm 的 q 与位置 nn 的 k，旋转后做点积（诚实地用旋转后的向量算）。
function scorePair(mm: number, nn: number, thetaDeg: number): number {
  const q = rot(Q0, mm * thetaDeg * D2R)
  const k = rot(K0, nn * thetaDeg * D2R)
  return dot(q, k)
}

// 闭式：分数只依赖相对距离 d。score(d) = |q₀||k₀|·cos(原夹角 + d·θ)。
const SCORE_MAX = norm(Q0) * norm(K0) // ≈ 4.44
const A0_RAD = Math.atan2(K0.y, K0.x) - Math.atan2(Q0.y, Q0.x) // q₀→k₀ 的有符号夹角
function scoreClosed(d: number, thetaDeg: number): number {
  return SCORE_MAX * Math.cos(A0_RAD + d * thetaDeg * D2R)
}

// ── 分数随相对距离 d=n−m 振荡的折线图 ──────────────────────────────────────────
const CW = 540
const CH = 170
const PADX = 38
const PADY = 24
const D_MIN = -8
const D_MAX = 8

function cxFor(d: number): number {
  return PADX + ((d - D_MIN) / (D_MAX - D_MIN)) * (CW - 2 * PADX)
}
function cyFor(s: number): number {
  return CH / 2 - (s / SCORE_MAX) * (CH / 2 - PADY)
}

function ScoreChart({ rel, thetaDeg }: { rel: number; thetaDeg: number }) {
  const pts: string[] = []
  for (let d = D_MIN; d <= D_MAX + 1e-9; d += 0.5) {
    pts.push(`${cxFor(d).toFixed(1)},${cyFor(scoreClosed(d, thetaDeg)).toFixed(1)}`)
  }
  const curY = cyFor(scoreClosed(rel, thetaDeg))
  const curX = cxFor(rel)

  return (
    <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', maxWidth: CW, display: 'block' }}
      aria-label="注意力分数随相对距离 d = n − m 振荡">
      {/* 零分横轴 */}
      <line x1={PADX} y1={CH / 2} x2={CW - PADX} y2={CH / 2} stroke="#d7dade" strokeWidth={1} />
      {/* d = 0 竖线 */}
      <line x1={cxFor(0)} y1={PADY} x2={cxFor(0)} y2={CH - PADY} stroke="#eceef1" strokeWidth={1} />
      {/* 整数刻度 */}
      {Array.from({ length: D_MAX - D_MIN + 1 }, (_, i) => D_MIN + i).map((d) => (
        <g key={d}>
          <line x1={cxFor(d)} y1={CH / 2 - 3} x2={cxFor(d)} y2={CH / 2 + 3} stroke="#bcc1c7" strokeWidth={1} />
          {d % 2 === 0 && (
            <text x={cxFor(d)} y={CH - 6} textAnchor="middle" fontSize={10} fill="#9aa1a9" fontFamily="monospace">
              {d}
            </text>
          )}
        </g>
      ))}
      {/* 振荡曲线 */}
      <polyline points={pts.join(' ')} fill="none" stroke={IKB} strokeWidth={2} strokeLinejoin="round" />
      {/* 当前 d 的竖直指示与点 */}
      <line x1={curX} y1={PADY} x2={curX} y2={CH - PADY} stroke={RUST} strokeWidth={1.5} strokeDasharray="4 3" />
      <circle cx={curX} cy={curY} r={5} fill={RUST} />
      <text x={curX} y={PADY - 6} textAnchor="middle" fontSize={11} fill={RUST} fontWeight={700} fontFamily="monospace">
        d = {rel}
      </text>
      {/* 轴标 */}
      <text x={CW - PADX} y={CH / 2 - 8} textAnchor="end" fontSize={10} fill="#9aa1a9" fontFamily="monospace">
        相对距离 d = n − m →
      </text>
    </svg>
  )
}

// ── apply_rope 代码 ───────────────────────────────────────────────────────────
const SNIPPET = `import numpy as np

def apply_rope(x, pos, base=10000.0):
    """
    把一个 d 维向量按位置 pos 做 RoPE 旋转。
    x   : (d,)  —— Query 或 Key 向量，d 为偶数
    pos : int   —— 这个 token 在序列里的位置
    返回旋转后的向量 (d,)
    """
    d = x.shape[0]
    # d/2 个频率：高频对管短距离、低频对管长距离（像一组不同刻度的时钟）
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)        # θ_i = base^(-2i/d)
    ang = pos * theta                     # 每一对的旋转角 = 位置 × 该对频率
    cos, sin = np.cos(ang), np.sin(ang)

    x_even, x_odd = x[0::2], x[1::2]      # 把向量拆成 d/2 个二维对
    out = np.empty_like(x)
    # 每一对 (x_even, x_odd) 用 2D 旋转矩阵 R(ang) 旋转（第 17 节）
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out

# 关键性质：位置 m 的 q 与位置 n 的 k，旋转后点积只依赖相对距离 (n − m)
d = 8
q = np.random.randn(d)
k = np.random.randn(d)
s1 = apply_rope(q, 0) @ apply_rope(k, 2)   # m=0, n=2 → 相对距离 2
s2 = apply_rope(q, 3) @ apply_rope(k, 5)   # m=3, n=5 → 相对距离 2
print(round(float(s1), 6), round(float(s2), 6))
print(np.allclose(s1, s2))                 # True —— 绝对位置消失，只剩相对`

// 给定相对距离 rel，列出几组绝对位置都不同、但 n−m 相同的 (m,n)。
function samePairs(rel: number): Array<[number, number]> {
  const out: Array<[number, number]> = []
  const lo = Math.max(0, -rel)
  const hi = Math.min(8, 8 - rel)
  for (let mm = lo; mm <= hi && out.length < 3; mm += Math.max(1, Math.ceil((hi - lo) / 2) || 1)) {
    out.push([mm, mm + rel])
  }
  // 去重并保证至少有一项
  const seen = new Set<string>()
  const uniq = out.filter(([a, b]) => {
    const key = `${a},${b}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
  return uniq.length ? uniq : [[lo, lo + rel]]
}

export function Rope() {
  const [m, setM] = useState(0)
  const [n, setN] = useState(2)
  const [thetaDeg, setThetaDeg] = useState(30)
  const [lock, setLock] = useState(false)

  // 锁定相对距离时，拖 m 滑块会带着 n 一起平移，保持 n−m 不变。
  const onM = (val: number) => {
    if (lock) {
      const rel = n - m
      let nm = val
      let nn = val + rel
      if (nn < 0) { nn = 0; nm = -rel }
      if (nn > 8) { nn = 8; nm = 8 - rel }
      nm = Math.max(0, Math.min(8, nm))
      setM(nm)
      setN(nn)
    } else {
      setM(val)
    }
  }
  const onN = (val: number) => {
    if (lock) {
      const rel = n - m
      let nn = val
      let nm = val - rel
      if (nm < 0) { nm = 0; nn = rel }
      if (nm > 8) { nm = 8; nn = 8 + rel }
      nn = Math.max(0, Math.min(8, nn))
      setM(nm)
      setN(nn)
    } else {
      setN(val)
    }
  }

  const qp = rot(Q0, m * thetaDeg * D2R)
  const kp = rot(K0, n * thetaDeg * D2R)
  const rel = n - m
  const angDeg = degrees(angleBetween(qp, kp)) // 几何夹角 0..180
  const score = dot(qp, kp)

  const pairs = samePairs(rel)

  const lede = (
    <>
      自注意力有个让人不安的盲点：它<strong>置换不变（permutation-invariant）</strong>——
      把句子里的 token 顺序打乱，<code>QKᵀ</code> 算出的注意力分数一个不差，
      模型分不清「猫追狗」和「狗追猫」。必须把<strong>位置信息</strong>注入进去。
      原始 Transformer 给 embedding <em>加</em>一个正弦位置向量；
      <strong>RoPE</strong> 换了个更干净的思路：按 token 的位置，把它的 Query / Key
      向量<strong>旋转</strong>一个角度——位置 m 的向量乘上旋转矩阵 <code>R(mθ)</code>，
      也就是第 17 节那个 det = +1、只转不拉的 2D 旋转。下面你亲手转动 q′、k′，
      会发现一件神奇的事：注意力分数只认<strong>相对距离</strong>。
    </>
  )

  return (
    <ChapterShell slug="rope" part="第八部分 · 合成：亲手拼出注意力" lede={lede}>
      {/* ── 痛点：置换不变 ── */}
      <section className="note">
        <p>
          <strong>为什么非加不可？</strong> 注意力分数 <code>q·k</code> 只看两个 token
          的内容向量，不看它们排第几。没有位置信号，「猫 坐 在 垫子 上」和
          「上 垫子 在 坐 猫」对模型完全等价。语言是有序的，所以位置必须进到点积里。
        </p>
      </section>

      {/* ── 核心思想 ── */}
      <section className="readouts">
        <h2 className="sec-h">RoPE 的招法：转，而不是加</h2>
        <p style={{ color: '#444', fontSize: '0.92rem', lineHeight: 1.7 }}>
          位置 m 的 query 旋转 <code>q′ = R(mθ)·q₀</code>，位置 n 的 key 旋转
          <code> k′ = R(nθ)·k₀</code>。两者点积时（旋转是正交变换，<code>R(α)ᵀR(α)=I</code>）：
        </p>
        <div style={{
          fontFamily: 'monospace', fontSize: '0.92rem', lineHeight: 2.1,
          color: '#222', background: '#f6f8ff', border: `1px solid rgba(0,47,167,0.18)`,
          borderRadius: 6, padding: '12px 18px', overflowX: 'auto', margin: '0.8rem 0',
        }}>
          q′·k′ = (R(mθ)q)·(R(nθ)k) = qᵀ R(mθ)ᵀR(nθ) k = qᵀ R((n−m)θ) k
        </div>
        <p style={{ color: '#444', fontSize: '0.92rem', lineHeight: 1.7 }}>
          绝对编号 m、n 全消掉了，只剩 <strong>(n − m)</strong>。几何上：旋转后两向量的夹角
          = 原夹角 + (n−m)θ，于是
          <code> q′·k′ = |q||k|·cos(原夹角 + (n−m)θ)</code>——
          <strong>注意力分数只是相对距离的函数</strong>。这就是 RoPE 把相对位置
          天然编码进点积的全部秘密。
        </p>
      </section>

      {/* ── 控制区 ── */}
      <section className="controls">
        <div className="control">
          <div className="control-head">
            <span className="slot-tag">m · query 位置</span>
          </div>
          <label className="slider-row">
            <input type="range" min={0} max={8} step={1} value={m}
              onChange={(e) => onM(Number(e.target.value))} />
            <span className="param-val" style={{ color: IKB }}>{m}</span>
          </label>
        </div>

        <div className="control">
          <div className="control-head">
            <span className="slot-tag" style={{ background: RUST }}>n · key 位置</span>
          </div>
          <label className="slider-row">
            <input type="range" min={0} max={8} step={1} value={n}
              onChange={(e) => onN(Number(e.target.value))} />
            <span className="param-val" style={{ color: RUST }}>{n}</span>
          </label>
        </div>

        <div className="control">
          <div className="control-head">
            <span className="slot-tag">θ · 基础角度</span>
          </div>
          <label className="slider-row">
            <input type="range" min={15} max={45} step={1} value={thetaDeg}
              onChange={(e) => setThetaDeg(Number(e.target.value))} />
            <span className="param-val">{thetaDeg}°</span>
          </label>
          <label style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem',
            marginTop: '0.7rem', cursor: 'pointer', fontSize: '0.86rem',
          }}>
            <input type="checkbox" checked={lock} onChange={(e) => setLock(e.target.checked)}
              style={{ accentColor: IKB, width: 15, height: 15 }} />
            <span>锁定相对距离 n − m（拖任一滑块，两个位置一起平移）</span>
          </label>
        </div>
      </section>

      {/* ── 画布 + 读数 ── */}
      <section className="stage" style={{ alignItems: 'flex-start', gap: '2rem' }}>
        <div>
          <VectorCanvas
            size={360}
            range={4}
            snap={0}
            vectors={[
              { id: 'q', x: qp.x, y: qp.y, color: IKB, label: "q′" },
              { id: 'k', x: kp.x, y: kp.y, color: RUST, label: "k′" },
            ]}
          >
            {({ sx, sy }) => (
              <g>
                {/* 原始 q₀、k₀（未旋转）以虚线灰示意 */}
                <line x1={sx(0)} y1={sy(0)} x2={sx(Q0.x)} y2={sy(Q0.y)}
                  stroke={GREY} strokeWidth={1.5} strokeDasharray="4 4" />
                <line x1={sx(0)} y1={sy(0)} x2={sx(K0.x)} y2={sy(K0.y)}
                  stroke={GREY} strokeWidth={1.5} strokeDasharray="4 4" />
                <text x={sx(Q0.x) + 6} y={sy(Q0.y) + 4} fill={GREY} fontSize={11} fontFamily="monospace">q₀</text>
                <text x={sx(K0.x) + 6} y={sy(K0.y) + 4} fill={GREY} fontSize={11} fontFamily="monospace">k₀</text>
              </g>
            )}
          </VectorCanvas>
          <p style={{ margin: '0.3rem 0 0', fontSize: '0.76rem', color: '#aaa', textAlign: 'center' }}>
            虚线 = 原始 q₀、k₀ &nbsp;|&nbsp; 实线 = 旋转后的 q′ = R(mθ)q₀、k′ = R(nθ)k₀
          </p>
        </div>

        <div style={{ minWidth: 220, fontFamily: 'monospace', fontSize: '0.9rem', lineHeight: 2 }}>
          <div>
            <span style={{ color: '#888' }}>相对距离 n − m = </span>
            <strong style={{ color: RUST, fontSize: '1.05rem' }}>{rel}</strong>
          </div>
          <div>
            <span style={{ color: '#888' }}>q′、k′ 夹角 = </span>
            <strong>{fmt(angDeg, 1)}°</strong>
          </div>
          <div style={{ marginTop: '0.4rem' }}>
            <span style={{ color: '#888' }}>点积分数 q′·k′ = </span>
            <strong style={{ color: IKB, fontSize: '1.1rem' }}>{fmt(score, 3)}</strong>
          </div>
          <div style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: '#999', lineHeight: 1.6 }}>
            = |q||k|·cos(原夹角 32.1° + {rel}×{thetaDeg}°)
            <br />= {fmt(SCORE_MAX, 2)} × cos({fmt(32.11 + rel * thetaDeg, 1)}°)
          </div>
        </div>
      </section>

      {/* ── 同相对距离 = 同分数 ── */}
      <section className="readouts">
        <h2 className="sec-h">关键演示：平移不改分数</h2>
        <p style={{ color: '#444', fontSize: '0.9rem', lineHeight: 1.7 }}>
          下面这几组 (m, n) <strong>绝对位置各不相同</strong>，但相对距离
          <code> n − m = {rel}</code> 都一样。看右边的分数——它们
          <strong>逐位相等</strong>。把整段话整体往后挪几个词（m、n 同时 +k），
          注意力分数纹丝不动。{lock ? '（你已开启锁定，拖滑块就是在做这件事。）' : '试试打开上面的「锁定相对距离」开关，拖滑块亲自验证。'}
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 13, marginTop: '0.4rem' }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${IKB}` }}>
                <th style={{ padding: '4px 16px', textAlign: 'left', color: IKB }}>m（query 位置）</th>
                <th style={{ padding: '4px 16px', textAlign: 'left', color: RUST }}>n（key 位置）</th>
                <th style={{ padding: '4px 16px', textAlign: 'left', color: '#666' }}>n − m</th>
                <th style={{ padding: '4px 16px', textAlign: 'right', color: '#222' }}>q′·k′</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map(([mm, nn]) => (
                <tr key={`${mm}-${nn}`}>
                  <td style={{ padding: '4px 16px', fontFamily: 'monospace', color: IKB }}>{mm}</td>
                  <td style={{ padding: '4px 16px', fontFamily: 'monospace', color: RUST }}>{nn}</td>
                  <td style={{ padding: '4px 16px', fontFamily: 'monospace', color: '#666' }}>{nn - mm}</td>
                  <td style={{ padding: '4px 16px', fontFamily: 'monospace', textAlign: 'right', fontWeight: 700, color: '#1a7f4e' }}>
                    {fmt(scorePair(mm, nn, thetaDeg), 4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '1.4rem' }}>
          <p style={{ margin: '0 0 0.5rem', fontSize: '0.82rem', color: '#555' }}>
            分数随相对距离 d = n − m 振荡——它<strong>只是 d 的函数</strong>，与绝对位置无关：
          </p>
          <ScoreChart rel={rel} thetaDeg={thetaDeg} />
        </div>
      </section>

      {/* ── 高维补充 ── */}
      <section className="note">
        <p>
          <strong>高维怎么办？</strong> 真实 RoPE 把 d 维向量拆成 <strong>d/2 个二维对</strong>，
          每一对用一个不同频率 <code>θ_i = base^(−2i/d)</code>（base 常取 10000）旋转：
          高频对（θ 大）转得快，分辨相邻几个词；低频对（θ 小）转得慢，刻画长程关系。
          像一组刻度从秒针到时针的时钟，合起来就能在很宽的距离范围上编码位置。
          下面的 <code>apply_rope</code> 就是这件事的最小实现。
        </p>
      </section>

      <section className="codeblock-wrap">
        <h2 className="sec-h">看代码：apply_rope（按位置旋转、按 d/2 个频率分组）</h2>
        <CodeBlock code={SNIPPET} language="python" title="rope.py" />
      </section>

      <Bridge>
        <p>
          <strong>RoPE 是当今主流 LLM 的标配位置编码</strong>——LLaMA、Qwen、GPT-NeoX
          等几乎都用它。原因正是这一页演示的：旋转是正交变换（第 17 节），
          <strong>不改 Q/K 的范数、只改相位</strong>，是一种「干净」的位置注入；
          det = +1（第 14 节）保证它只转不翻、数值无损可逆。
        </p>
        <p>
          相对位置 <code>n − m</code> 直接落在点积里，这是 RoPE 能较好<strong>外推到更长上下文</strong>
          （配合 NTK / YaRN 等技巧）的基础：训练时没见过的绝对位置，
          只要相对距离在分布内，分数依然合理。
        </p>
        <p>
          下一节（第 34 节）把它和注意力、残差、归一化、FFN 装在一起，
          组装成一个完整的 <strong>Transformer Block</strong>。
        </p>
      </Bridge>
    </ChapterShell>
  )
}
