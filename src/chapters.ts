// 全站唯一的章节清单 —— 首页大纲、顶栏、上一页/下一页都从这里读。
// 加新章节 = 往这里加一条 + 写对应页面组件 + 在 App.tsx 挂一条路由。
//
// 整条线索：从「一个向量」一路搭到「自注意力」，再落到「训练 / 微调 / 推理」。
// 两根支柱：几何(线性代数) + 概率(经典统计)，在注意力里合流，在 LLM 工程里落地。

export type ChapterStatus = 'live' | 'planned'

export interface Chapter {
  slug: string        // 路由：/ch/<slug>
  num: string         // 章号，如 "08"
  title: string
  hook: string        // 这一章回答的「为什么」
  bridge: string      // 「这就是 LLM 里的 ___」
  status: ChapterStatus
  core?: boolean      // 是否在「通往注意力」的最短主线上（首页打 ★）
}

export interface Part {
  name: string
  blurb: string
  chapters: Chapter[]
}

export const parts: Part[] = [
  {
    name: '第一部分 · 向量：意义的载体',
    blurb: '一串数字怎么就装下了「意义」。',
    chapters: [
      { slug: 'vectors', num: '01', title: '向量与坐标系', hook: '一串数字怎么表示「意义」？', bridge: '词嵌入 embedding', status: 'live', core: true },
      { slug: 'bow-to-embedding', num: '02', title: '从词袋到词向量', hook: '文本怎么变成向量？从 TF-IDF 到稠密 embedding', bridge: '稠密词向量、one-hot 的进化', status: 'live' },
      { slug: 'vector-arithmetic', num: '03', title: '向量的加减与缩放', hook: '语义能做算术吗？', bridge: 'king − man + woman', status: 'live' },
      { slug: 'dot-product', num: '04', title: '内积', hook: '两个向量「像不像」，怎么用一个数衡量？', bridge: '注意力打分 q·k', status: 'live', core: true },
      { slug: 'projection', num: '05', title: '投影', hook: '一个向量在某方向上「占多少」？', bridge: '信息的读出', status: 'live' },
      { slug: 'norms', num: '06', title: '范数与单位向量', hook: '怎么公平比较长短不一的向量？', bridge: 'cosine 相似度、缩放点积里的 √d', status: 'live' },
    ],
  },
  {
    name: '第二部分 · 矩阵：一个动作',
    blurb: '矩阵不是数表，是一次空间变换。',
    chapters: [
      { slug: 'matrix-as-transform', num: '07', title: '矩阵是变换', hook: '矩阵到底「做」了什么？', bridge: '线性层 = 一次空间变换', status: 'live', core: true },
      { slug: 'matrix-mult', num: '08', title: '矩阵乘法的几何', hook: '为什么换了顺序，结果就不一样？', bridge: '线性层与 Q·K·V 投影', status: 'live', core: true },
      { slug: 'matmul-views', num: '09', title: '矩阵乘法的四种视角', hook: '同一个乘法，四种「看法」', bridge: '看懂 attention 的不同写法', status: 'live' },
      { slug: 'transpose-shape', num: '10', title: '转置与形状', hook: 'QKᵀ 那个「ᵀ」为什么必须有？', bridge: '注意力分数矩阵、维度对齐', status: 'live', core: true },
      { slug: 'batching', num: '11', title: '批量与张量', hook: '一次处理一整句话怎么算？', bridge: 'batch、多 token 并行', status: 'live' },
    ],
  },
  {
    name: '第三部分 · 方阵的秘密',
    blurb: '行列式、逆、特征值——方阵里藏着的结构。',
    chapters: [
      { slug: 'determinant', num: '12', title: '行列式', hook: '一个变换会不会把信息「压扁」？', bridge: '可逆性、信息瓶颈', status: 'live' },
      { slug: 'rank', num: '13', title: '矩阵的秩', hook: '一个变换到底「留下」了几维？', bridge: '低秩、LoRA 里的秩 r', status: 'live', core: true },
      { slug: 'inverse', num: '14', title: '逆矩阵', hook: '变换能不能还原？', bridge: '为什么有些层不可逆', status: 'live' },
      { slug: 'eigen', num: '15', title: '特征值与特征向量', hook: '一个变换里，哪些方向「岿然不动」？', bridge: '主轴、稳定表征', status: 'live' },
      { slug: 'spectral', num: '16', title: '对称矩阵与谱分解', hook: '最「温顺」的一类矩阵长什么样？', bridge: '协方差、Gram 矩阵', status: 'live' },
    ],
  },
  {
    name: '第四部分 · 正交、回归与投影',
    blurb: '旋转、垂直、最小二乘——这里把统计接进几何。',
    chapters: [
      { slug: 'orthogonal-rotation', num: '17', title: '正交矩阵与旋转', hook: '不改变长度的变换长什么样？', bridge: 'RoPE 旋转位置编码', status: 'live' },
      { slug: 'orthogonal-projection', num: '18', title: '正交投影', hook: '把一个向量「拍」到子空间上', bridge: '多头注意力的子空间分工', status: 'live' },
      { slug: 'gram-schmidt', num: '19', title: '格拉姆-施密特与 QR', hook: '怎么造一组互相垂直的基？', bridge: '正交基、数值稳定', status: 'live' },
      { slug: 'regression-projection', num: '20', title: '回归 = 投影', hook: '「拟合一条线」为什么等价于「投影」？', bridge: '最小二乘的几何（统计与几何合流）', status: 'live' },
    ],
  },
  {
    name: '第五部分 · 降维：抓住主要矛盾',
    blurb: 'SVD 一把钥匙，开降维、压缩、LoRA 三把锁。',
    chapters: [
      { slug: 'svd', num: '21', title: 'SVD 奇异值分解', hook: '任意矩阵都能拆成「旋转-拉伸-旋转」？', bridge: '万能分解', status: 'live', core: true },
      { slug: 'low-rank', num: '22', title: '低秩近似', hook: '为什么留几个最大奇异值就够了？', bridge: 'LoRA、压缩', status: 'live', core: true },
      { slug: 'pca', num: '23', title: 'PCA 主成分分析', hook: '高维数据怎么「拍扁」还不丢信息？', bridge: 'embedding 可视化、降维', status: 'live' },
    ],
  },
  {
    name: '第六部分 · 学习：模型怎么变聪明',
    blurb: '梯度、反向传播、正则化——训练的全部机理。',
    chapters: [
      { slug: 'gradient', num: '24', title: '梯度', hook: '往哪个方向走 loss 降得最快？', bridge: '训练信号', status: 'live' },
      { slug: 'chain-rule', num: '25', title: '链式法则与 Jacobian', hook: '误差怎么一层层传回去？', bridge: '反向传播', status: 'live' },
      { slug: 'gradient-descent', num: '26', title: '梯度下降', hook: '怎么一步步逼近答案？', bridge: '训练全过程、Adam', status: 'live' },
      { slug: 'regularization', num: '27', title: '正则化：Ridge 与 Lasso', hook: 'L2 给参数加弹簧、L1 逼出稀疏——为什么有用？', bridge: 'weight decay；Ridge/Lasso = 高斯/拉普拉斯先验下的 MAP', status: 'live' },
      { slug: 'normalization', num: '28', title: '归一化', hook: '为什么每层后面都要「拉平」一下？', bridge: 'LayerNorm / RMSNorm', status: 'live' },
    ],
  },
  {
    name: '第七部分 · 概率视角：模型在「猜下一个词」',
    blurb: '把分数变概率，把训练变成极大似然。',
    chapters: [
      { slug: 'softmax', num: '29', title: 'Softmax 与概率分布', hook: '怎么把一堆分数变成一组「权重」？', bridge: '注意力权重、输出分布', status: 'live', core: true },
      { slug: 'cross-entropy', num: '30', title: '交叉熵与极大似然', hook: 'LLM 的训练目标到底在最小化什么？', bridge: '交叉熵 loss = MLE（统计的主场）', status: 'live' },
    ],
  },
  {
    name: '第八部分 · 合成：亲手拼出注意力',
    blurb: '前面所有零件，在这里组装成 Transformer。',
    chapters: [
      { slug: 'self-attention', num: '31', title: '自注意力', hook: '吓人的注意力公式，到底在算什么？', bridge: '这就是 attention 的全部', status: 'live', core: true },
      { slug: 'multi-head', num: '32', title: '多头注意力', hook: '为什么要「分头」看？', bridge: 'multi-head', status: 'live' },
      { slug: 'rope', num: '33', title: '位置编码与 RoPE', hook: '注意力本身分不清词序，位置信息怎么加进去？', bridge: 'RoPE 旋转位置编码（旋转矩阵的回归）', status: 'live' },
      { slug: 'transformer-block', num: '34', title: '一个 Transformer Block', hook: '残差 + 归一化 + MLP 怎么组装成一层？', bridge: '完整的一层', status: 'live' },
    ],
  },
  {
    name: '第九部分 · 尾声：接到 LLM 工程',
    blurb: '地基打完，落到你的真实目标：训练、微调、推理。',
    chapters: [
      { slug: 'lora-finetuning', num: '35', title: 'LoRA 与高效微调', hook: '为什么改一个低秩的小矩阵就能微调大模型？', bridge: '微调 / PEFT（低秩在这集大成）', status: 'live' },
      { slug: 'sampling-decoding', num: '36', title: '采样与解码', hook: '模型怎么从概率分布里「选词」生成？', bridge: '推理：temperature / top-k / top-p', status: 'live' },
      { slug: 'quantization', num: '37', title: '量化与数值', hook: '为什么权重能从 float 压到 int8/int4 还能用？', bridge: '推理效率、数值精度', status: 'live' },
    ],
  },
  {
    name: '第十部分 · 终点：完整的前向传播',
    blurb: '把前面所有零件串成一次完整的前向传播，然后交棒给正课。',
    chapters: [
      { slug: 'forward-pass', num: '38', title: '跑通一次前向传播', hook: '一句话喂进模型，从 token 到下一个词，到底流过哪些矩阵？', bridge: '完整的 forward pass；接下来去哪', status: 'live', core: true },
    ],
  },
]

export const allChapters: Chapter[] = parts.flatMap((p) => p.chapters)

export const chapterPath = (c: Chapter) => `/ch/${c.slug}`

export const firstLiveChapter = (): Chapter | undefined =>
  allChapters.find((c) => c.status === 'live')

export const findChapter = (slug: string): Chapter | undefined =>
  allChapters.find((c) => c.slug === slug)

/** 取某章的上一/下一章（用于章节底部导航）。 */
export function neighbors(slug: string): { prev?: Chapter; next?: Chapter } {
  const i = allChapters.findIndex((c) => c.slug === slug)
  return {
    prev: i > 0 ? allChapters[i - 1] : undefined,
    next: i >= 0 && i < allChapters.length - 1 ? allChapters[i + 1] : undefined,
  }
}
