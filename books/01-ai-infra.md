# 知乎蓝宝书第一册｜AI infra：模型如何被训练、运行与持续优化

> [!NOTE]
> **这份地图的读法**  
> ○ 到八章逐步递进：先补一级台阶，再依次回答「AI Infra 管什么」「推理怎样被调度」「vLLM 与 SGLang 两个框架差在哪」「训练规模上去以后难在哪」「RL 为什么逼着系统重构」「Agent 又加了什么」「怎么判断优化真的有效」「想深入从哪进」。顺着读一遍约 5–6 小时，能搭起完整骨架；已有基础的可以直接翻每章末尾的「📚 本章出现的知乎内容」。  
> 难度标注：🟢 入门｜🔵 进阶｜🟣 实战（含代码与实验）
> **前置知识**：Python、深度学习基础，大致知道 Transformer 长什么样  
> **系列关系**：这是「蓝宝书」系列的第一册，讲模型能力怎样被高效训练、稳定运行，并在真实反馈中持续更新。第二册《RL 蓝宝书》聚焦其中的 RL 环节，读到第五章可对照；第三册《具身智能蓝宝书》讲机器人侧的系统问题，读到第二章与第六章时可对照。  
> **选稿标准**：均来自知乎站内技术内容，按赞同数与内容质量双重筛选（筛选记录见文末维护说明）  
> **版本**：v3 ｜ **最后更新**：2026-09-03

---

## ○ 先补一级台阶：一条请求背后，系统在做哪些事？

> [!IMPORTANT]
> **本章问题**：向大模型发一条消息，看到的只是不断出现的文字，中间那段时间里发生了什么？

模型不能一次吐出一整段话，因为下一个字取决于前面已经写出的所有字——生成是**自回归**的。这个约束把一次推理分成了两段：

- **Prefill**：把你输入的整段 prompt 一次性读完，算出每个位置的 Key / Value 并缓存下来。这一步的输入是已知的，可以整段并行处理。
- **Decode**：此后每生成一个 token，都要把此前所有位置的信息重新过一遍。输入长度为一，无法并行，只能一轮一轮往下走。

为了避免每轮都重算前文，Prefill 算出的 Key / Value 会被缓存，这就是 **KV Cache**。它省下了重复计算，代价是随上下文长度和并发数线性膨胀，在长上下文场景下常常比模型权重本身更占显存。

后面几章会反复回到这两个概念：推理系统的绝大多数优化，本质上都是在处理「Prefill 和 Decode 是两种截然不同的负载」以及「KV Cache 又大又难管」这两件事。

---

## 一、先画一张地图：AI Infra 究竟在管理什么？

> [!IMPORTANT]
> **本章问题**：面对 Megatron、DeepSpeed、vLLM、SGLang、VeRL、Slime、ROLL、Ray 这一堆框架，从哪个角度看它们？

上一章把一次请求拆成了 Prefill 与 Decode。但当模型大到一张卡放不下、请求多到来不及排队、还要靠强化学习持续更新时，问题就不再是「怎么算得快」，而是「怎么组织」。

理解任何一个 Infra 框架，可以先不急着看它支持多少模型和算法，而是观察它怎样管理**计算、状态、数据**。

### 1.1 计算：模型下一步在哪里执行？

训练框架里的数据并行、张量并行、流水线并行，推理框架里的请求调度、Continuous Batching、P/D 分离，本质上都在重新安排计算——哪张卡执行什么、何时执行、哪些工作可以并行。

### 1.2 状态：参数、梯度和 KV Cache 放在哪里？

训练要保存参数、梯度、优化器状态和中间激活；推理要保存不断增长的 KV Cache；RL 系统还要在训练端与推理端之间同步模型权重。模型越大，状态越不可能放在一张卡上，切分和传输也就越容易成为瓶颈。

先摆出账本作为基准，后面各章再来具体讨论「怎么省」时：

| 状态 | 训练时每参数大致占用（混合精度） | 说明 |
|-|-|-|
| 参数 | 2 字节（BF16 / FP16）+ 4 字节（FP32 主副本） | 混合精度下 FP32 主副本是标配，不是可选项 |
| 梯度 | 2 字节 | 与参数同形状；部分实现另留一份 FP32 buffer |
| 优化器状态 | 8 字节（Adam 的一阶、二阶动量，FP32） | ZeRO 主要省下的就是这部分；换成 SGD+momentum 是 4 字节，8-bit Adam 约 2 字节 |
| 激活 | 随 batch size 与序列长度增长 | 可用重计算换显存 |

四项相加约 16 字节/参数，这是 ZeRO 系列工作里通用的口径。

推理侧的状态则以权重和 KV Cache 为主。KV Cache 的每 token 占用约为 `2 × 层数 × KV 头数 × head_dim × 精度字节数`。这个公式有两个常见的套错方式：主流模型用的是 **GQA**，KV 头数远小于注意力头数（常见 4–8 倍），按注意力头数代进去会高估；**MLA**（DeepSeek 系）则根本不适用——MLA 缓存的是低秩 latent，要按 `层数 × latent 维度 × 精度字节数` 算，每 token 占用比同规模 MHA 小约一个量级。

### 1.3 数据：样本、请求和轨迹怎样进入下一阶段？

训练数据通常成批读取，在线请求动态到达，RL 轨迹则由当前模型与环境现场生成。三种工作负载的到达方式、生命周期和失败语义都不一样，也就不能共用同一套调度逻辑。

| 管理对象 | 训练侧 | 推理侧 | RL 侧 |
|-|-|-|-|
| **计算** | DP / TP / PP / EP 并行 | 请求调度、Continuous Batching、P/D 分离 | Rollout 与训练交替 |
| **状态** | 参数、梯度、优化器状态、激活 | KV Cache | 训练端与推理端的权重同步 |
| **数据** | 成批读取的样本 | 动态到达的请求 | 现场生成的轨迹 |

Infra 框架名字看着多，回答的主要是三个问题：模型太大一张卡放不下怎么训、请求长短不一怎么高效推理、训练-推理-环境怎么凑成一个能持续更新的闭环。后面每一章，也都可以回到这张表来定位。

顺带说一句，广义的 AI Infra 还包括 GPU、NPU、服务器、网络、存储和数据中心。它们是上层系统的基础，但不是这一册的重点——本册只看硬件之上的软件系统，尤其是训练系统、推理系统，以及正在把两者连接起来的 RL 与 Agent Infra。

想先建立全局视野，可以从 [《AI Infra 软核教程（一）：为什么我们需要 AI Infra》](https://zhuanlan.zhihu.com/p/2063938554061967779) — 锦恢 · 🟢 开始。它把计算集群、训练、推理、数据和 Agent Infra 放在同一张图里，读完再看后面各章不容易迷路。如果想先看清训练、推理、RL 为什么会各长成一套系统，[《AI Infra 核心逻辑与大模型行业趋势》](https://zhuanlan.zhihu.com/p/1950625325127014130) — ZOMI酱 · 🟢 从产业视角梳理了各层的分工。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [AI Infra 软核教程（一）：为什么我们需要 AI Infra](https://zhuanlan.zhihu.com/p/2063938554061967779) | 锦恢 | 🟢 | AI Infra 各层的分工与全貌 |
| [AI Infra 核心逻辑与大模型行业趋势](https://zhuanlan.zhihu.com/p/1950625325127014130) | ZOMI酱 | 🟢 | 从产业视角看各环节的位置 |

> [!TIP]
> **自检**：随便挑一个你听过的框架，说说它主要管的是计算、状态，还是数据？它在另外两项上做了妥协吗？

---

## 二、推理系统：怎样让不同请求高效共享模型？

> [!IMPORTANT]
> **本章问题**：有的输入 10 个 token，有的 10 万个；有的只生成一句话，有的 Agent 连跑几十轮——同一张卡怎么服务它们？

上一章把 AI Infra 拆成了计算、状态、数据三件事。这一章则看它们在推理侧的具体形态。

一个请求大致会经历：

> **进入队列 → 调度与组批 → Prefill → 保存 KV Cache → 逐步 Decode → 返回结果**

这条流水线上每一环都在做取舍，下面将逐段拆开。

### 2.1 Prefill 和 Decode ：两种不同的负载

Prefill 一次处理整段输入，计算量大但数据复用率高，通常偏**计算密集**；Decode 每轮只生成少量 token，却要反复读取模型权重和整段 KV Cache，更容易被**显存带宽**卡住。两者的最优 batch size、并行策略和算子实现都不一样。

把它们放在同一组设备上，部署简单，却很难同时匹配两种资源需求——于是有了 **P/D 分离**：Prefill 与 Decode 各自部署，按负载独立扩缩容。

拆开不是免费的：Prefill 产出的 KV Cache 必须传给 Decode 节点。传输、格式转换和排队的开销，可能把分离带来的收益吃掉。所以 P/D 分离真正的工程问题不是「要不要拆」，而是「KV Cache 怎么传」。

想看清传输这一层，[《vLLM PD 分离 KV cache 传递机制详解与演进分析》](https://zhuanlan.zhihu.com/p/1906741007606878764) — kaiyuan · 🔵 是这一节的主入口，把几种传递路径与其代价讲透了。配套推荐读 [《1.5×提升：PD 分离 KV cache 传输的实践经验》](https://zhuanlan.zhihu.com/p/1946608360259577576) — kaiyuan · 🟣，能看到通信、数据转换与计算重叠具体怎么配合。想从零搭一遍，[《nanoPD：一个 LLM P/D 分离推理引擎的实现笔记》](https://zhuanlan.zhihu.com/p/2026307825358382436) — 暮易 · 🟣 跟着读一遍，结构上的细节会记得更牢。

### 2.2 KV Cache 是推理系统的核心状态

2.1 说 P/D 分离的代价在传输，而传输、分配、复用这些动作的对象都是同一个东西——KV Cache。它避免了每轮重算前文，但代价是随上下文和并发不断膨胀。怎么**分配、复用、迁移、压缩**，成了各家框架的主要差异点。

碎片是第一个问题：按最长可能长度预留整块显存，实际利用率往往很低。**PagedAttention**（vLLM 的标志性机制）把 KV Cache 切成固定大小的块按需分配，碎片不再靠预留来避免，代价只是多查一张块表。

复用是第二个问题：多轮对话、共享系统提示词、树搜索这类负载，请求之间往往共享大量前缀。**RadixAttention**（SGLang 的核心机制）用树状结构组织可复用的前缀，让跨请求、跨轮次的状态复用变成一个调度问题，而不只是单个请求内部的显存问题。

前缀缓存不是某一家框架的独有能力：vLLM 原生支持基于 block hash 的 Automatic Prefix Caching，SGLang 用 RadixAttention 实现前缀复用。两者的差异在**复用粒度与淘汰策略**——前者按块哈希（Block Hash）匹配，后者按树结构组织、支持任意位置分叉并按 LRU 淘汰。

围绕 KV Cache 有四个基本动作，各家框架的差异基本都落在这些动作上：

- **分配**：按块分配（PagedAttention）还是按请求预分配，决定了碎片率与能否做抢占；
- **复用**：跨请求、跨轮次共享前缀，命中率直接影响 TTFT 与成本；
- **迁移**：P/D 分离时要在设备间传输，显存放不下时要在 GPU 与 CPU 内存之间换入换出，迁移带宽常常成为隐藏瓶颈；
- **压缩**：对 KV Cache 做低比特量化、改用 MLA 这类低秩表示，或按重要性丢弃部分 token——代价是精度或实现复杂度。

判断一个 KV Cache 方案时，可以顺着这四个问题看它优化了哪一项、又在哪一项上让了步。至于 PagedAttention 与 RadixAttention 背后的两个框架——vLLM 与 SGLang 各自怎么管理状态、消除什么等待，第三章会专门对照。

### 2.3 推理优化不只靠调度

当前推理优化大致可以分成六类：

| 方向 | 代表技术 | 省的是什么 |
|-|-|-|
| 少算 | 稀疏注意力、条件计算 | 不满足条件的计算直接跳过 |
| 少存 | FP8 / FP4 / INT8 / INT4 量化、KV Cache 压缩 | 数值精度换显存与带宽 |
| 少访存 | IO 感知算子（FlashAttention 系）、算子融合 | 减少 HBM 读写，FLOPs 未必下降 |
| 多复用 | Prefix Cache（vLLM 的 block hash、SGLang 的 RadixAttention）、跨请求共享 | 算过一次就不算第二遍 |
| 更好调度 | Continuous Batching、P/D 分离、请求路由、分层 Serving、CUDA Graph | 尽量别让 GPU 闲着；CUDA Graph 省的是 kernel 启动与 CPU 调度开销 |
| 串行转并行 | Speculative Decoding（draft 模型 / MTP / n-gram 起草，再一次并行验证） | 用空闲算力换访存次数与端到端延迟 |

「少算」和「串行转并行」的区别值得展开：投机采样先让小模型起草多个 token、再由大模型一次性验证，它并不会减少总计算量——draft 模型的前向和验证阶段的前向都是额外支出。它减少的是 Decode 的**串行步数**与随之而来的反复访存，本质是用冗余算力换端到端延迟。这个区别决定了它的适用场景：算力有富余、但被访存带宽卡住的 Decode 阶段才划算。

「少访存」这一类也不能并入前两类：FlashAttention 既不减少 KV Cache 大小，FLOPs 也和标准 attention 相近，收益几乎全部来自减少 HBM 读写。

这几类其实互相牵连：量化改变了显存和带宽压力，长上下文改变了 KV Cache 的管理方式，MoE 改变了通信模式，最后都会反馈到调度上。

### 2.4 Agent 正在放大「推理税」

普通对话通常只有少量轮次；Agent 会反复规划、调用工具、读取长上下文并继续生成。同一个任务消耗的 Token、KV Cache 和运行时间可能成倍增长，「推理税」这个说法指的就是这部分额外开销。

因此，推理系统不能只追求平均吞吐，还要看这几个指标：

- **TTFT**（Time To First Token）：从请求进来到吐出第一个 token 的耗时
- **TBT**（Time Between Tokens）：相邻 token 的间隔，决定生成是否流畅
- **P99**：99 分位延迟，反映长尾体验
- **单位 Token 成本**：输入与输出应分开算——Prefill 与 Decode 的成本结构不对称，打包成一个数字无法指导优化

有两个口径上的坑：**TBT 同样需要看 P99**（平均 TBT 会掩盖抖动），以及**长任务是否挤占了短请求**（平均吞吐提升可能以牺牲短请求延迟为代价）。

调度这一层，[《重读 LLM Serving 调度论文》](https://zhuanlan.zhihu.com/p/2022806041373550259) — 雪人 · 🔵 梳理了 Serving 的调度粒度是怎样一步步变细的。想建立整体路线感，[《大模型推理加速技术的学习路线是什么？》](https://www.zhihu.com/question/591646269/answer/1909169222518567197) — 骑虎南下 · 🟢 把推理优化的各个方向和入门次序串了一遍。想直接读代码，[《推理框架极简入门：用 Nano-vLLM 搭建知识体系》](https://zhuanlan.zhihu.com/p/2008285806222132143) — kaiyuan · 🟣 与 [《2025 最快下手 vLLM 的项目——nanovllm 源码解读》](https://zhuanlan.zhihu.com/p/1925484783229698084) — Tiannuo Yang · 🟣 都可以从调度器、模型执行和显存管理三条线切进去。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [vLLM PD 分离 KV cache 传递机制详解与演进分析](https://zhuanlan.zhihu.com/p/1906741007606878764) | kaiyuan | 🔵 | KV Cache 怎么传，各方案代价如何 |
| [1.5×提升：PD 分离 KV cache 传输的实践经验](https://zhuanlan.zhihu.com/p/1946608360259577576) | kaiyuan | 🟣 | 通信、转换与计算重叠怎么配合 |
| [nanoPD：一个 LLM P/D 分离推理引擎的实现笔记](https://zhuanlan.zhihu.com/p/2026307825358382436) | 暮易 | 🟣 | 从零搭一个最小 P/D 分离引擎 |
| [重读 LLM Serving 调度论文](https://zhuanlan.zhihu.com/p/2022806041373550259) | 雪人 | 🔵 | 调度粒度是怎么一步步变细的 |
| [大模型推理加速技术的学习路线是什么？](https://www.zhihu.com/question/591646269/answer/1909169222518567197) | 骑虎南下 | 🟢 | 推理优化有哪些方向、按什么次序学 |
| [推理框架极简入门：用 Nano-vLLM 搭建知识体系](https://zhuanlan.zhihu.com/p/2008285806222132143) | kaiyuan | 🟣 | 调度器 / 执行器 / 显存管理怎么读代码 |
| [2025 最快下手 vLLM 的项目——nanovllm 源码解读](https://zhuanlan.zhihu.com/p/1925484783229698084) | Tiannuo Yang | 🟣 | 最小可跑的 vLLM 代码路径 |

> [!TIP]
> **自检**：某篇优化文章的方案落地后，它省的是算力、显存、带宽还是串行步数？被省下的那部分，代价挪到哪儿去了？

---

## 三、服务框架侧写：vLLM 与 SGLang

> [!IMPORTANT]
> **本章问题**：这两个框架到底在管什么状态、消除什么等待？

上一章讲的是推理系统要解决的问题，这一章看两个具体系统各自怎么解。如果只把它们写成「两个高性能推理框架」，很难理解它们为什么重要；更准确的观察方式，是回到第一章那三个问题：管什么状态、消除什么等待。

**vLLM：从显存管理出发，把模型变成高吞吐服务**

标志性机制是 **PagedAttention**：把连续增长的 KV Cache 切成可分页管理的块，减少预留和碎片；再配合 Continuous Batching，让调度器不停地把新请求塞进正在运行的批次里。它的定位偏向通用推理引擎，重点在模型兼容、分布式执行、吞吐和服务化落地。

想摸透它，可以顺着四条线走：KV Cache 怎么分配、回收、换出 → Scheduler 怎么在吞吐、TTFT、TBT 之间取舍 → Chunked Prefill、Prefix Caching、Speculative Decoding 各自消除哪一类等待 → TP / PP 和多机部署怎么改变通信成本。

**SGLang：把前缀复用和复杂生成程序放进 Runtime**

它同样有很强的模型服务能力，但更强调「生成程序」的执行：多轮对话、分支生成、结构化输出、工具调用，这些场景往往共享大量前缀。**RadixAttention** 用树状结构管理可复用的前缀 KV Cache，让跨请求、跨轮次的状态复用变成一个调度问题，而不只是单个请求内部的显存问题。

观察它可以盯这几处：Prefix Cache 命中率怎么影响 TTFT 和成本 → RadixAttention 怎么做前缀的插入、匹配、淘汰 → 结构化生成和约束解码是怎么进运行时的 → 多轮 Agent、共享系统提示词、树搜索这类负载为什么更能发挥它的优势。

**选型时值得问的几件事（而不是「谁更快」）**

- 工作负载里有没有大量**共享前缀**？
- 是单轮短请求，还是长上下文加多轮 Agent？
- 模型和硬件的支持成不成熟？
- 团队更看重生态和部署稳定性，还是缓存复用与复杂程序执行？
- 压测时**至少同时记下** TTFT、TBT、吞吐、P99、显存占用和失败率，并固定住模型、并发、输入 / 输出长度和缓存命中条件。

两者的边界正在收敛，不宜用「谁绝对更快」下结论。想看具体差异，[《大模型推理框架，SGLang 和 vLLM 有哪些区别？》](https://www.zhihu.com/question/666943660/answer/1937585837995975343) — 杞鋂 · 🔵 与 [《vllm 和 sglang 的真正区别是什么？》](https://www.zhihu.com/question/2045055313053843631/answer/2051092309400409916) — WingEdge777 · 🔵 是同主题下两份角度不同的回答，对照着读能看出分歧在哪一层。想动手，[《LLM 推理框架（vLLM/SGLang）入门 Notebook 练习（2026 年第 1 期）》](https://zhuanlan.zhihu.com/p/1999518738303693534) — kaiyuan · 🟣 与 [《浅尝 mini sglang，回顾 nano vllm》](https://zhuanlan.zhihu.com/p/1984934838818604631) — Shengguang · 🟣 都可直接跟着跑。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [大模型推理框架，SGLang 和 vLLM 有哪些区别？](https://www.zhihu.com/question/666943660/answer/1937585837995975343) | 杞鋂 | 🔵 | 两者在设计取向上的具体差异 |
| [vllm 和 sglang 的真正区别是什么？](https://www.zhihu.com/question/2045055313053843631/answer/2051092309400409916) | WingEdge777 | 🔵 | 另一角度的对照 |
| [LLM 推理框架（vLLM/SGLang）入门 Notebook 练习](https://zhuanlan.zhihu.com/p/1999518738303693534) | kaiyuan | 🟣 | 边跑边对照两个框架 |
| [浅尝 mini sglang，回顾 nano vllm](https://zhuanlan.zhihu.com/p/1984934838818604631) | Shengguang | 🟣 | 从精简实现反推设计意图 |

### 🔧 更进一步：动手教程

来自 Datawhale 社区的 [从零手搓 SGLang 的 AI Infra 教程](https://github.com/datawhalechina/zero-to-sglang)，带你把 AI Infra 的推理原理真正落到代码时间里。

> [!TIP]
> **自检**：你的工作负载里，共享前缀占比大概多少？按这个数字判断，RadixAttention 的收益会落在 TTFT 还是吞吐上？

---

## 四、训练系统：规模扩大以后，为什么难点转向通信与可靠性？

> [!IMPORTANT]
> **本章问题**：把模型拆到几百张卡上之后，真正的对手是谁？

前两章看的是推理——输入已知、请求动态、状态以 KV Cache 为主。训练是完全不同的负载：一次迭代要完成读数据、前向、反向和参数更新，四步之间环环相扣，且要连续跑上数周。

模型一大，参数、梯度、优化器状态和中间激活塞不进一张卡，就得拆。DP、TP、PP、CP、EP、ZeRO 解决的是「怎么拆」；拆完之后，问题变成另一句：

**拆开以后，许多设备怎样高效、稳定地协同数周？**

### 4.1 通信决定扩展能否继续

不同并行方式引入的通信模式完全不同：

- **DP**（数据并行）每张卡持有完整模型副本，反向后需要同步梯度，通信量正比于参数量；
- **TP**（张量并行）把单个算子切开，前向和反向都要频繁交换中间结果；
- **EP**（专家并行）依赖动态的 All-to-All，token 路由到哪个专家事先未知。

规模越大，AllReduce、AllGather、ReduceScatter 这些集合通信越容易挤进关键路径。

所以系统一方面要减少通信量，另一方面要让通信与计算重叠。但重叠有适用范围：可重叠的主要是反向阶段的梯度 AllReduce / ReduceScatter，以及 EP 的 All-to-All；TP 的激活通信位于关键路径、存在数据依赖，难以完全隐藏——常见做法是先引入 **Sequence Parallel**，把 TP 的 AllReduce 拆成 ReduceScatter + AllGather 并与 LayerNorm / Dropout 的计算重叠，再靠切分粒度与算子融合继续压缩。

这也解释了为什么「多一倍设备不等于快一倍」——新增的算力，可能被通信、流水线空泡和负载不均吃掉。

并行策略的体系化梳理，可以从 [《大模型分布式训练并行技术（一）- 概述》](https://zhuanlan.zhihu.com/p/598714869) — 吃果冻不吐果冻皮 · 🟢 入门，再用 [《一文捋顺千亿模型训练技术：流水线并行、张量并行和 3D 并行》](https://zhuanlan.zhihu.com/p/617087561) — 白强伟 · 🔵 把 PP / TP / 3D 并行的来龙去脉捋顺。

### 4.2 MoE 和长上下文正在重新定义训练 Infra

MoE 不只是多了一种并行方式。Token 路由是否均衡、专家会不会过载、跨节点 All-to-All 怎么走，都会改变系统设计；具体到算子层，dense 模型上 prefill 与 decode 本来就走不同的 kernel，MoE 则进一步要求 grouped GEMM 在大 M（prefill）与小 M（decode）两种形状下各有实现——这是它特有的工程负担。

长上下文给训练侧带来的是激活与注意力计算的膨胀（注意力的中间结果随序列长度呈平方增长），KV Cache 的压力则主要落在推理侧。这推动了 **Context Parallel**（Ring Attention、DeepSpeed-Ulysses 等都是它的具体实现）与稀疏注意力进入训练与推理系统。

Infra 不是在一个固定的模型上做事后优化，**模型结构和系统在互相塑造**：MoE 改变通信，长上下文改变状态管理，Agent 改变任务生命周期。

这一节可以从 [《MOE 大模型架构与机制详解——以 DeepSeek-v3 为例》](https://zhuanlan.zhihu.com/p/22570639120) — 北方的郎 · 🟢 建立结构认知，再看它对系统提出了什么要求。

### 4.3 跑得快之前，先要跑得完

大规模训练面对的故障可以按因果分成两层：

**底层是硬件与通信异常**——静默数据损坏（SDC）、掉卡、Straggler（个别节点明显变慢）。

**上层是它们表现出来的现象**——NCCL Timeout、训练 Hang、Loss Spike。三者需要的是三种不同手段：检测、定位、恢复。Straggler 常常就是 NCCL Timeout 和 Hang 的成因，而 Loss Spike 未必是数值问题，也可能由 SDC 直接诱发。

因此，训练 Infra 还必须处理：Checkpoint 的保存与恢复、故障检测和自动拉起、通信 Hang 的定位、训练确定性与问题复现、慢节点和异常节点的识别、任务级监控与容量规划。

一句话概括：**并行策略决定模型怎样拆开，通信系统决定拆开后能否高效协作，可靠性系统决定这场协作能否持续完成。**

把通信优化、Hang 诊断和稳定性放回真实的大规模训练场景，[《大模型基建这三年：AI Infra 通信演进之路》](https://zhuanlan.zhihu.com/p/1989653116635870625) — 老七 · 🔵 适合作为并行策略之后的进阶阅读。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [大模型分布式训练并行技术（一）- 概述](https://zhuanlan.zhihu.com/p/598714869) | 吃果冻不吐果冻皮 | 🟢 | 各种并行方式的动机与区别 |
| [一文捋顺千亿模型训练技术：流水线并行、张量并行和 3D 并行](https://zhuanlan.zhihu.com/p/617087561) | 白强伟 | 🔵 | PP / TP / 3D 并行的来龙去脉 |
| [MOE 大模型架构与机制详解——以 DeepSeek-v3 为例](https://zhuanlan.zhihu.com/p/22570639120) | 北方的郎 | 🟢 | MoE 的结构与它对系统的要求 |
| [大模型基建这三年：AI Infra 通信演进之路](https://zhuanlan.zhihu.com/p/1989653116635870625) | 老七 | 🔵 | 通信优化、Hang 诊断与稳定性实践 |

> [!TIP]
> **自检**：给你 64 张卡训一个 70B 模型，你会怎么拆？通信量最大的那一段在哪，它能被计算重叠掉吗？训练中途挂了，怎么接着跑？

---

## 五、RL Infra：为什么行业开始重构训练流水线？

> [!IMPORTANT]
> **本章问题**：SFT 的数据是提前存好的，RL 的数据要现场生成——这条流水线为什么没法直接拿训练框架凑合？

前两章看的是「数据已就绪」的负载：推理的输入是请求，训练的输入是语料。RL 不一样，**训练数据本身就是模型跑出来的**——这一条改变了整个系统的形态。

RL 后训练的基本链路是这样：

**Rollout（循环调用 Environment）→ Reward / Verifier → Trainer → Weight Transfer → 新一轮 Rollout**

链路里最容易画错的是 Environment 的位置。它不是 Rollout 之后的一个独立阶段，而是**被 Rollout 内部反复调用**的：生成 action → 环境执行 → 观测回填 → 继续生成，一条轨迹内的交互次数可达上千次。把它画成串行阶段，会让人以为 rollout 是纯生成、环境在之后统一执行，从而低估环境延迟的影响——而环境调用次数乘以单次延迟，恰恰是这一阶段最主要的耗时来源。

### 5.1 每个阶段面对的瓶颈不同

| 阶段 | 主要瓶颈 |
|-|-|
| Rollout（含 Environment） | 推理吞吐、KV Cache、长轨迹、环境启动延迟、工具调用与失败恢复 |
| Reward / Verifier | 验证成本、批处理、反馈延迟 |
| Trainer | 显存、训练吞吐、集合通信 |
| Weight Transfer | 带宽、同步频率、策略新鲜度 |
| Buffer | 数据积压、背压、样本时效 |

全同步的话，快的环节要等慢的环节；全异步的话，训练用的轨迹可能来自好几代以前的策略。这个取舍差不多是 RL Infra 的核心。

策略滞后的代价在梯度层面：**采样分布偏离当前策略后，重要性采样比值失配会放大梯度方差，极端情况下直接发散**。工程上通常靠 staleness 上限、截断重要性采样和丢弃过旧轨迹来控制。

### 5.2 当前主趋势是角色解耦与独立扩缩容

RL Infra 正从「一套程序跑完整条流水线」，走向 Rollout、Environment、Reward 和 Trainer 分别运行，再通过明确的接口交换轨迹、奖励和权重。

好处是每个阶段能按自身负载独立扩容，环境和 Agent 框架也不必嵌进 Trainer。代价是四件事被重新摆上台面：数据版本怎么管理、策略滞后怎么控制、权重何时同步、系统怎么处理背压。

VeRL、Slime、ROLL 等框架，正是在这些取舍上形成了不同设计。

想从整体上认识 RL Infra，[《【AI Infra】VeRL 框架入门 & 代码带读》](https://zhuanlan.zhihu.com/p/27676081245) — 不关岳岳的事 · 🟢 是这份清单里赞同最高的一篇，Ray Actor、资源池、角色分工都讲到了，从它开始比较省时间。想理解为什么需要这样组织，[《RL Scaling 时代，我们需要什么样的 RL 框架呢？》](https://zhuanlan.zhihu.com/p/1919107858110316886) — zhuzilin · 🔵 从 Slime 出发讨论训练后端、推理后端和数据生成的组合方式。

架构层面，[《基于 Ray 的分离式架构：veRL、OpenRLHF 工程设计》](https://zhuanlan.zhihu.com/p/26833089345) — 杨远航 · 🔵 对比了两种分离式实现的取舍；[《Agentic RL 时代的 Infra 重构》](https://zhuanlan.zhihu.com/p/2022786148087464077) — 低级炼丹师 · 🔵 横向对比 Forge、ROLL、Seer、Slime 四个系统；[《从各家技术文章看 26 年 Agentic RL Infra 优化方向》](https://zhuanlan.zhihu.com/p/2007250216227729670) — attack204 · 🔵 看行业在往哪走；这类年度盘点过一阵子再读会是另一番景象。

想读源码，[《深入浅出理解 verl 源码（Part 1）》](https://zhuanlan.zhihu.com/p/1920751852749849692) — Chayenne Zhao · 🟣 与 [《Slime 框架深度解析：面向大规模 RL 的训推一体化实践》](https://zhuanlan.zhihu.com/p/1921606246454239436) — 曹宇 · 🔵 分别对应两条不同的设计路线。

> [!NOTE]
> 关于 RL 的算法侧（奖励从哪来、PPO / GRPO 的分野、能力是否真的提升），本系列第二册《RL 蓝宝书》第二章与第三章有完整展开，本册只讲它给系统带来的负担。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [【AI Infra】VeRL 框架入门 & 代码带读](https://zhuanlan.zhihu.com/p/27676081245) | 不关岳岳的事 | 🟢 | Ray Actor、资源池与角色分工 |
| [RL Scaling 时代，我们需要什么样的 RL 框架呢？](https://zhuanlan.zhihu.com/p/1919107858110316886) | zhuzilin | 🔵 | 训练 / 推理后端与数据生成怎么组合 |
| [基于 Ray 的分离式架构：veRL、OpenRLHF 工程设计](https://zhuanlan.zhihu.com/p/26833089345) | 杨远航 | 🔵 | 分离式架构的两种实现取舍 |
| [Agentic RL 时代的 Infra 重构](https://zhuanlan.zhihu.com/p/2022786148087464077) | 低级炼丹师 | 🔵 | 四个系统的横向对比 |
| [从各家技术文章看 26 年 Agentic RL Infra 优化方向](https://zhuanlan.zhihu.com/p/2007250216227729670) | attack204 | 🔵 | 行业优化方向（⚠️ 时效性强） |
| [深入浅出理解 verl 源码（Part 1）](https://zhuanlan.zhihu.com/p/1920751852749849692) | Chayenne Zhao | 🟣 | verl 的代码结构 |
| [Slime 框架深度解析：面向大规模 RL 的训推一体化实践](https://zhuanlan.zhihu.com/p/1921606246454239436) | 曹宇 | 🔵 | 训推一体化这条路线怎么走 |

> [!TIP]
> **自检**：如果 Environment 突然慢了 3 倍，系统哪个环节先出问题？改成异步后，你会用哪几个手段控制策略滞后？

---

## 六、Agent Infra：模型服务正在变成执行系统

> [!IMPORTANT]
> **本章问题**：当「请求」变成一个可能跑几小时的 Agent Program，系统要多管些什么？

前面几章里，系统的管理对象始终是「请求」：进来、算完、返回。Agent 改变了这个前提。

传统模型服务接收一段输入，返回一段输出；Agent 会持续规划、调用工具、等待环境、修改上下文，再继续运行。于是管理对象从「请求」变成了**一个长期运行的程序**，系统要多管的东西可以按层次排开：

| **层次** | 要管什么 |
|-|-|
| **编排层** | Agent Loop 与 Workflow、多 Agent 并发与协作 |
| **执行层** | 工具调用与外部事件、状态保存 / 暂停 / 恢复、执行回放与可观测性 |
| **引擎层** | 上下文与 KV Cache 的生命周期 |
| **隔离层** | 不可信代码的隔离与安全（Sandbox） |

**Sandbox 只是最底下的一层**，解决的是「不可信代码不要烧到整栋楼」。更值得关注的变化在上面两层：Agent Runtime、Workflow IR，以及运行时按当前状态动态组装上下文与执行计划的做法，正在长成新的系统抽象——往后的推理引擎可能不再只是一个 Token Server，而会成为有状态 Agent 执行系统的一部分。

这一层目前站内系统性的中文内容不多，[《Agent Infra：Sandbox 技术和选型》](https://zhuanlan.zhihu.com/p/1999938129465979624) — ZY lian · 🔵 讨论了容器、轻量虚拟机、自托管和云服务之间的取舍；[《Agent sandbox 可能的选型以及 unikernel 的机会》](https://zhuanlan.zhihu.com/p/2007212078172234607) — gaocegege · 🔵 提供了另一个视角。

> [!NOTE]
> 机器人侧的对应问题（数据、部署、延迟与降级），本系列第三册《具身智能蓝宝书》第三章有更具体的展开。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [Agent Infra：Sandbox 技术和选型](https://zhuanlan.zhihu.com/p/1999938129465979624) | ZY lian | 🔵 | 各类 Sandbox 方案的取舍 |
| [Agent sandbox 可能的选型以及 unikernel 的机会](https://zhuanlan.zhihu.com/p/2007212078172234607) | gaocegege | 🔵 | 隔离方案的另一条技术路线 |

> [!TIP]
> **自检**：一个跑了两小时的 Agent 任务中断了，恢复的时候你要还原哪些东西？只存对话历史够不够？

---

## 七、怎样判断一项 Infra 优化是否真的有效？

> [!IMPORTANT]
> **本章问题**：Kernel 快了 30%，为什么用户那边没什么感觉？

前六章讲了各类系统的机制。这一章给一套通用的检查方法——它既用来评价别人的优化工作，也用来组织自己的实验报告与面试回答（第八章会直接复用它）。

AI Infra 很容易陷入局部指标竞争：一个 Kernel 快了 30%，不代表完整模型快 30%；GPU 利用率提高，不一定意味着用户等待时间下降；RL 系统产生更多轨迹，也不一定意味着有效训练数据增加。碰到一份优化方案，可以按这五件事过一遍。

### 7.1 工作负载是什么？

模型多大？输入输出多长？请求怎样到达？是否包含 MoE、长上下文、工具调用或多轮 Agent？脱离工作负载谈性能，结论很难复用。

### 7.2 原来的瓶颈在哪里？

系统受限于计算、显存容量、显存带宽、网络通信、CPU 调度，还是外部环境？没打中主要瓶颈，局部加速换不来端到端收益。

### 7.3 成本被转移到了哪里？

量化降低显存压力，却引入精度与转换成本；P/D 分离提高资源匹配度，却增加 KV Cache 传输；异步 RL 提高吞吐，却带来策略滞后。优化很少是白拿的，它通常只是把成本挪了个地方。

### 7.4 端到端指标是否改善？

训练看吞吐、扩展效率、任务稳定性和恢复成本；推理看 TTFT、TBT（两者都还要看各自的 P99）、吞吐、显存占用与失败率，单位 Token 成本要**区分输入与输出**；RL 还要看轨迹吞吐、策略新鲜度和有效样本比例（指通过 verifier、未被 clip 掉的样本占比）。

### 7.5 系统是否可运维？

生产环境还要面对设备故障、请求突发、模型切换、环境污染和性能退化。能够稳定运行、发现问题并恢复，往往比一次 Benchmark 的峰值更重要。

一套好的 Infra 工作，不只是让某个组件更快，而是讲明白：系统原来为什么慢、瓶颈怎样被消除、成本被转移到哪里，以及最终谁真正获得了收益。

想把这些落到工具和方法上，[《如何系统地分析和定位大模型推理框架（如 SGLang, vLLM）的性能瓶颈？》](https://www.zhihu.com/question/1993781500349539243/answer/2030665166598174331) — chouheiwa · 🔵 回答的正是「瓶颈到底怎么测出来」。另可备一份索引型资料 [《LLM Infra 学习资料整理——推理（持续更新）》](https://zhuanlan.zhihu.com/p/2036442749319296948) — TensorDance · 🟢，按需查缺。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [如何系统地分析和定位大模型推理框架的性能瓶颈？](https://www.zhihu.com/question/1993781500349539243/answer/2030665166598174331) | chouheiwa | 🔵 | 瓶颈怎么测、用什么工具 |
| [LLM Infra 学习资料整理——推理（持续更新）](https://zhuanlan.zhihu.com/p/2036442749319296948) | TensorDance | 🟢 | 推理方向的索引型资料库 |

> [!TIP]
> **自检**：找一篇你最近看过的优化文章，用这五问逐条过一遍，能答上几条？

---

## 八、从哪条路走下去？

> [!IMPORTANT]
> **本章问题**：读完之后，下一步具体做什么？

前面七章建立了判断力，这一章把它变成行动。三个小节对应三种诉求，可以按需取一段。

### 8.1 按技术方向选

| **方向** | **建议路径** | **一点提醒** |
|-|-|-|
| **推理系统** | 请求调度 → Prefill → KV Cache → Decode → P/D 分离 → Profiling | 先把 vLLM、SGLang、Nano-vLLM 摸一遍，CUDA Kernel 可以晚点再钻 |
| **分布式训练** | 训练显存 → DP/TP/PP/CP/EP → 集合通信 → 计算—通信重叠 → MoE → 可靠性 | 停在并行策略的定义上有点可惜，最好能算得出通信量 |
| **RL Infra** | VeRL 的角色与资源编排 → Slime / ROLL 怎么组织各阶段与权重同步 | 重点是同步与异步之间怎么取舍 |
| **Agent Infra** | Runtime、Sandbox、Workflow、状态恢复、工具编排、可观测性 | 核心问题是：怎么跑一个长期、有状态、随时可能失败的程序 |
| **算子与编译器** | GPU 内存层级 → 并行计算 → Profiling → CUDA / Triton → 算子融合 → 编译优化 | 边写边 profile，比光看资料快 |

先选一个方向，不必同时铺开。全局路线可以看 [《AI Infra 学习路线》](https://zhuanlan.zhihu.com/p/2021970155182326008) — 草帽路飞 · 🟢；算子与编译器方向，[《AI 编译器的概览、挑战和实践》](https://zhuanlan.zhihu.com/p/508345356) — 金雪锋 · 🔵 讲体系，[《浅谈 AI 编译器趋势：从更快的 kernel 到重新定义执行边界》](https://zhuanlan.zhihu.com/p/2040199323170951424) — 画饼充饥 · 🔵 讲趋势，[《Triton 算子开发及编译器资源整理》](https://zhuanlan.zhihu.com/p/2018815133590271874) — BobHuang · 🔵 是资源合集。

### 8.2 把阅读变成能力：一条可执行的四周路线

AI Infra 的入门难点不是资料少，而是很容易把论文、框架与 CUDA 学成互不相连的知识点。更有效的方法是先选定一类工作负载，再用一个可测量的小项目贯穿学习。

- **第 1 周｜补齐账本**：能手算参数、激活与 KV Cache 的大致占用，解释 Prefill / Decode 的计算特征，说清 TTFT、TBT、吞吐和 P99 分别代表什么。口径照 1.2 的账本表——GQA 要用 KV 头数，MLA 要换 latent 维度的算法。
- **第 2 周｜跑对照实验**：用同一模型分别跑 vLLM 与 SGLang。固定硬件和模型，改变并发、输入 / 输出长度、共享前缀比例，记录吞吐、TTFT、TBT 与显存占用。**前缀缓存的开关与淘汰策略也要固定**：两个框架的默认配置不同，不控制这个变量，曲线就没有可比性。重点不是「跑出最快数字」，而是解释曲线为什么这样变。
- **第 3 周｜读一条源码链路**：从 Nano-vLLM 或框架的 scheduler、block manager、model runner 入手，把「请求进入 → 分配 KV 块 → 组批 → 执行 → 释放状态」画成一张图，再用 profiler 找到一次真实的等待。
- **第 4 周｜写一份可展示的实验报告**：至少包含环境、工作负载、基线、变量控制、结果、瓶颈判断与失败案例。它比复述十篇论文更接近真实的 Infra 工作，也能直接转化为简历项目。

分布式训练与 RL Infra 可以照同样的节奏走：先在单机小模型上做显存估算，再跑 DDP / FSDP 或 ZeRO 对照实验；随后把 Megatron 里的 DP、TP、PP、CP、EP 对应到具体通信操作。进入 RL Infra 后，先跑 VeRL 的单机示例，理解 Ray actor、resource pool、rollout 与 trainer 的角色；再比较 Slime 等框架的 co-located 与 disaggregated 设计，重点观察权重同步、资源空转、推理训练切换和 policy staleness。

**如果只做一个作品**，建议做「vLLM 与 SGLang 在三类工作负载下的对照实验」：普通聊天、共享长前缀、多轮 Agent。它能同时展示模型理解、框架实践、实验设计、Profiling 和技术表达，比泛泛写「熟悉大模型推理框架」有说服力得多。

### 8.3 准备科研实习或求职

面试真正高频的不是框架 API，而是五类问题：

- **基础账本**：Attention 的计算量与显存占用，训练和推理的状态分别在哪里
- **推理机制**：Prefill / Decode、PagedAttention、Continuous Batching、Prefix Cache、P/D 分离
- **并行通信**：DP、TP、PP、ZeRO / FSDP 的拆分方式、通信量与适用边界
- **性能分析**：算力、显存容量、显存带宽、网络与 CPU 调度中，怎样定位真正的瓶颈
- **系统设计**：给定模型、SLA、流量和硬件，如何选框架、定指标、做容量规划，并说明优化把成本转移到了哪里

回答时可以直接套用第七章那五问：**先说明工作负载，再指出瓶颈，解释方案消除了哪种等待，同时承认它新增的通信、内存或复杂度，最后用端到端指标而不是单个 kernel 峰值证明收益。**

准备材料方面，[《AI infra 面试经验贴》](https://zhuanlan.zhihu.com/p/1970722821522061231) — 抠抠歪 · 🟢 覆盖较全；[《C++/CUDA/AI-infra 面试经验总结》](https://zhuanlan.zhihu.com/p/2005325241803621742) — jinboom · 🔵 偏系统与 C++；[《AI Infra 实习准备记录 01：从三个项目开始》](https://zhuanlan.zhihu.com/p/2041895503454015779) — 哼嗯恒 · 🔵 讲的是怎么用项目补齐经历。

> [!WARNING]
> 面经与实习类内容时效性较强，阅读时注意发布时间；原理部分不会过期，具体题目每年都在换。

### 📚 本章出现的知乎内容

| 篇目 | 作者 | 难度 | 读它能回答什么 |
|-|-|-|-|
| [AI Infra 学习路线](https://zhuanlan.zhihu.com/p/2021970155182326008) | 草帽路飞 | 🟢 | 全局学习路径 |
| [AI 编译器的概览、挑战和实践](https://zhuanlan.zhihu.com/p/508345356) | 金雪锋 | 🔵 | 编译器方向的体系认知 |
| [浅谈 AI 编译器趋势](https://zhuanlan.zhihu.com/p/2040199323170951424) | 画饼充饥 | 🔵 | 编译器在往哪个方向走 |
| [Triton 算子开发及编译器资源整理](https://zhuanlan.zhihu.com/p/2018815133590271874) | BobHuang | 🔵 | 算子方向的资源入口 |
| [AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231) | 抠抠歪 | 🟢 | 面试覆盖哪些议题（⚠️ 时效型） |
| [C++/CUDA/AI-infra 面试经验总结](https://zhuanlan.zhihu.com/p/2005325241803621742) | jinboom | 🔵 | 系统与 C++ 侧的准备（⚠️ 时效型） |
| [AI Infra 实习准备记录 01：从三个项目开始](https://zhuanlan.zhihu.com/p/2041895503454015779) | 哼嗯恒 | 🔵 | 怎么用项目补齐经历（⚠️ 时效型） |

> [!TIP]
> **自检**：对照 8.1 的五个方向，你现在最接近哪一个？列出还差的一两样，以及补齐它的最短路径。

---

## 📖 术语速查

| 术语 | 含义 |
|-|-|
| Prefill | 一次性处理整段输入、生成首个 token 并缓存 K/V 的阶段，偏计算密集 |
| Decode | 逐 token 生成的阶段，反复读取权重与 KV Cache，易受显存带宽限制 |
| KV（Key-Value） Cache | 缓存历史 token 的 Key / Value，避免每轮重算前文 |
| TTFT（Time To First Token） | 首个 token 返回耗时 |
| TBT（Time Between Tokens） | 相邻 token 的间隔；长尾体验要看它的 P99 |
| P99（99th Percentile） | 99 分位延迟，衡量长尾 |
| P/D（Prefill-Decode Disaggregation）分离 | Prefill 与 Decode 部署在不同设备组，各自按负载扩缩容 |
| Continuous Batching | 每个 decode step 重新组批：完成的请求立刻释放槽位，新请求随时插入 |
| PagedAttention | vLLM 的分页式 KV Cache 管理，缓解碎片 |
| RadixAttention | SGLang 的树状前缀 KV Cache 复用机制 |
| Prefix Cache | 跨请求复用共享前缀的 KV Cache；vLLM 用 block hash，SGLang 用 RadixAttention |
| Speculative Decoding / MTP（Multi-Token Prediction） | 先起草多 token 再并行验证，用冗余算力换串行步数与访存 |
| DP / TP / PP（Data / Tensor / Pipeline Parallelism） | 数据 / 张量 / 流水线并行 |
| CP / EP（Context / Expert Parallelism） | 上下文并行 / 专家并行 |
| ZeRO（Zero Redundancy Optimizer） / FSDP（Fully Sharded Data Parallel） | 分片训练状态以省显存，按等级递进：ZeRO-1 分片优化器状态、ZeRO-2 再加梯度、ZeRO-3 再加参数；FSDP 是 PyTorch 侧的实现，默认 FULL_SHARD 对应 ZeRO-3 |
| Context Parallel | 沿序列维度切分长上下文，Ring Attention、DeepSpeed-Ulysses 是它的具体实现 |
| MoE（Mixture of Experts） | 混合专家结构，靠路由器只激活部分专家 |
| AllReduce / AllGather / ReduceScatter / All-to-All | 集合通信原语：AllReduce 求和后广播给所有卡，AllGather 把各卡分片汇聚成完整张量，ReduceScatter 先求和再切分下发，All-to-All 两两交换（EP 的 token 路由靠它） |
| Straggler | 明显慢于同批的其他节点，常是 Timeout 与 Hang 的成因 |
| SDC（Silent Data Corruption） | Silent Data Corruption，静默数据损坏，可能表现为 Loss Spike |
| NCCL（NVIDIA Collective Communications Library） Timeout / 训练 Hang | 通信超时的报错形式 / 训练停滞的现象；两者常由底层异常（Straggler、SDC）引发 |
| Checkpoint | 训练状态的保存与恢复 |
| Rollout | 用当前策略生成轨迹的过程，其间循环调用 Environment |
| Environment | 执行 action 并回传观测的外部系统，被 Rollout 反复调用 |
| Verifier | 判定输出是否符合客观标准的程序 |
| Weight Transfer | 训练端向推理端同步新权重 |
| Policy Staleness | 采样所用策略落后于当前策略的程度。不校正则引入偏差，用重要性采样校正后，比值的方差又被放大 |
| Backpressure | 下游处理不过来时向上游的反压 |
| Agent Runtime / Workflow IR（Intermediate Representation） | Agent Runtime 是长期运行的 Agent 程序的执行环境；Workflow IR（中间表示）是把流程写成可执行描述的一层 |
| Sandbox | 隔离不可信代码执行的机制 |

---

## 📚 全部知乎内容总清单（35 篇）

### 🟢 入门（8 篇）

1. [AI Infra 软核教程（一）：为什么我们需要 AI Infra](https://zhuanlan.zhihu.com/p/2063938554061967779) — 锦恢
2. [AI Infra 核心逻辑与大模型行业趋势](https://zhuanlan.zhihu.com/p/1950625325127014130) — ZOMI酱
3. [大模型推理加速技术的学习路线是什么？](https://www.zhihu.com/question/591646269/answer/1909169222518567197) — 骑虎南下
4. [大模型分布式训练并行技术（一）- 概述](https://zhuanlan.zhihu.com/p/598714869) — 吃果冻不吐果冻皮
5. [MOE 大模型架构与机制详解——以 DeepSeek-v3 为例](https://zhuanlan.zhihu.com/p/22570639120) — 北方的郎
6. [【AI Infra】VeRL 框架入门 & 代码带读](https://zhuanlan.zhihu.com/p/27676081245) — 不关岳岳的事
7. [AI Infra 学习路线](https://zhuanlan.zhihu.com/p/2021970155182326008) — 草帽路飞
8. [AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231) — 抠抠歪 · ⏳ 时效型

### 🔵 进阶（19 篇）

1. [vLLM PD 分离 KV cache 传递机制详解与演进分析](https://zhuanlan.zhihu.com/p/1906741007606878764) — kaiyuan
2. [重读 LLM Serving 调度论文](https://zhuanlan.zhihu.com/p/2022806041373550259) — 雪人
3. [大模型推理框架，SGLang 和 vLLM 有哪些区别？](https://www.zhihu.com/question/666943660/answer/1937585837995975343) — 杞鋂
4. [vllm 和 sglang 的真正区别是什么？](https://www.zhihu.com/question/2045055313053843631/answer/2051092309400409916) — WingEdge777
5. [一文捋顺千亿模型训练技术：流水线并行、张量并行和 3D 并行](https://zhuanlan.zhihu.com/p/617087561) — 白强伟
6. [大模型基建这三年：AI Infra 通信演进之路](https://zhuanlan.zhihu.com/p/1989653116635870625) — 老七
7. [RL Scaling 时代，我们需要什么样的 RL 框架呢？](https://zhuanlan.zhihu.com/p/1919107858110316886) — zhuzilin
8. [Agentic RL 时代的 Infra 重构](https://zhuanlan.zhihu.com/p/2022786148087464077) — 低级炼丹师
9. [从各家技术文章看 26 年 Agentic RL Infra 优化方向](https://zhuanlan.zhihu.com/p/2007250216227729670) — attack204 · ⏳ 时效型
10. [基于 Ray 的分离式架构：veRL、OpenRLHF 工程设计](https://zhuanlan.zhihu.com/p/26833089345) — 杨远航
11. [Slime 框架深度解析：面向大规模 RL 的训推一体化实践](https://zhuanlan.zhihu.com/p/1921606246454239436) — 曹宇
12. [Agent Infra：Sandbox 技术和选型](https://zhuanlan.zhihu.com/p/1999938129465979624) — ZY lian
13. [Agent sandbox 可能的选型以及 unikernel 的机会](https://zhuanlan.zhihu.com/p/2007212078172234607) — gaocegege
14. [如何系统地分析和定位大模型推理框架的性能瓶颈？](https://www.zhihu.com/question/1993781500349539243/answer/2030665166598174331) — chouheiwa
15. [AI 编译器的概览、挑战和实践](https://zhuanlan.zhihu.com/p/508345356) — 金雪锋
16. [浅谈 AI 编译器趋势](https://zhuanlan.zhihu.com/p/2040199323170951424) — 画饼充饥
17. [Triton 算子开发及编译器资源整理](https://zhuanlan.zhihu.com/p/2018815133590271874) — BobHuang
18. [C++/CUDA/AI-infra 面试经验总结](https://zhuanlan.zhihu.com/p/2005325241803621742) — jinboom · ⏳ 时效型
19. [AI Infra 实习准备记录 01：从三个项目开始](https://zhuanlan.zhihu.com/p/2041895503454015779) — 哼嗯恒 · ⏳ 时效型

### 索引型（1 篇）

1. [LLM Infra 学习资料整理——推理（持续更新）](https://zhuanlan.zhihu.com/p/2036442749319296948) — TensorDance

### 🟣 实战（7 篇）

1. [1.5×提升：PD 分离 KV cache 传输的实践经验](https://zhuanlan.zhihu.com/p/1946608360259577576) — kaiyuan
2. [nanoPD：一个 LLM P/D 分离推理引擎的实现笔记](https://zhuanlan.zhihu.com/p/2026307825358382436) — 暮易
3. [推理框架极简入门：用 Nano-vLLM 搭建知识体系](https://zhuanlan.zhihu.com/p/2008285806222132143) — kaiyuan
4. [2025 最快下手 vLLM 的项目——nanovllm 源码解读](https://zhuanlan.zhihu.com/p/1925484783229698084) — Tiannuo Yang
5. [LLM 推理框架（vLLM/SGLang）入门 Notebook 练习](https://zhuanlan.zhihu.com/p/1999518738303693534) — kaiyuan
6. [深入浅出理解 verl 源码（Part 1）](https://zhuanlan.zhihu.com/p/1920751852749849692) — Chayenne Zhao
7. [浅尝 mini sglang，回顾 nano vllm](https://zhuanlan.zhihu.com/p/1984934838818604631) — Shengguang

---

## 📌 维护说明

- 清单共 35 篇知乎技术内容，经赞同数与内容质量双重筛选；以及 1 篇 Datawhale 实践教程，特别感谢内容共建伙伴 Datawhale 的支持。
- 难度标注为初判，可按实际阅读体验微调。
