# CineDance Match-Cut Retriever

这是 CS180 final project（*Berkeley and World Cinema Through Match Cuts*）里由我负责的**检索器**部分。项目背景见 `docs/proposal.md`，课程要求见 `docs/instruction.md`。`docs/retriever-dev-plan.md` 是早期计划，**和本文件冲突时以本文件为准**。

你的角色：不断提出实现方案、实现、跑出结果给我看，再根据反馈迭代。算法细节由你来提案，产品层面的决定（下面"已定决策"一节）不要擅自改；想改就先问我。

---

## 1. 检索器做什么

上传**任意一张图片**（主要是校园画面，也可以是电影帧或手机照片），**在图上点选一个主体**，系统从 CineDance 电影镜头库里返回 top-K 个镜头：这些镜头里有某个对象和被点选主体**形状相似**，可以和查询画面形成 match cut。

每条结果包含：clip id、镜头编号和起止时间、命中的关键帧时间、命中对象的 mask 和 bbox、对齐所需的变换（平移、缩放、旋转）、分数分解（形状分、变换代价、语义相似度）。

## 2. 已定决策（改动前先问我）

| 方面 | 决定 |
|---|---|
| 查询输入 | 任意图片；**必须点选主体**，用 SAM 单点出 mask（可在多个候选 mask 之间切换） |
| 匹配目标 | 只做**主体之间的形状匹配**。不做构图/布局、运动通道 |
| 两种排序模式 | **硬约束**：候选对象必须能在允许的变换范围内对齐到查询主体的位置和大小，默认缩放 0.8–1.25、平移 ≤ 画面宽度 10%、旋转 ≤ 6°，参数写在配置里，界面上也能改。**软约束**：只看形状相似度，所需变换越大扣分越多，不过滤 |
| 语义滑杆 | 查询主体和候选对象各自用 mask 抠出来算 CLIP 向量，取余弦相似度。界面滑杆从"偏好不同类"（-1）经"不管语义"（0，默认）到"偏好同类"（+1） |
| 镜头切分 | 用官方 annotation 的 `start_sec` / `end_sec`，**但要先测边界准不准**（见第 7 节第 1 阶段）。**库里不对镜头去重** |
| 关键帧 | 每个镜头取 3 帧：镜头时长的 20% / 50% / 80% 处 |
| 库的单位 | **关键帧里的每个对象**都是一条独立索引（一个镜头可能有几十条） |
| 分割模型 | **Mask2Former 全景分割 + SAM2 自动分割，两者合并**（第 3 阶段 bake-off 后选定，2026-11-23）。同一帧里两者 IoU > 0.8 的 mask 只留一个（留 Mask2Former 的，带类别名）。Mask2Former 给完整物体剪影，SAM2 补小物体和部件 |
| 结果合并 | 同一个镜头在结果里只出现一次，保留得分最高的帧和对象；同一个 clip 不限制 |
| 规模推进 | 先用 500 个 clip 开发和调参 → 方法定了再跑全量 → 之后新下载的 clip 增量入库 |
| 界面 | 先做命令行（`--image --x --y`）加结果网格图，再做 **Gradio 界面并开公开 share 链接** |
| 评估 | 自动自测加人工标注（见第 6 节） |
| `auto_matchcut` | **不依赖、不导出 PairConfig**，检索器独立交付。它的代码可以参考，但不 import |
| 日程 | 不赶日程，按正常节奏开发 |

## 3. 数据

### 3.1 视频

- 位置：服务器 `/data0/lvanzhe/CineDance_01/video_live/<video_id>.mp4`。截至 2026-11-24 有 18,423 个真人电影 clip（约 250 小时、490 GB），来自 HF 分片 part0000–part0015。下载任务已于 2026-11-23 因 /data0 空间不足自动停止（剩约 115 GB），暂不继续。
- 这个目录**只读**：不要在 `CineDance_01/` 里写任何东西，也不要干预 `meta/download_more.sh` 和 `split_shard_server.py`。
- `video_review/`（mixed / uncertain，44 个）不入库。`*.part` 是正在写入的文件，跳过。
- `manifest/*.tsv` 和 `meta/labels_v5.csv` 记录了每个 clip 来自哪个 shard、属于哪个标签。增量入库时用它们判断哪些 clip 是新的。
- clip 格式：h264，23.976 / 24 / 25 fps，宽度多为 1920，**高度多为 436–982**，看起来黑边已经裁掉了。每个 clip 10–111 秒（中位数 44 秒），是**多镜头**的叙事片段。
- 数据集是 CineDance-1M 的第 1 部分（[HF](https://huggingface.co/datasets/CineDance/CineDance_01)，[论文 arXiv 2606.09639](https://arxiv.org/abs/2606.09639)），HF 上是门控访问，许可证 **CC-BY-NC-SA**。

### 3.2 官方标注

- HF 上的 `annotation/cinedance-annotation-part1.tar.gz`（1.07 GB），每个 clip 一个 `<video_id>.json`。下载解压到 `/data2/lvanzhe/matchcut_library/raw/annotations/`。
- 服务器上通过 `HF_ENDPOINT=https://hf-mirror.com` 加 `HF_TOKEN` 下载（两者都在 `~/.bashrc` 里）。**token 绝不写进仓库或日志。**
- 用到的字段：
  - `media`：`duration_sec`、`fps`、`width`、`height`、`num_shots`
  - `shots[]`：`shot_no`、`start_sec`、`end_sec`、`shot_type`（`scale` / `angle` / `movement` / …）、`shot_description`、`brief_caption`
- **已核对（第 1 阶段，2026-11-23）**：
  - annotation 的 `width`/`height`/`duration_sec` 和实际文件一致；时间都以秒为单位。
  - dev500 里有 13/500 个 clip 的 `media.fps` 和实际帧率不一致（时长一致），按秒换算帧号时**一律用视频的实际帧率**。
  - 30 个 clip、353 个官方边界：89.5% 与检出的切点同帧，99.4% 在 ±2 帧内，最多偏 3 帧（官方边界偶尔比实际晚 1–3 帧）。官方漏标的硬切约 4 处（约 1%），其中 2 处是 clip 结尾切黑。
  - 结论：官方镜头边界可以直接用；关键帧取在 20/50/80% 处，几帧的偏差没有影响。

### 3.3 检索库（我们生成的）

根目录 `/data2/lvanzhe/matchcut_library/`（**不放在 /data0**，那里的空间留给下载任务）。`raw/annotations/` 放官方 JSON，所有库共用；每个检索库各有一个子目录，即 `configs/local.toml` 里的 `library_root`。开发集是 `dev500/`（从现有 clip 中按固定种子随机抽 500 个，名单在 `dev500/clips.txt`），全量库以后另建 `full/`。每个库目录的结构：

```
clips.txt                 本库包含的 video_id
shots.parquet             clip_id, shot_no, start_s, end_s, shot_type…, caption
keyframes/<clip_id>/<shot_no>_<k>.jpg   分析分辨率的关键帧（长边约 640）
objects.parquet           一行一个对象：object_id, clip_id, shot_no, k, t_s, bbox, centroid, area_frac, 主轴角, mask 的 RLE …
index/shape_fp.npy        形状指纹 (N_obj × D, float16)
index/clip_emb.npy        CLIP 向量 (N_obj × 512/768, float16)
processed_clips.txt       关键帧已抽完的 clip（data.py keyframes 写）
objects_processed.txt     对象已入库的 clip（library build 写），用于对象库增量入库
parts/<clip_id>/          每个 clip 的对象分块（library build 写，consolidate 合并成上面的整库文件）
bakeoff/、eval/           实验输出
```

磁盘预算先按 ≤ 100 GB 控制；第一批 500 个 clip 跑完后按实际占用重新估算全量，报给我。

## 4. 环境、资源与规则

### 4.1 机器

| | 本地（`/home/lazic/Desktop/cs180_final_project`） | 服务器（`ssh lvanzhe@172.18.33.12`，`/data0/lvanzhe/cinedance_matchcut_retriever`） |
|---|---|---|
| 用途 | 改代码，跑单元测试（合成数据）；查看服务器拉回来的帧、拼图等结果 | 改代码，跑真实数据：建库、实验、评估、Gradio |
| 硬件 | 无 GPU，16 核，30 GB 内存，磁盘仅剩约 39 GB | 8× A100 80GB（共享，大多被别人占满），256 线程，440 GB 内存 |
| 数据 | 可以放电影数据（帧、拼图、小样本等），前提是本地磁盘够用；重计算（建库、模型推理）仍在服务器上做 | 见第 3 节 |

- Python 环境：两边都用 `uv` 在仓库内建 **Python 3.11** 的 `.venv`，依赖写进 `pyproject.toml` 并锁版本（`uv.lock` 进 git）。服务器上 PyPI 可以直连。
- 模型权重：服务器连不上 huggingface.co，通过 `hf-mirror.com` 下载（`HF_ENDPOINT` 已在 `.bashrc` 里设置），缓存目录用 `/data2/lvanzhe/hf_cache`。
- 服务器连不上 github.com，git 的 fetch/push 和 Gradio share 都要走代理：`python3 scripts/proxy_run.py -- <命令>`。它会用 `~/clashctl` 订阅 2 里的一个节点，临时起一个只给这条命令用的 mihomo，命令结束就关掉。不要用其他用户的 mihomo 进程，也不要常驻开代理。
  - 例：`python3 scripts/proxy_run.py -- git pull --ff-only`、`python3 scripts/proxy_run.py -- git push`
  - 服务器的 remote 是 SSH（`git@github.com:...`），服务器上的 `~/.ssh/id_ed25519` 已经能以 lazitech 身份认证，`proxy_run.py` 会把 ssh 也转进代理。
  - 服务器上的 `uv` 在 `~/.local/bin`；非交互 ssh 时先 `export PATH=$HOME/.local/bin:$PATH`。
- PyPI：`files.pythonhosted.org` 在两台机器上都很慢或连不上，`pyproject.toml` 已把默认 index 设为清华镜像，`uv.lock` 里记录的也是镜像地址，`uv sync` 直接用即可。

### 4.2 GPU

**每次占用 GPU 之前先问我**：说明用哪张卡（先看 `nvidia-smi`）、预计显存和时长。获批只对那一个任务有效。

### 4.3 磁盘

- `/data0` 只剩约 115 GB（使用率 99%），所以我们的输出一律放 `/data2/lvanzhe/`。
- 大任务开始前先估算输出大小。

## 5. Git 工作流

- 仓库：GitHub [`lazitech/cinedance_matchcut_retriever`](https://github.com/lazitech/cinedance_matchcut_retriever)，**公开仓库**，分支 `main`。本地目录和服务器目录都是这个仓库的工作副本，两边都可以改代码。
- **两边改代码前都先 `git pull`，改完、测完、提交后立刻 `git push`。** 不允许两边同时有未推送的修改。pull 出现冲突就停下来处理，不要用 force push。
- 由你自主提交和推送，不用逐次问我。一个可运行的步骤做完就提交，提交信息用英文、写清楚做了什么。
- **绝对不进 git**：电影帧、缩略图、mask、特征文件、annotation、任何数据集内容（门控数据加 NC-SA 许可证，而仓库是公开的），以及 HF token 和代理配置。`.gitignore` 里要排除 `.venv/`、`data*/`、`outputs/`、`*.npy`、`*.parquet`、`*.mp4`、`*.jpg`（`docs/figures/` 下的自制图除外）、`.playwright-mcp/`、`configs/local.toml`。
- 机器相关的路径（数据根目录、库根目录、缓存目录）写在 `configs/local.toml` 里，不进 git；仓库里提交一份 `configs/local.example.toml` 作模板。
- 文档放在 `docs/`（`proposal.md`、`retriever-dev-plan.md`、`instruction.md`），`CLAUDE.md` 放在仓库根目录。

## 6. 方法（当前方案，可以迭代）

### 6.1 建库

1. **镜头**：从 annotation 读出镜头，丢掉短于 1.0 秒的（阈值可调）。
2. **关键帧**：每个镜头在 20% / 50% / 80% 处各抽一帧（ffmpeg 精确 seek），缩放到长边约 640 px，作为分析分辨率。
3. **分割**：每帧跑 Mask2Former 全景分割和 SAM2 自动分割，合并去重（见第 2 节），得到每帧所有对象的 mask。只保留面积占画面 0.5%–60% 的对象，太碎的去掉；每帧最多保留 40 个（按面积从大到小）。
   - bake-off（dev500 的 1000 帧，A100）：Mask2Former 0.10 秒/帧、平均 5.2 个对象；SAM2 0.90 秒/帧、平均 12.4 个对象。合并后预计每帧约 15 个对象、约 1 秒/帧，全量（约 47 万帧）约 130 GPU·小时，需要分到多张卡上跑。
   - dev500 实测（2026-11-24，3 张共享 A100 各跑 1/3，每张 2.5–3.2 小时）：16,950 帧（约 1% 黑帧/纯色帧被跳过）、294,608 个对象（每帧 17.4 个）；约 9 GPU·小时，即约 1.9 秒/帧（GPU 被别人占用，比 bake-off 慢）。磁盘约 6 MB/clip：index 1.2 GB + parts 1.2 GB + 关键帧 0.6 GB。
4. **对象特征**：
   - 几何：bbox、质心、面积占比、主轴角、偏心率（都按帧宽高归一化）。
   - 形状指纹：把 mask 平移到质心、按 √面积 归一化尺度，栅格化成固定大小（如 64×64），再编码成定长向量（候选：降采样的距离变换、或傅里叶/Zernike 描述子），L2 归一化。
   - 语义：用 mask 抠出对象（背景置灰）后算 CLIP 向量。
5. **增量**：关键帧阶段对照 `processed_clips.txt`、对象阶段对照 `objects_processed.txt` 只处理新 clip；对象阶段每个 clip 写一个 `parts/<clip_id>/`，再用 `consolidate` 重新合并出整库索引。

### 6.2 查询

1. 读入查询图，自动去掉黑边（自己实现，参考 `auto_matchcut` 的 `framing.detect_active_area`），缩放到分析分辨率。
2. 用 SAM2.1 在用户点击处出多个候选 mask，用户选一个。
3. 按第 6.1 节同样的方法算查询对象的几何、形状指纹和 CLIP 向量。

### 6.3 排序

- **软约束模式**：
  1. 用形状指纹的余弦相似度做向量化粗筛，取 top-N（如 2000）。
  2. 用闭式相似变换精排：质心对齐，缩放 = √面积比，偏心率都明显大于某阈值时再按主轴角旋转。在归一化形状空间里算 IoU，再减去 λ·变换代价。
- **硬约束模式**：
  1. 先按几何量预筛：质心距离 ≤ 平移上限，面积比在 [s_min², s_max²] 内。
  2. 再在允许的变换范围内求最佳 IoU，**在画面坐标里算**。两张帧的画幅不一样时，把候选帧 cover-fit 到查询帧的画幅。超出范围的候选直接排除。
  3. 这是 Netflix 论文里"原地 IoU"的推广，见 Chen et al., *Match Cutting*, WACV 2023, [arXiv 2210.05766](https://arxiv.org/abs/2210.05766)。
- **语义项**：最终分 = 几何分 + w·cos(CLIP_q, CLIP_c)，w 由滑杆决定，默认为 0。
- **结果合并**：同一个镜头只保留最高分的那条，然后取 top-K（默认 20）。
- 粗筛用 numpy 或 torch 暴力矩阵乘法即可；对象数到了千万级、延迟不够时再考虑 FAISS。

### 6.4 输出与界面

- **命令行**：`query --image IMG --x X --y Y --mode hard|soft --semantic W --k K`。输出：
  - 结果网格图：查询图、候选关键帧、对象轮廓叠加、对齐后的叠加图、分数
  - `results.json`：完整的分数分解和变换参数
- **Gradio**：
  - 上传图片、点选主体、在候选 mask 之间切换
  - 切换模式、调硬约束参数、调语义滑杆
  - 结果网格；点开某条结果可以播放该镜头的视频片段
  - 用 `share=True` 开公开链接（走代理）

### 6.5 评估

- **自动自测**：从库里随机抽对象，对它的关键帧做轻微扰动（平移、缩放、色彩、JPEG 压缩），在 mask 内部取一个点模拟点击，走完整的查询流程，统计原镜头在结果里的 Recall@1/10/50。硬约束和软约束分别统计。
- **人工标注**：15–20 个查询 × top-10。我做一个标注页（Gradio 或 HTML），你标"能否做成 match cut"。计算 P@10 和"top-10 至少有 1 个可用"的比例；对比硬约束和软约束模式。
- **消融**：
  - 有无变换代价项
  - 不同的形状指纹
  - 不同的分割模型
  - 语义 w 的影响

## 7. 开发阶段（按顺序；每个阶段做完把结果给我看，再进下一个）

| 阶段 | 内容 | 验收 |
|---|---|---|
| 0 仓库与环境 | 文档移进 `docs/`；加 `.gitignore`、`pyproject.toml`、`configs/local.example.toml`；服务器上的空仓库换成 GitHub 的 clone；跑通代理下的 push；两边建 `.venv` | 两边都能 pull、push；本地 `pytest` 能跑 |
| 1 数据核对 | 下载 annotation；抽约 30 个 clip，把官方镜头边界和 ffmpeg / TransNetV2 检出的切点对比，并人工看边界前后 ±2 帧；核对分辨率和时间轴 | 一份边界准确率报告。如果不准，给出替代方案让我选 |
| 2 抽帧 | 500 个 clip 的镜头和关键帧 | 镜头数、耗时、磁盘占用 |
| 3 分割 bake-off | 约 1000 帧上对比 Mask2Former 全景分割和 SAM2 自动分割（**用 GPU 前先问**） | 对比拼图、每帧对象数、速度，交给我选 |
| 4 特征与索引 | 500 个 clip 的对象特征和索引 | 抽查形状指纹的近邻是否合理 |
| 5 查询与排序 | 命令行加两种模式加语义项 | 几个真实查询的结果网格；单元测试覆盖变换范围和合并逻辑 |
| 6 自动自测 | Recall@K 脚本 | 一张数字表 |
| 7 Gradio | 界面加 share 链接 | 我能在浏览器里完成一次查询 |
| 8 全量与增量 | 全量建库，之后增量入库 | 全量统计；一次查询的延迟 |
| 9 人工评估 | 标注页、P@10、消融表 | 评估报告 |

## 8. 明确不做

- 和 `auto_matchcut` 对接、导出 PairConfig、渲染转场
- 构图/布局通道、运动/光流通道
- 库内镜头去重
- 本地重计算：建库、模型推理只在服务器上做（本地只用来看结果、做轻量调试）

## 8.5 后续可尝试（第一版不做）

- **DINOv2 剪影向量**：把对象 mask 画成黑白剪影交给 DINOv2 取向量，作为手工形状指纹之外的第二种粗筛描述子，用自动自测的召回率和人工查看对比。不直接用 RGB 的 DINOv2 向量，因为它按类别组织（钟 → 月亮会离得很远），而且对位置、大小不敏感。
- **BiRefNet 显著性加分**：命中对象与显著区域重叠越多，加分越多（match cut 通常建立在画面主体上）。
- **圆形专用匹配**：参考 `auto_matchcut` 的 `detect_circle`，圆心对圆心、按半径缩放；只在评估发现圆形物体对不准时再做。

## 9. 沟通约定

- 和我说中文；代码、注释、提交信息用英文。
- 每个阶段结束时汇报：做了什么、结果（附图或数字）、遇到的问题、下一步建议。负面结果照实说。
- 碰到不在第 2 节决策范围内、又会影响结果的取舍：给出推荐方案，并说明理由。
