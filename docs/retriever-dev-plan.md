# Match-Cut Retriever 开发计划

> 项目：Berkeley and World Cinema Through Match Cuts（CS180 Final Project）
> 相关文档：`match-cut-retriever-mvp.md`（匹配原理与核心代码）、`auto_matchcut` 仓库（下游剪辑模块）

---

## 1. 目标与边界

**retriever 要做的事**：给定一段校园视频和其中选中的区域，从电影素材库中返回 top-K 个候选电影镜头。每个候选附带时间点、区域 bbox 和分数。输出格式直接就是 `auto_matchcut` 的输入配置（PairConfig）。

**不做的事**（`auto_matchcut` 已经做了）：精确剪切点、最终对齐、Anchor-Lock & Release 渲染、黑边钳制、诊断图。

**原则**：
- 能 `import` 的就不写：几何、去黑边/字幕、结构特征、圆形检测，全部直接调用 `matchcut_edit` 里已有的函数。
- 先打通端到端（查询 → 候选 → 渲染），再加通道。

```
校园视频 + 点选区域（SAM 2.1 单点）
        │
        ▼
 retriever：粗筛（向量化） → 精排（对齐后打分） → 结果网格 → 人工挑选
        │ PairConfig JSON
        ▼
 auto_matchcut：剪切点 + 对齐 + 渲染
```

## 2. 范围

| 功能 | 优先级 | 说明 |
|---|---|---|
| `shape` 通道：形状指纹粗筛 + 闭式对齐精排 | **必须** | mvp 文档里的代码已写好并测过，主要工作是改成剪辑模块的画布约定 |
| 建库：切镜头、去黑边/字幕、BiRefNet 主体 mask | **必须** | 每个镜头取中间帧即可，剪切点交给剪辑模块去搜 |
| 导出 PairConfig + 调用 `auto_matchcut` | **必须** | 打通这一步才算"检索 + 对齐原型" |
| 最简 Gradio 界面 | **必须** | 选帧、点选、结果网格、导出 |
| `circle` 通道（钟面、车轮、月亮……） | 应该做 | `detect_circle` 现成，建库时多存 3 个数 + 一个排序函数 |
| `layout` 通道（拱廊、窗列等没有单个主体的建筑画面） | 应该做 | 照搬 `prune_pairs` 的 20 维结构描述子；DINOv2 全局向量可选 |
| 主轴角（小角度旋转） | 可以做 | 十几行代码 |
| `Anchor.mask_path`（让剪辑模块接收精确 mask） | 可以做 | 改 `auto_matchcut` 约 20 行；不改就传 bbox，也能用 |
| SAM 2 自动分割进库、CLIP"意外感"项、FAISS、多帧采样、多通道自动合并 | 不做 | 成本高、收益小；库只有几千条，界面上手动切换通道即可 |

## 3. 接口约定

### 3.1 输出

每个候选输出一份 PairConfig：

```json
{
  "video_a": "campus/sather_gate_03.mp4",
  "video_b": "library/clips/<uuid>.mp4",
  "anchor_time_a_s": 4.25,
  "anchor_time_b_s": 2.10,
  "match_type": "shape",
  "anchor_a": {"bbox_xyxy_norm": [0.41, 0.22, 0.63, 0.88]},
  "anchor_b": {"bbox_xyxy_norm": [0.12, 0.30, 0.37, 0.95]},
  "edit": {"search_window_s": 1.0}
}
```

另外写一份 `retrieval.json`（剪辑模块不读），记录排名、分数分解和初始变换 M，供评估和展示。

### 3.2 和剪辑模块共用的约定（直接 import，不复制常量）

| 约定 | 取值 | 来源 |
|---|---|---|
| 缩放 / 旋转范围 | 0.8–1.25、≤6°；两边都有可靠圆时缩放放宽到 1.7，且 B 的放大不超过原始分辨率 1.5 倍 | `schema.EditParams` |
| 画布 | `match_a`：保持 A 的真实画幅；B 用 cover-fit 放进画布 | `framing.plan_canvas`、`geometry.cover_fit` |
| 黑边 | 不允许；按 4 个角点判断可行性 | `geometry.corners_covered` |
| 只变换 B，A 保持不变 | — | 剪辑模块的规定；mvp 文档里的 `cover_campus` 删掉 |
| 真实画面 | 先去黑边、屏蔽字幕带 | `framing.detect_active_area` / `detect_subtitles` |

## 4. 数据

| 数据 | 规模 | 说明 |
|---|---|---|
| 校园视频 | 15–25 段，每段 5–15 s | 拍主体清晰的画面：钟面、Sather Gate 拱门、自行车轮、窗列、台阶、走向镜头的人；**同一场景拍远、中、近三种景别**（缩放范围只有 0.8–1.25，多景别能大幅提高对上的概率） |
| CineDance-1M 子集 | 1 个 TAR（约 39 GB，约 1,600 个 clip，按总数估算） | HF gated，需要申请；仅限非商业使用 |
| 手动精选片 | 10–20 段 | 按想要的意象（月亮、拱廊、圆盘、窗户）收集，补随机子集里缺的类型；CineDance 审批下来之前先用这批开发 |

库规模估计：约 1,600 clip × 约 3 镜头 × 1 帧 ≈ 5,000 条，粗筛用 numpy 即可。

## 5. 要做的东西

### 5.1 `build_library.py`：建库
- 切镜头（PySceneDetect `AdaptiveDetector`），过滤掉太短、放不下 hold + release 的镜头
- 去黑边、识别字幕带（`framing`）
- 取中间帧，保存缩略图
- BiRefNet 主体 mask，用 `mask_reliable` 过滤（GPU 服务器上跑）
- 每条记录存：形状指纹（1024 维）、质心、面积、bbox、mask png
- （应该做）`detect_circle` 的圆心/半径/可靠性；20 维结构描述子
- 存储：`library/index.npz`、`library/meta.csv`、`library/masks/`、`library/thumbs/`

**验收**：随机抽 20 个镜头查看——黑边被裁掉、字幕区被屏蔽、mask 落在主体上的比例过半。

### 5.2 `core.py`：匹配核心
- 把 mvp 文档的代码改成 cover-fit 画布 + 角点可行性；缩放/旋转范围从 `EditParams` 读取
- `shape` 通道：指纹余弦 − α·代价（向量化，不可行的设为 −inf）→ top-300 → 闭式对齐后 IoU − α·代价
- （应该做）`circle` 通道：两边都有可靠圆 → 圆心对圆心、缩放 = 半径比，按代价排序
- （应该做）`layout` 通道：20 维结构描述子余弦 → top-300 → `Aligner.structure_score` 精排
- （可以做）主轴角：两个形状都明显细长（偏心率 > 0.3）时才加旋转
- 同一镜头去重，只保留最佳一条

**验收**：合成测试中，精排 top-10 全部落在粗筛 top-300 内；超出缩放范围或会露黑边的候选不出现在结果里；代码里没有硬编码的 1.25。

### 5.3 `query.py` + `export.py`：查询与导出
- 命令行：`--video --t --x --y --channel` → SAM 2.1 单点出 mask → top-K
- 导出 PairConfig（通过 `schema.validate`）和 `retrieval.json`
- 一键调用 `auto_matchcut render`

**验收**：一个查询从输入到渲染完成，不需要手动改任何文件。

### 5.4 `app.py`：Gradio 界面
- 选校园视频、拖动选帧、点选区域（显示 SAM mask，可在多个候选 mask 中切换）
- 切换通道（shape / circle / layout）
- 结果网格：原帧、对齐后叠加图、分数、所需缩放；超出范围时提示"需要 X 倍缩放"
- 勾选候选 → 导出 → 渲染

**验收**：查询延迟 < 10 s（不含模型首次加载）。

### 5.5 `eval.py`：评估
- 标注格式：CSV（查询 id、候选 id、能否做成 match cut）
- 计算 P@10、"top-10 至少有 1 个可用"的比例、下游可用率（被 `auto_matchcut` 判为非 fallback 的比例）
- 对选中的转场，计算对齐前后的区域 IoU、质心距离、面积比

### 5.6 `tests/test_core.py`
- 沿用 mvp 文档里的合成形状测试，加上角点可行性和缩放范围的测试

### 5.7 （可以做）`auto_matchcut` 的 `mask_path`
- `Anchor` 增加可选字段 `mask_path`，读到后替代 BiRefNet saliency；需要和维护者确认

## 6. 评估内容

| 指标 | 怎么测 | 参考目标 |
|---|---|---|
| 检索相关性 | 15–20 个查询 × top-10，自己标注；再请一位同学独立标注其中 5 个查询，看一致率 | P@10 ≥ 0.3；每个查询 top-10 至少有 1 个可用的比例 ≥ 80% |
| 下游可用率 | top-10 中被 `auto_matchcut` 判为非 fallback 的比例 | ≥ 70% |
| 对齐质量 | 对选中的转场，测对齐前后的区域 IoU、质心距离、面积比 | 对齐后 IoU 中位数 ≥ 0.6 |
| 消融 | 有/没有代价项；有/没有去黑边和字幕；`shape` vs `layout` 在建筑类查询上的对比 | 一张对比表 |
| 用户研究 | 硬切 / 仅对齐硬切 / 完整流程，每人看 6 组；评价连续性、意外感、视觉吸引力、是否注意到剪切点 | ≥ 10 人 |

> 目标值是凭经验定的参考，第一版评估出来后再调整。

## 7. 开工前要确认的 2 件事

1. 成片画幅：每个转场跟随校园画面的画幅（`match_a`），还是统一 16:9（`fill`）？这决定了画布约定。
2. `auto_matchcut` 如果是别人维护的，是否同意加 `mask_path`？不同意就只传 bbox，不影响主流程。
