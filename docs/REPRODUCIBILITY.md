# SkinIR方法与复现指南

## 1. 实现与版本

分类入口为 `code/ZS_classification.py`。Qwen实现位于 `code/transformers550/models/qwen2_5_vl/modeling_qwen2_5_vl.py`，LLaVA/SkinVL适配位于 `code/llava/memvr_llava.py`。

方法缓存视觉表征，支持熵触发与显式目标层重访。当前Qwen默认采用top-10归一化熵，不能直接等同于论文的全词表熵。视觉种子拟合到旁路投影维度，进行幅值匹配与融合；实际文本 `Qwen2MLP` 分支不能笼统视为所有架构统一的SiLU旁路。

LLaVA当前熵触发路径可进入视觉token重写。仅有 `injection_mode=ffn` 配置名，不足以证明历史上执行的是FFN旁路。

**当前源码快照不是经过历史哈希证明的所有旧实验版本。** 拆包未改算法，但代码与旧结果等价性仍需独立核查。

## 2. 数据、环境和固定配方

| 系列 | 数据 | 模型 | 主要参数 |
|---|---|---|---|
| `skinir_patch16_full_*` | 28039张，16类 | Qwen2.5-VL-7B | batch8；熵窗口5–16、阈值0.60、重访比例0.5 |
| `skinir_patch16_640_*` | 640张，16类 | 同上 | batch4；显式层18、比例0.5 |
| `skinvl_patch16_*` | 2804张，2类 | SkinVL-PubMM | 自然verbalizer、病理prompt、固定margin阈值；重访窗口5–12、阈值0.6、比例0.1 |

Qwen配方seed42、greedy、beam1、max_new_tokens32，保持各自batch。Base默认使用安装版Transformers，方法使用本地 `transformers550`，这一后端差异不能隐藏。若统一后端做新公平对照，应另立新实验。

CSV需包含 `image`、`categories`，使用固定targets标签拼写。具体数量、CSV哈希与观测标签在 `configs/datasets.json`；实际候选顺序以入口的实验targets为准。不要把二分类数据传给16类配置，不要依赖 `set` 自动筛Test。

历史640张CSV名称含 `sub580`，不能凭文件名改成580张。CSV有绝对路径时迁移会改变字节哈希，需明确记录映射，不能默认为同一原文件。

环境：建议Python3.12.9；`requirements.txt` 是当前导入环境的版本约束，不是历史lockfile，尚未在全新环境安装验收。本地 `transformers550` 不能被pip安装包替代。模型加载路径决定精度，不能把所有SkinVL/Qwen分支一概称为相同BF16设置。

## 3. 执行

```bash
python -m pip install -r requirements.txt
python run.py --list
python run.py --profile skinir_patch16_640_layer18 \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --csv /path/to/Patch16_test_sub580.csv --images /path/to/image_root \
  --output runs/layer18
```

默认只预检和打印命令，追加 `--execute --gpu 0` 才实际运行一组。新输出不覆盖历史，不自动附带Base。`--allow-different-data` 用于小样本或新数据，结果须明确标记而不能混入历史主表。

SkinVL工程使用 `skinvl_patch16_engineering`，需要SkinVL与匹配视觉塔；设置 `SKINIR_CLIP_PATH=/path/to/clip` 可提供原视觉塔的本地路径。默认离线。工程配方调用原评分实现，不重新用Test选阈值或扫参。

历史阈值为 **−1.1640625086426735**，来自1999张训练校准样本。自然verbalizer为 `non-tumor tissue` / `tumor tissue`，取首token分数。输出 `scores.csv` 和 `metrics.json`，前者是病例级材料，应保持私有。

## 4. 已保存结果

| 协议 | 配方 | n | Accuracy | 宏F1 |
|---|---|---:|---:|---:|
| Qwen，Patch16全量16类 | Base | 28039 | 24.4980% | 9.3291% |
| 同上 | 熵触发 | 28039 | 24.0379% | 9.7648% |
| Qwen，Patch16子集16类 | Base | 640 | 13.1250% | 8.0975% |
| 同上 | 显式层18 | 640 | 13.9063% | 7.5607% |
| SkinVL，Patch16二分类 | 旧候选Base | 2804 | 75.6063% | 61.5858% |
| 同上 | 已校准工程Base | 2804 | 86.1626% | 84.3512% |
| 同上 | 工程组合＋重访 | 2804 | 86.2340% | 84.4233% |

聚合文件保留原始精度与正确数。层位和协议敏感；Accuracy升不代表宏F1也升。工程组相对已校准Base仅增加2张正确预测，不能把旧候选Base到最终结果的全部变化归因于SkinIR。

不同骨干、类别数和数据子集不能直接排序。没有本包范围内核实的DermNet完整成绩，不把其他方法数字借用为SkinIR结果，也不将论文草稿数值当本地实测。

## 5. 验收与发布边界

- 检查退出码、实际预测数、图片/真值顺序、prompt、类别解析和指标；如有原始合法预测，再逐条比较。
- `provenance/source_snapshot.json` 固定当前源码，不能证明它与全部历史结果完全对应。
- 运行记录与新增日志保存在忽略的 `runs/` 或私有目录；包内不含图像、模型、原CSV、病例预测。
- 不因复现差异而删样本、换Base或改指标；先核对视觉塔、精度、版本、batch和注入路径。
- 没有多种子、患者级独立统计或临床有效性证据，不能宣称稳定泛化或临床安全。
- 项目级授权尚需确认。自动自检不是法律或隐私审核；详见 `THIRD_PARTY_NOTICES.md`。
