# SkinIR：视觉重访代码与实验结论

本目录可单独上传到一个 GitHub 仓库，不依赖其他方法仓库。首版范围是Qwen2.5-VL分类与SkinVL工程实验，不包含训练、多模型矩阵和私有医学任务。

**公开前需确认授权：原SkinIR目录未找到项目级许可证，不能直接把所有文件视为已有统一开源许可。** 见 `THIRD_PARTY_NOTICES.md`。

## 目录

- `code/ZS_classification.py`：分类入口。
- `code/transformers550/`：本地改造的Transformers源码与Qwen视觉重访实现。
- `code/llava/`：SkinVL/LLaVA推理及重访适配。
- `code/engineering.py`：候选打分与训练校准相关实现。
- `run.py`、`tools/replay_engineering.py`：6个固定配方的独立运行入口。
- `configs/`：配方与数据整体指纹；`results/`：7行聚合指标及证据来源。
- `docs/REPRODUCIBILITY.md`：方法、数据、复现步骤与结论边界。
- `requirements.txt`、`tests/`、`tools/check_release.py`：依赖和自检。
- `provenance/`：源码来源、路径适配、哈希与验证范围。

`transformers550` 约42 MB，是本地Python源码依赖，不是模型权重。不能只留一个modeling文件或直接替换成官方安装包。

## 结果与结论

| 协议 | 对照 Accuracy | 方法 Accuracy | 结论 |
|---|---:|---:|---|
| Qwen2.5，Patch16全量16类，28039张 | 24.4980% | 24.0379% | 熵触发；Accuracy下降、宏F1略升 |
| Qwen2.5，Patch16子集16类，640张 | 13.1250% | 13.9063% | 显式层18；Accuracy升、宏F1下降 |
| SkinVL，Patch16二分类，2804张，已校准工程基线 | 86.1626% | 86.2340% | 额外重访仅净多判对2张 |

旧SkinVL候选基线为75.6063%，但提升至86.2340%包含自然verbalizer、病理prompt、校准阈值等工程组件，**不能将整段增益归因于视觉重访**。

不同骨干、类别数和子集的数字不能直接排名。此包未核实完整DermNet成绩，不提供该数据集的SkinIR性能结论。

**代码状态：**这是整理时的当前工作源码，不是已经证明与全部历史实验一一对应的冻结版本。后续还需验证历史环境、实际注入路径和完整输出一致性。

## 安装与运行

建议Linux与Python3.12.9。依赖清单是当前可导入环境的版本约束，不是历史实验或干净环境验证的lockfile。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --list
python -m unittest discover -s tests -v
python tools/check_release.py
```

```bash
python run.py --profile skinir_patch16_640_layer18 \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --csv /path/to/Patch16_test_sub580.csv \
  --images /path/to/image_root --output runs/patch16_layer18
```

文件名虽含 `sub580`，历史实际样本数是640，以指纹为准。默认只检查和打印；追加 `--execute --gpu 0` 才运行一组。不要把二分类CSV传给16类配方。

SkinVL工程选择 `skinvl_patch16_engineering`，模型路径改为SkinVL；必要时设置 `SKINIR_CLIP_PATH` 指向合法本地CLIP视觉塔。该配方复用历史阈值，不重新扫描测试集。

模型、原CSV、图像、病例预测和日志未打包。精确复现需另行合法获得历史数据与模型；`--allow-different-data` 仅用于明确标记的小样本或新实验。

## 发布

本目录自包含，不需要其他方法目录。授权确认后仅上传这里的内容，不上传父研究仓库及历史。已有隐私和大文件忽略规则，但自动检查不能替代人工审查。

拆包没有启动新推理或修改原实验。实际检查范围见 `provenance/verification.json`。
