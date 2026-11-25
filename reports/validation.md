# 验证记录

验收日期：2026-10-23。按 ResearchPlanV2 的完整 SOC 块掩码重建约束实现初版训练代码；本机仅进行有界冒烟，没有加载或训练完整 4,000 材料。

## 环境与来源

- WSL Ubuntu，独立 conda 环境 `/home/wang_zy/miniconda3/envs/MARCH`；Python 3.10.22。
- NVIDIA GeForce RTX 4070，PyTorch 2.2.0+cu121，CUDA runtime 12.1，e3nn 0.5.6，PyG 2.6.1，timm 1.0.17，wandb 0.28.0。
- `pip check`、CUDA 矩阵前反向、torch_scatter GPU 前反向、torch_cluster GPU radius graph 均通过。
- 21 项直接依赖与 104 项完整锁定依赖匹配。脚本使用 LF，bash 语法检查通过。
- NextHAM 版本 `2a022290b83569e9b46a6d97ecb11334150993dd`，24 份源码/许可证文件 SHA256 全部匹配 `thirdParty/nextHam/UPSTREAM.json`；适配代码独立存放。已移除随上游目录带入的旧 Python 字节码。
- `mp_properties.csv` 已复制至根目录；性质标签没有进入预训练或模型选择。

## 数据范围

只加载以下 8 个真实材料：mp-1、mp-10、mp-100、mp-1000、mp-10000、mp-10004、mp-10006、mp-1056418。全部通过格式、轨道支持、周期反向边和掩码可行性检查，无排除项。最大样本 mp-10004 包含 1,520 个保存块，全部保留。

紧凑缓存位于 `/home/wang_zy/.cache/MARCH/smoke`；清单和冻结统计位于 `runs/smokePreparation`。划分为 **6 训练、2 验证、0 测试**，8 个结构组互异；statistics 的 trainIds 与训练分区逐项匹配。全部统计仅拟合这 6 个训练材料。

这 8 个样本的最大 Hermitian 误差不超过 5.7e-14。保守掩码组与已验证监督轨道保持独立，例如 mp-10000 为 23/25，mp-10004 为 115/232。目标比例为 30%；闭包离散约束下实际比例可能偏离，例如 mp-1 为 40%，而验证集平均约为 29.88%。

## 测试与运行证据

| 项目 | 实际结果 | 证据 |
| --- | --- | --- |
| SOC codec 与模型物理测试 | 18 passed，0 skipped，195.06 秒 | `reports/modelCodecTests.xml` |
| 数据与分层损失 | 19 passed，20.52 秒，含真实 mp-1 | `reports/dataTests.log` |
| 优化器参数与检查点状态 | 2 passed，2.49 秒 | `reports/trainingTests.xml` |
| CUDA 真实数据完整训练链路 | 8 个 Adam 步骤、2 次验证、有限损失/梯度；峰值分配显存 1,558.97 MiB | `reports/gpuSmokeOnline.log`，`runs/smokeOnline/summary.json` |
| CUDA 断点续训 | 从 step 8、epoch 1/offset 2 继续至 step 10、offset 4；优化器状态均为 step 10 | `reports/gpuResumeOffline.log`，`reports/checkpointAudit.json` |
| 主宽度模型 CUDA 前反向 | 392 维、四层、完整 SOC 输出 5,832 维，真实 mp-1 全部 15 块；loss 1.64793，梯度有限，末层节点梯度非零 | `reports/mainGpu.json` |
| 固定真实样本优化检查 | smoke 四层、mp-1 固定输入/掩码，25 步 loss 1.01535 → 0.31182，下降 69.29%，全部梯度有限 | `reports/overfitGpu.json` |
| DDP/串行材料梯度对照 | 2 个 CPU rank，完整四层模型与 TraceGrad，2 个 Adam 步骤；最大梯度误差 9.69e-8 | `reports/cpuDdpConsistency.json` |
| 真实 CPU DDP 训练 | 2 个 rank，累积窗口 2，step 1 用 4 个材料、尾窗口 step 2 用 2 个材料；验证共 2 个材料 | `reports/cpuDdpTraining.log` |
| 真实 CPU DDP 恢复 | 2 → 3 步，epoch 1/offset 0 → 2；两个 rank 的随机状态齐全，优化器状态均为 step 3 | `reports/cpuDdpResume.log`，`reports/checkpointAudit.json` |
| wandb 在线 | SDK 确认同步完成 | [在线冒烟运行](https://wandb.ai/wangzywave13-zhejiang-university/MARCH/runs/n4qbz9by) |
| wandb 离线 | 续训日志保存为本地 offline-run，未自动上传 | `runs/smokeResume/wandb/offline-run-20261023_042428-n4qbz9by` |

合计 **39 项单元/物理测试通过**，另完成表中独立集成运行。源码测试覆盖：完整自旋块往返、J≤7 梯度路径、联合旋转/反演/原子重排、轨道支持、末层节点监督、冻结 TraceGrad、重复副本及 Hermitian 投影、手算层次损失、padding/真实零值、参考能量平移、闭包比例、S 分母可行性兜底和隐藏真值隔离。

浮点等变检查使用误差容限：官方高 J 旋转/分解的 float64 对照容限 2e-5，完整 float32 模型旋转检查容限 2e-3。没有将浮点近似误差表述为逐位等变。隐藏真值隔离检查直接比较前向输入数值；模型白名单禁止目标字段进入 forward。

CUDA 最终验证重建损失为 6.54433，raw MAE 为 0.394928（数据原始单位）。这些值来自极小冒烟划分，仅用于确认训练链路工作；固定样本损失下降也只证明优化能够进行，不构成泛化或性质迁移结论。

## 脚本与恢复检查

- `runSmoke.sh` 的 CONFIG、`--config`、`--config=` 三种配置入口，对准备/统计/训练生效一致；其他参数仅传给训练。5 种有效路径及 3 种缺失配置错误路径经 mock 检查。
- `runMultiGpu.sh` 单机/多机 torchrun 参数及参数转发经 mock 检查。
- `setupEnvironment.sh` 默认要求 CUDA；隐藏 GPU 时默认失败，显式 `MARCH_SKIP_CUDA_CHECK=1` 可仅安装依赖，并明确报告没有验证 CUDA。
- `server.example.yaml` 继承主模型、4,000 总样本、30% 掩码及 NextHAM 优化器参数；源数据、缓存、清单、统计、输出使用服务器示例路径。
- 实际 GPU 与 DDP 恢复检查确认模型参数已更新、优化器步数递增、清单/统计不变、逐 rank 随机状态齐全。CUDA scatter 不保证逐位确定性。

## 验收边界

本机只有一张 RTX 4070，**真正多 GPU/NCCL 训练未执行**。DDP 的本机证据是两个 CPU/Gloo 进程，服务器脚本已提供。主宽度模型的 CUDA 测试仅使用真实小图；smoke 宽度运行覆盖全部 8 个所选样本，不据此推断主宽度大图显存。

**完整 4,000 材料的加载、尺度拟合和预训练尚未运行**，符合用户明确限定的本机冒烟范围。服务器需先修改配置路径并依次执行准备、统计、训练。下游冻结探测、弹性/介电少标签对照及分布外实验属于后续研究阶段。
