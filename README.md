# MARCH

完整 SOC 哈密顿矩阵的掩码重建预训练初版。默认选择约 4,000 个晶体，按材料/等价结构分组为约 3,500 训练、250 验证、250 测试；目标掩码比例 30%，仅优化重建损失。此阶段提供预训练及可复用编码器，不包含下游性质训练。

## 本机运行

已在 WSL Ubuntu 创建独立 conda 环境 **MARCH**，并在 RTX 4070 上完成真实数据冒烟测试。环境位置为 `/home/wang_zy/miniconda3/envs/MARCH`。

```bash
cd /mnt/d/prjs/MARCH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate MARCH
bash scripts/runSmoke.sh
# 离线模式：
bash scripts/runSmoke.sh --wandbMode offline
```

`runSmoke.sh` 只准备配置中明确列出的 8 个源文件，划分为 6 个训练、2 个验证材料，并运行至多 8 个优化步骤；不会加载 4,000 个材料。它对准备、统计和训练使用同一配置，可通过 `CONFIG` 或 `--config` 指定配置，其余参数传给训练入口。

新机器安装：

```bash
bash scripts/setupEnvironment.sh
# 无 GPU 的登录节点可显式只安装依赖；之后在 GPU 节点运行默认检查：
MARCH_SKIP_CUDA_CHECK=1 bash scripts/setupEnvironment.sh
```

安装脚本需要已有 conda；非默认位置可设置 `CONDA_BIN=/path/to/conda`。默认检查 CUDA、scatter 反传和 cluster GPU 算子。核心依赖沿用 NextHAM：PyTorch 2.2.0/CUDA 12.1、e3nn 0.5.6、PyG 2.6.1、timm 1.0.17 及匹配的 scatter/cluster wheels。`environment.yml` 提供 conda 引导，`requirements.txt` 提供直接依赖，`requirements.lock.txt` 固定实际验证的完整依赖。

## 服务器预训练与恢复

复制项目到服务器并安装环境。复制示例配置，按服务器实际路径修改 `sourceRoot`、`cacheRoot` 等字段：

```bash
cp configs/server.example.yaml configs/server.yaml
# 编辑 configs/server.yaml 后，三个阶段使用同一配置：
python prepareData.py --config configs/server.yaml
python fitStatistics.py --config configs/server.yaml
CONFIG=configs/server.yaml NPROC_PER_NODE=4 bash scripts/runMultiGpu.sh
# 同配置、同卡数恢复：
CONFIG=configs/server.yaml NPROC_PER_NODE=4 bash scripts/runMultiGpu.sh --resume runs/serverPretrain/latest.pt
# 离线服务器：
CONFIG=configs/server.yaml NPROC_PER_NODE=4 bash scripts/runMultiGpu.sh --wandbMode offline
# 最终测试集只在显式调用时读取：
python evaluate.py --config configs/server.yaml --checkpoint runs/serverPretrain/best.pt --split test
```

准备和统计流程只执行一次，不由各训练 rank 重复执行。紧凑缓存、清单、统计文件及恢复检查点应在各节点相同路径可见。多机在各节点设置相同的 `NNODES`、`NPROC_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT` 及各自的 `NODE_RANK`，运行同一脚本。训练不硬编码 GPU 数量；验证分片不填充重复样本。训练每 epoch 最多丢弃 worldSize-1 个随机尾部材料以对齐 rank，并记录实际丢弃数。

默认每卡一次处理一个完整晶体，通过 `gradientAccumulation` 累积材料梯度。主模型为四层、392 维隐藏表示；smoke 仍为四层，使用 98 维。大图显存取决于全部有效块数量，不会静默截断长程块。初版使用 FP32。主宽度模型已在真实小图上验证前反向；大图和真正多卡的显存规模需在服务器测量。

检查点保存模型、优化器、调度器、逐 rank 随机状态、epoch 内样本偏移、清单、冻结统计和 wandb ID。恢复要求相同 worldSize、数据清单及统计，允许修改输出目录、运行名、日志设置和 `maxOptimizerSteps`；其余训练语义须保持一致。CUDA scatter 的数值非确定性意味着不承诺逐位一致。

## 模块边界和可替换接口

| 模块 | 责任 |
| --- | --- |
| `march/data` | 18 字段加载、周期图、结构划分、保守闭包、可见参考、冻结统计 |
| `march/codec` | 官方 SOC 分解/逆变换、壳层排列、有效 irrep 元数据、官方旋转 |
| `march/models` | H/S 输入编码、四层 NextHAM 骨架、TraceGrad、编码器接口 |
| `march/heads` | 独立完整 SOC 重建头，可替换或扩展性质头 |
| `march/losses` | 重复副本合并、Hermitian 投影、纯重建分层损失 |
| `march/optim` | 独立优化器和调度器工厂 |
| `march/training` | 单卡/DDP、累积、验证、完整状态恢复 |
| `march/logging` | rank 0 wandb 在线/离线和本地 JSONL |

`prepareSample` 返回隔离的 `modelInput` 与 `target`。`ReconstructionModel.forward(modelInput)` 默认只返回 `qPrediction`；`encode` 或显式 `returnFeatures=True` 可供后续读出。模型拒绝输入白名单之外的字段。重建头预测归一化的全部 5,832 个实 SOC 系数，覆盖四个复自旋块和 J=0…7；损失负责逆归一化和物理投影。最终节点特征合入重建上下文，确保末层节点更新也受到监督。

自有脚本和普通函数采用 lowerCamelCase；Python 特殊方法、pytest 注入参数和第三方源码保留框架原名。`thirdParty/nextHam/UPSTREAM.json` 记录来源、版本和 24 份原始源码/许可证文件的哈希；上游文件保持原样，适配代码置于 `march/`。`mp_properties.csv` 已复制到项目根目录，预训练不读取性质标签。

## 数学与数据约定

- 原始文件为已审核的 18 字段列表，只使用结构、元素、边、H[15]、S[16] 和支持[17]；内存映射避免加载保留在文件中的大型本征向量。紧凑缓存不保存谱、H0、deltaH、费米能等 H 派生输入。
- 原始矩阵为 spin-major `[E,54,54]`，空间基组为四 s、两 p、两 d、一 f 壳层。官方分解使用 shell-pair-major `[E,4,729]`。编码器和头的张量积为全部高阶 SOC 通道提供数值及梯度通路，包括 onsite。轨道支持显式编码为 81 个壳层对偶标量。
- 周期边满足 `edgeVec = T @ lattice + position[dst] - position[src]`。全部保存的有效块参与消息传递，没有距离专家或按 H 幅值构图。支持由轨道元数据决定；padding 不监督，真实零值监督。
- `maskGroup` 只依赖几何、元素和轨道支持，联合 Hermitian、重复和候选空间群副本；按有效块数选择最接近目标比例的闭包。对称性强的材料可能无法精确达到 30%，日志报告实际比例；无法形成可见/隐藏集合时明确排除。默认不使用时间反演。
- `lossOrbit` 由完整轨道—自旋 H/S 协变验证得到，仅供监督分组。实际电子态不满足某几何对称性时仍保守联合遮蔽，不对其施加空间群投影，也不通过该协变检查决定输入或抽样。
- `cVis` 仅从可见 H/S 求得，固定块权重为 1。隐藏 H 输入严格为零，不填入 `-cVis*S`。参考分母太小时重采样；必要时仅依赖 S 选择最近的可行闭包比例，无法满足则报错，不读取隐藏 H 决定掩码。
- 归一化仅用训练集拟合 RMS；同一无序元素—径向壳层对的全部 m、J、自旋、实虚部及反向副本共享尺度。未见分组使用冻结的训练集全局尺度，不减去分量均值。
- 预测恢复物理尺度并官方逆变换后，先平均重复存储副本，再做 Hermitian 投影，再官方分解。损失按有效 irrep 的维度与尺度归一化，依次平均块内 irrep、等价成员、等价轨道、存在的 onsite/offsite 类型以及材料。
- TraceGrad 用作前三层特征增强，不添加迹或范数辅助损失；验证/冻结编码器也保留前向所需特征导数，不能用 `torch.inference_mode()` 包裹整个模型。

## 优化、日志与验收

主预设参考 NextHAM 实际训练路径：Adam，lr=5e-4，betas=(0.9,0.999)，eps=1e-8，weightDecay=0，100 epochs；timm cosine，warmup 5 epochs 从 1e-6 开始，最低 1e-5，默认不裁剪梯度。优化器模块将配置值传入实际优化器。smoke 关闭 warmup 并限制步骤数。

wandb 默认 project=MARCH、mode=online；凭据从标准环境变量或已有登录读取，项目不保存密钥。配置可设 `offline`，随后用 `wandb sync <离线目录>` 同步。只记录指标、配置和运行元数据，不上传原始 H/S。`runs/...` 保存 resolvedConfig、metrics.jsonl、wandbRun、latest/best 检查点及 summary；best 只由验证重建损失选择。

实际验收：39 项单元/物理测试通过；真实 CUDA 8 步训练及续训至 10 步通过；主宽度 CUDA 前反向通过；双进程 CPU DDP 梯度一致性、梯度累积和恢复通过；固定真实小样本 25 步损失从 1.01535 降至 0.31182。详细证据见 [验收记录](reports/validation.md)。

```bash
python -m pytest
MARCH_SMOKE_SOURCE=/mnt/y/Materials_HAM_SOC_DATABASE/mp-1.pth python -m pytest tests/testData.py
python tests/checkModelGpu.py
python tests/checkOverfit.py
# CPU 双进程验证，包含梯度累积和不完整的尾窗口：
OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 train.py --config configs/cpuDdpSmoke.yaml
# 已有 GPU 冒烟检查点的离线续训示例：
python train.py --config configs/smoke.yaml --resume runs/smokeOnline/latest.pt --outputDir runs/smokeResume --wandbMode offline --maxOptimizerSteps 10
```

本机只有一张 RTX 4070。真正多 GPU/NCCL 运行及完整 4,000 材料加载/训练尚未执行，按本阶段要求留到服务器。
