# 分类训练流水线

新入口为项目根目录的 `run_fast.py`。原训练器、模型、数据模块和配置文件不作修改。

在项目根目录执行：

```bash
# 查看实际启动命令，不读取数据
bash cmd_scripts/train.sh --dry-run

# 生产训练：要求已准备 train/val/test 目录，训练仅读取 train 和 val
bash cmd_scripts/train.sh

# 在 MobileMamba 环境，以随机 Tensor 做两轮真实 GPU smoke
bash cmd_scripts/train.sh --synthetic-smoke

# 恢复 smoke；填写已有 latest_ckpt.pth 所在目录，输出写入新的运行目录
conda run --no-capture-output -n MobileMamba python run_fast.py --synthetic-smoke \
  trainer.resume_dir=/absolute/path/to/previous/run trainer.epoch_full=3

# 独立生产评估；val/test/test_net 均不构建训练集和优化器
conda run --no-capture-output -n MobileMamba python run_fast.py -m test \
  model.model_kwargs.checkpoint_path=/absolute/path/to/latest_ckpt.pth
```

CLI 选项必须在 `key=value` 覆盖项之前；train.sh 会将这两类参数分开整理。直接入口和脚本默认均使用 `fft + layeroperator`；加载其他模式的权重时应显式覆盖相同模型参数。配置覆盖不会重新求值旧配置中的派生字段，例如修改类别数时需要同时修改 data、model 和 mixup 的类别数。

新增入口默认启用持久化 workers、预取、非阻塞传输。可通过 `trainer.data.persistent_workers=False`、`trainer.data.non_blocking=False` 对照测试；workers 为 0 时实际关闭持久化，并省略 prefetch_factor。训练损失按全局日志周期与轮末的并集归约，验证补齐样本不计入指标。原逐步非有限值检查及异常训练分支保留，包括原有 `0 * 非有限输出` 的局限。

生产输出默认位于原 checkpoint 根目录的 `fast_pipeline` 下；smoke 默认位于 `runs/smoke_fast_cls`。每次启动创建独立运行目录，恢复也不会覆盖源实验。每轮保存 `latest_ckpt.pth`，按 val top1 更新 `best_epoch*.pth`；训练、验证、测试及 EMA 损失分别记录。

## 实际验证（2026-09-07）

在 conda MobileMamba 中，以标准库 unittest 完成 17 项不同测试，全部通过；采样器修正后的 7 项相关测试重新通过。测试使用真实依赖，没有 mock。本次新增测试文件已按 coding 技能要求清理。

- 配置深拷贝、混合 Namespace/dict 覆盖、错误参数与 split 路径验证。
- 根目录、cmd_scripts 和 /tmp 启动 dry-run，含空格/等号参数、端口验证；实际 train.sh 启动单进程 DDP；bash -n 和 Python 编译检查。
- 临时图片 val/test 变换一致、缺目录与类别映射失败；随机 Tensor workers=0/2 组批、跨两轮 worker PID 复用、旧 RepeatAugSampler 零 batch 检查。
- 独立统计存储和尾窗口、补齐有效计数；两个 CPU gloo 进程验证 N=5 和 N=1 的全局统计。
- NVIDIA GeForce RTX 4060 Laptop GPU 上真实 FMobileMamba_T2，192 输入、100 类、fft/layeroperator、原生 AMP、CutMix、EMA：两轮训练、恢复至第三轮、独立 val/test、checkpoint 元数据、原 checkpoint 未覆盖。
- 真实模型输出仅断言 shape `[2,100]`；阻塞/非阻塞传输输入一致；非有限训练检测、无效评估损失、评估模式恢复、第 401 轮损失记录和幂等清理。

沙箱内 CUDA 不可见，多进程 DataLoader 发生等待；获准在沙箱外运行后，上述 CPU 多进程和真实单卡验证均通过。仅有一张 GPU，未验证双卡 NCCL/DDP 前向；未测真实数据吞吐、精度或性能对照，不报告提速比例。
