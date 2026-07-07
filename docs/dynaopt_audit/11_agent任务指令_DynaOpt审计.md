# 任务说明：复现并审计 DynaOpt（多目标动态权重强化学习方法）

## 背景

我在做一个研究项目：我们此前在REINVENT4分子生成任务上，系统性验证了一个"验证驱动
动态权重"机制，发现在DRD2靶点的两组目标设置下，动态权重都没有比静态等权重展现出
经得起随机对照检验的优势（在分子任务里我们用了 random-validation control、positive-control
sensitivity ladder、paired bootstrap CI、power analysis等一整套审计方法）。

现在我们想把这套审计方法用到另一个已发表的相关工作上——MichiganNLP的DynaOpt
（论文：Min et al. 2024, COLING, "Dynamic Reward Adjustment in Multi-Reward
Reinforcement Learning for Counselor Reflection Generation"，仓库：
github.com/MichiganNLP/dynaopt）。这篇论文用EXP3多臂老虎机算法，根据训练时的
reward反馈动态调整多个目标（reflection质量、fluency、coherence）的权重。

**已知的关键线索**：这篇论文自己报告的结果里，自动化评测（Table 3）显示
DynaOpt(91.41)比Uniform Weighted静态基线(92.07)更差，但人工评测（Table 4）
显示DynaOpt(32.10)比静态基线(28.29)更好——同一个方法在两种评测方式下结论相反，
这是我们审计的切入点。

我已经整理了一份《10_DynaOpt复现与审计指南.md》，里面有：
- 环境搭建步骤
- 我已经核查过的代码结构（`bandit_alg.py`里的`Exp3`类是核心bandit算法，直接吃
  训练reward，没有任何独立验证信号）
- 具体要跑的对比命令（`--learning_mode weighted` 静态基线 vs
  `--learning_mode bandit_weighted` DynaOpt动态方法）

## 当前任务

1. **先把环境装好、跑通baseline**：按指南第2-4节，装环境、训练warm-start模型、
   跑通Uniform Weighted和DynaOpt(bandit_weighted)各一个seed，确认整个流程没问题。

2. **实现 random-feedback / shuffled-feedback control**：在`rl_train.py`里找到`bandit_weighted`
   这个learning_mode对应的代码分支，定位到`Exp3.__call__(reward, choice)`被调用
   的地方，加一个可配置的开关——当开关打开时，把传给`Exp3`的reward替换成同一batch
   内被随机打乱后的reward（reward数值分布不变，但"这个reward对应哪个目标"这个
   映射关系被打乱），其余训练流程不变。

3. **跑多seed对比**：Uniform Weighted、真实DynaOpt、shuffled-feedback版DynaOpt，
   每组至少n=8-10个seed（不要只跑n=5——我们在DRD2实验里吃过亏，n=5时看起来有
   效应，翻倍到n=10后效应被压缩到接近0，方向甚至反转）。

4. **统计分析**：复用我们DRD2项目里已有的统计脚本逻辑（paired bootstrap
   confidence interval、sign test、power analysis），把输入换成这边的自动化评测
   指标（Reflection score、Fluency、Coherence，来自`compute_stats.py`的输出）。

5. **只做自动化评测这一层即可，不需要人工评测**：人工评测需要找人工评审员，
   成本高。本 pilot 只能回答 DynaOpt 在自动化 reward/metric 层面的
   feedback-specific advantage 是否经得起 shuffled-feedback control，不能声称推翻或验证
   原论文的人工评测结论。

## 关键设计要求（务必遵守）

- 不要修改DynaOpt仓库里除了`Exp3`调用点之外的核心训练逻辑，改动应该尽量小、
  局部化，方便别人复查你具体改了什么。
- shuffled-feedback control 的实现必须是"保持reward数值分布不变，只打乱对应关系"，不能是
  简单的"全部设成0"或"全部设成随机数"这种会改变数值分布的做法，否则不是一个公平
  的对照组。
- shuffled feedback 只允许影响 Exp3 bandit 的权重更新步骤，不允许改变 generator 训练所用的真实 reward、loss 或文本生成流程。也就是说，生成模型仍按原始 DynaOpt 训练；只有传入 `Exp3.__call__(reward, choice)` 的 reward 被替换为 shuffled reward。
- 优先实现 arm-label shuffle：在同一个 update step 内打乱 reward 与 reward-arm/choice 的对应关系，保持 reward 数值分布不变，但破坏"哪个目标产生了哪个 reward"的语义关系。如果实现成本允许，再增加 temporal shuffle：保留每个 reward arm 的边际分布，但打乱时间对应关系。
- 在正式多 seed 运行前，必须预先指定 primary metric。优先使用原论文 Table 3 的主自动化总分或 aggregate score；如果没有明确总分，则以 reflection score 为 primary metric，fluency 和 coherence 作为 secondary metrics。不得事后挑选最有利指标。
- 每组实验都要记录原始的per-seed结果（不要只存均值），方便后续做bootstrap和
  sign test。
- 所有实验必须保存 per-seed 原始输出、bandit 权重轨迹、chosen arm、actual reward、bandit-update reward、最终生成文本和 `compute_stats.py` 输出。统计分析必须以 seed/repeat 为单位，不得按单条文本样本做 bootstrap。
- 复现别人的工作要尽量忠实，不要为了让"审计发现问题"这个预设结论更容易成立，而
  在实现上偷工减料或者故意引入偏差——如果你发现某个地方原论文描述不清楚、
  不确定该怎么实现，明确记录下来问我，不要自己猜一个可能对DynaOpt不利的实现。

## 约束

- 先在小规模（比如n=3）跑通验证流程没问题，再扩大到n=8-10的正式规模，避免大规模
  跑完才发现某个环节有bug。
- 每次改动后，先用baseline（Uniform Weighted）确认没有意外改变基线结果，再继续
  往下推进。
- 如果发现原始仓库的README或代码有和我描述不一致的地方（比如flag名称、目录结构
  变了），以你实际看到的代码为准，并告诉我具体的差异。
