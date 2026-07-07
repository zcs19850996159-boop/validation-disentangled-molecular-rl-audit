# DynaOpt 复现与审计实验指南

> 基于对 MichiganNLP/dynaopt 仓库的实际克隆和代码核查（bandit_alg.py、rl_train.py、
> model_multi.py），不是凭记忆写的。

## 0. 这次要验证的核心问题

DynaOpt论文自己报告的结果里有一个内部不一致，是这次审计的切入点：

| 评测方式 | Uniform Weighted (静态) | DynaOpt (动态bandit) |
|---|---|---|
| 自动化评测 (论文Table 3) | 92.07 | 91.41（更低）|
| 人工评测 (论文Table 4) | 28.29 | 32.10（更高）|

同一个方法，自动化指标下动态权重反而更差，人工评测下动态权重更好——这本身就值得
用你们的审计框架去检验：在自动化 reward/metric 层面，DynaOpt 的动态权重是否真的
利用了有意义的 feedback-specific 信息，还是也像你们 DRD2 实验里那样，主要来自
训练随机性或 bandit 探索机制本身。注意：本 pilot 不审计人工评测优势；若要判断
Table 4 的人工评测结论，需要独立人工评测或其他独立外部 evaluator。

## 1. 底层模型规模，决定算力需求

核查代码发现：
- 生成模型（policy）：`t5-base`（约2.2亿参数）
- reward/评分模型：`gpt2`、`bert-base-uncased`（都是基础规模模型）

**这个规模比你们跑REINVENT4轻量很多**，不需要额外申请更大的GPU资源——你们之前用的
RTX 4080应该完全够用，单卡就能跑，不需要多卡。

## 2. 环境搭建

```bash
git clone https://github.com/MichiganNLP/dynaopt.git
cd dynaopt

conda create -n dynaopt python=3.9
conda activate dynaopt   # 注意README里写的是activate bolt，这是原作者笔误，实际用你自己创建的环境名
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

下载预训练的reflection scorer权重（论文里reward模型之一）：
- 链接在README里（Google Drive），下载后放进 `dynaopt/weights/` 目录

## 3. 跑通基线：先训练warm-start模型

```bash
python supervised_train.py --experiment MI --num_epochs 5
```

记下训练完的模型保存路径（下面记作 `$start_dir`）。

## 4. 你们要对比的核心几组，对应仓库里的具体flag

| 方法 | 命令 |
|---|---|
| Uniform Weighted（静态等权重，对照组） | `python rl_train.py --learning_mode weighted --seed $i --experiment MI_rl --model_start_dir $start_dir` |
| DynaOpt（论文的动态方法，Exp3 bandit） | `python rl_train.py --learning_mode bandit_weighted --seed $i --experiment MI_rl --model_start_dir $start_dir` |
| C-DynaOpt（contextual版本） | `python con_rl_train.py --seed $i --experiment MI_rl --model_start_dir $start_dir` |

## 5. 关键发现：DynaOpt的权重更新信号，100%来自训练reward本身

核查了`bandit_alg.py`里的`Exp3`类——这是经典的EXP3多臂老虎机算法，`__call__(reward, choice)`
直接吃训练时算出的reward去更新每个目标的权重分布。**这里没有任何独立于训练reward
的验证信号**，跟你们DRD2实验里"训练用RF、验证用SVM"的分离设计完全不同——这也
印证了我们之前查文献时的判断：DynaOpt是"用训练信号自己的统计特征做动态调整"，
不是"validation-driven"。

**这个发现直接决定了你们审计实验要怎么设计**（见下一节）。

## 6. 审计实验设计（这是你们真正要做的部分，不是简单复现）

### 6.1 Shuffled-feedback control（不是 random-validation control）

因为 DynaOpt 的 bandit 奖励信号就是训练 reward 本身，这里应称为
**random-feedback / shuffled-feedback control**，不要称为 random-validation control。
具体做法是：**把喂给 Exp3 bandit 的 reward，替换成随机打乱后的 reward**（reward 数值分布不变，但
和"哪个目标产生了这个reward"这个对应关系被打乱）。具体改动点在`rl_train.py`里
调用bandit的地方——建议先搜索`bandit_weighted`这个flag对应的代码分支，找到
`Exp3.__call__`被调用的位置，在那里插入一个可选的shuffle开关。

关键限制：shuffled feedback 只能影响 `Exp3.__call__(reward, choice)` 的权重更新输入，
不允许改变 generator 训练所用的真实 reward、loss 或文本生成流程。优先实现
arm-label shuffle：在同一个 update step 内打乱 reward 与 reward-arm/choice 的对应关系，
保持 reward 数值分布不变，但破坏"哪个目标产生了哪个 reward"的语义关系。如果实现成本允许，
再增加 temporal shuffle：保留每个 reward arm 的边际分布，但打乱时间对应关系。

### 6.2 复现原论文的两种评测方式

- 用`compute_stats.py`跑自动化评测（对应论文Table 3）
- 本 pilot 只使用自动化评测，因此不能声称推翻或验证原论文的人工评测结论。报告中只能写：
  本实验审计 DynaOpt 在自动化 reward/metric 层面的 feedback-specific advantage。
  如果要复现人工评测（论文Table 4），需要独立人工评审员或其他独立外部 evaluator。

### 6.3 多seed重复，别再犯DRD2那次n=5的教训

论文里似乎每个方法只跑了很少的seed（README里`--seed $i`暗示是循环跑多个seed，
但论文正文没有明确写清楚重复次数）。**这次直接从n=8-10起步**，不要再重复"n=5看
起来有效、翻倍后被打平"这个教训。

### 6.4 可以直接复用你们已有的统计工具

`paired bootstrap CI`、`sign test`、`power analysis`这几个脚本，逻辑跟DRD2那边完全
通用，只需要把输入换成DynaOpt这边的评测指标（Reflection score、Fluency、
Coherence），不需要重新开发。

正式多 seed 运行前必须预先指定 primary metric。优先使用原论文 Table 3 的主自动化总分或
aggregate score；如果没有明确总分，则以 reflection score 为 primary metric，fluency 和
coherence 作为 secondary metrics。不得事后挑选最有利指标。统计分析必须以 seed/repeat
为单位，不得按单条文本样本做 bootstrap。

## 7. 预期的产出

- 如果 real（DynaOpt 真实 bandit）vs shuffled-feedback control（打乱 reward-choice 对应关系的 bandit）在
  n=8-10重复下，效应量被压缩到接近0、置信区间跨零——这会是一个和你们DRD2结果
  高度呼应的发现，可以直接写成"审计已发表工作"这一节的核心证据
- 如果 DynaOpt 真的稳定赢过 shuffled-feedback control——这也是有价值的结果，说明至少在这个
  具体任务里，训练信号自身驱动的动态权重是有效的，可以作为你们Discussion部分
  "什么条件下动态权重可能有效"的一个正面对照案例
