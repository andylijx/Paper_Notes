# RLIF: Interactive Imitation Learning as Reinforcement Learning

> Summarized & reorganized by Jingxiao Li
>
> Last update: 23.12.12
>

| Publisher                | arXiv                                                        |
| :----------------------- | :----------------------------------------------------------- |
| **Published Date**       | **23.11.21**                                                 |
| **Author (Affiliation)** | **Sergey Levine, Yi Ma (UC Berkeley)**                       |
| **Source**               | https://arxiv.org/pdf/2311.12996.pdf; https://rlif-page.github.io/ |
| **Code**                 | https://github.com/pd-perry/RLIF                             |

[TOC]

## 1 Abstract

Although reinforcement learning methods offer a powerful framework for automatic skill acquisition, for practical learning-based control problems in domains such as robotics, imitation learning often provides a more convenient and accessible alternative. In particular, an interactive imitation learning method such as DAgger, which queries a near-optimal expert to intervene online to collect correction data for addressing the distributional shift challenges that afflict naïve behavioral cloning, can enjoy good performance both in theory and practice without requiring manually specified reward functions and other components of full reinforcement learning methods. In this paper, we explore how off-policy reinforcement learning can enable improved performance under assumptions that are similar but potentially even more practical than those of interactive imitation learning. Our proposed method uses reinforcement learning with user intervention signals themselves as rewards. This relaxes the assumption that intervening experts in interactive imitation learning should be near-optimal and enables the algorithm to learn behaviors that improve over the potential suboptimal human expert. We also provide a unified framework to analyze our RL method and DAgger; for which we present the asymptotic analysis of the suboptimal gap for both methods as well as the nonasymptotic sample complexity bound of our method. We then evaluate our method on challenging high-dimensional continuous control simulation benchmarks as well as real-world robotic vision-based manipulation tasks. The results show that it strongly outperforms DAgger-like approaches across the different tasks, especially when the intervening experts are suboptimal. Additional ablations also empirically verify the proposed theoretical justification that the performance of our method is associated with the choice of intervention model and suboptimality of the expert. Code and videos can be found on the [project website](rlif-page.github.io)

尽管强化学习方法为自动技能获取提供了强大的框架，但对于机器人等领域中基于学习的实际控制问题，模仿学习通常提供了更方便、更容易获得的替代方案。 特别是像 DAgger这样的交互式模仿学习方法，它询问近乎最优的专家在线干预以收集校正数据，从而解决困扰naive行为克隆的分布转移 (distributional shift) 挑战，在理论和实践中都有良好的性能，而无需人工奖励函数及强化学习方法的其他组成部分。在本文中，我们探讨了off-policy强化学习如何在**与交互式模仿学习相似但可能更实用的前提**下提高性能。我们提出的方法使用强化学习，并**将用户干预信号本身作为奖励**。这放宽了交互式模仿学习中的干预专家应该接近最优的假设，并使算法能够学习比潜在的次优人类专家更好的行为。我们还提供了一个统一的框架来分析我们的  RL方法和 DAgger；为此，我们提出了两种方法的次优差距的渐近分析以及我们方法的非渐近样本复杂度界限。 随后，我们在具有挑战性的高维连续控制仿真基准以及现实世界中基于机器人视觉的操作任务上评估我们的方法。 结果表明，它在不同的任务中都**远远优于**类似 DAgger 的方法，**特别是当干预专家不是最优的时候**。额外的消融实验验证了所提出的理论依据，即我们的方法的性能与干预模型的选择和专家的次优性相关。代码和视频可以在项目网站上找到。

> https://zhuanlan.zhihu.com/p/668858737
>
> 这篇论文主要介绍了交互式模仿学习作为一种比直接行为克隆更为便利和可行的方法，在实际的学习基于控制问题中，尤其是在机器人领域。通过使用强化学习的离策略方法，该方法使用用户干预信号本身作为奖励，使得算法可以学习到比人类专家更优的行为，而不需要手动指定奖励函数和其他全面强化学习方法的组成部分。此外，论文还提供了分析该论文的RL方法和DAgger的统一框架，同时给出了两种方法的非渐进样本复杂度上界。实验结果表明，在具有挑战性的高维连续控制仿真基准测试和实际的基于机器人视觉的操作任务中，该论文的方法明显优于类似DAgger的方法，特别是在干预专家为次优时。

## 2 Motivation & Insight

### 2.1 Motivation

相比于强化学习，模仿学习 (Imitation Learning, IL) 方法因其简单方便等特点，在机器人等领域更为常用。模仿学习一个经典方法——行为克隆 (behavioral cloning)，在使用的过程中会因为误差的累积从而导致covariate (distributional) shift现象出现。

为解决这一问题，交互式模仿学习 (Interactive Imitation Learning, IIL) 例如DAgger及其变种，通过在线咨询 (query)专家动作，以监督学习的方式完成模型的交互式训练。但这种方法依赖于干预是接近最优的 (near optimal) 这一设定，而人类示范通常不是最优的，并且这种情况下机器的表现不会超过专家的表现。

### 2.2 Insight

在交互式模仿学习的设定下，**干预决策本身可以视作是一个奖励信号**。

作者放宽了IIL的设定——**不假设干预行为是最优的**。通过对直接引起人类干预的动作标注一个负奖励值，利用RL最大化奖励从而减少干预次数，以满足人类预期。作者将他们的方法称为Reinforcement Learning via Intervention Feedback (RLIF) 。

学习策略的最终表现并不会因为专家干预的次优性而受到影响，而是通过**避免干预发生**从而向最优策略逼近。

## 3 Method

只对干预介入的前一瞬时转移（s~t-1~, a~t-1~, s~t~）标注负奖励值为-1，其余为0。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/fig1.png)

放宽交互式模仿学习关于最优干预的设定，采用更*natural*的设定：专家对**何时干预**的决策，与学习策略的次优性有关。

### 3.1 Algorithm

#### 3.1.1 Interactive Imitation vs RLIF

标红部分体现两种算法的主要差异。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/def_a_4.png)

#### 3.1.2 Practical Implementation

考虑到数据包含on-policy样本和潜在次优干预，需要off-policy RL算法高效利用先验数据和在线经验，作者采用近期提出的RLPD算法^[1]^。RLPD是一种基于SAC的off-policy RL算法，可以将离线数据应用到在线强化学习。具体可参考[原文](https://arxiv.org/pdf/2302.02948.pdf)及[开源代码](https://github.com/ikostrikov/rlpd)。

### 3.2 Intervention Strategies

为了从干预中学习，需要干预专家决策**何时干预**来传达有关任务的信息。作者对仿真中的各种干预策略进行建模，并通过实验验证RLIF方法相对于"何时干预"这一设定的稳定性。

#### 3.2.1 Random Intervention

*Baseline Strategy*：随机均匀干预，每一个timestep概率相同。

在实验中，作者考虑30%, 50%, 及85%的干预率（干预率=干预下的timestep数与总timestep数之比），定义i为专家干预瞬时的特定timestep，k为专家干预瞬时后的接管过程的timestep数。在任意非干预timestep t，不同概率的随机均匀干预的定义如下。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/eq5_1.png)

#### 3.2.2 Value-Based Intervention

Value-Based Intervention 假定：当在专家动作和机器人动作之间存在一个差异δ（与参考的值函数Q^πref^ (reference value function) 有关）时，专家有概率为β进行干预。这个模型体现了一个随机的专家，他可能有 1) 一个用于选取动作的策略π^exp^（这些动作可能高度次优）,以及 2) 一个值函数Q^πref^和其相应的策略π^ref^，用于判断机器人是否表现良好和决策何时干预。

注意策略π^ref^可能比π^exp^要好很多，例如一个人类专家或许能正确地判断机器人在执行任务中的动作不好，但他自己的策略π^exp^对于任务而言可能也不够好。

δ表示干预专家的信心水平：δ越小，专家越倾向于干预，即使是对一些稍微次优的机器人动作。作者将此模型如下式规范化。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/eq_a_2-17023022435321.png)

在实际中，作者发现下式的一个相对阈值比较 (relative threshold comparison)也是有效的。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/alg1vs2.jpg)

实验中，作者将β设置为接近1的值，如0.95；将α设置为接近1的值，如0.97。

#### 3.2.3 Real Human Intervention

鉴于上述模型可能不足以充分表示真实的人类行为，作者用真实的人类干预评估RLIF方法是否反映真实人类干预，如下图示。操作员碰到手柄瞬间开始接管机器人，松手则将控制权归还给机器人。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/fig2.png)

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/table2_offline_datasets-17023092359452.png)

机器人接受视觉反馈，作者采用ImageNet预训练的EfficientNet-B3^[2]^作为视觉backbone进行快速策略学习。

## 4 Experiments

主要寻求三个问题的答案：(1) 干预奖励对于RL学习有效的策略而言是否充足；(2) 相比于DAgger，RLIF的表现如何，尤其是在次优专家的干预下；(3) 不同干预策略对于实验表现的影响。

### 4.1 Experiment Setup

#### 4.1.1 Offline Datasets

所有仿真任务的初始数据集都来自D4RL^[3]^提供的数据集的下采样。

真实世界机器人实验的初始数据集来自人工采集的次优轨迹的下采样。

下表列出了用于下采样的特定数据集和各个任务的初始数据集的大小。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/table3_ablation_results.png)

#### 4.1.2 Expert Training

根据任务的不同，不同水平 (level)的专家在 BC, IQL, SAC, 或 RLPD 数据集上进行训练。通过对各种大小的人类数据集或专家数据集下采样进行训练来获得各种水平的专家。对于每个级别和任务，使用同一个专家从 RLIF, DAgger, 和 HG-DAgger 的所有干预策略中进行干预。

### 4.2 Simulation Results

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/table1_comparison_exp.png)

### 4.3 Robot Experiment

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/fig6.png)

### 4.4 Ablation Results

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/main_theoretical_results-17023108045703.png)

## 5 Theoretical Analysis

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/fig5.png)

## 6 Discussion

在与交互式模仿学习设定极为相似的情况下，提出了一种从干预中学习的强化学习方法。

### 6.1 Contributions

1. 不依赖于最优专家（干预）；
2. 无需真实奖励函数，而是从专家关于干预时机的决策中推导奖励信号。

### 6.2 Limitations

1. 需要一种可以实际在off-policy数据 (rollout data)和on-policy数据 (expert interventions) 上训练的RL方法；
2. 研究人员青睐模仿学习的原因是其完全无需在线部署，而交互式模仿学习本身有些“画蛇添足”；
3. 未来需要对这种在线学习方法的安全性问题开展研究。

## 7 References

[1] Philip J. Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 1577–1594. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/ball23a.html.

[2] Mingxing Tan and Quoc Le. EfficientNet: Rethinking model scaling for convolutional neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 6105–6114. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/tan19a.html.

[3] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020b.
