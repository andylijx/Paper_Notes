# Preference-based Reinforcement Learning (PbRL)

> Summarized & reorganized by Jingxiao Li
>
> Last update: 24.02.02

[TOC]

**研究问题**：如何利用人类引导完成奖励函数难以定义的任务

**适用场景**：

* 人类只能识别所需的行为，但不一定能够提供演示

* 允许无领域知识的非专家用户进行操作

**核心思路**：根据人类偏好拟合奖励函数，接入RL算法优化当前预测的奖励函数



## 01 Deep Reinforcement Learning from Human Preferences

**从人类偏好中进行深度强化学习**

[NeurIPS'17] [[Paper](https://arxiv.org/pdf/1706.03741.pdf)]

### 研究问题

复杂强化学习系统中的目标表达（奖励构建）问题

### 方法

<img src="https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/DRLHPref_overview.png" style="zoom: 67%;" />

1. 与传统强化学习算法一样，利用策略与环境进行交互，并利用交互数据优化策略，此时还需要为奖励函数的学习打包轨迹数据
2. 将这些轨迹数据进行**成对**地采样并发送给人类，让人类对这些成对的数据进行偏好数据的标注
3. 利用人类标注好的数据，构造如下损失函数，训练奖励回报

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/DRLHP.PNG)

4. 第3步训练的瞬时奖励再给到第1步

### 创新点

1. 使用人类（非专家）对于成对轨迹片段的偏好来将人类的目标传达给强化学习系统，将监督学习（偏好学习）与深度强化学习结合
2. 人类反馈的效率很高，需要很少的人工就能做出很好的效果（提供的人类反馈数目比智能体与环境交互总数的1%还要少）

### 不足

1. 关于选择哪些轨迹片段出来让人类做出偏好标注，文章的方法较为“粗糙”
2. 需要进一步提升学习人类偏好的效率，拓展应用场景

## 02 PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training

**PEBBLE：通过经验重标记和无监督预训练的高效反馈交互式强化学习**

[ICML'21] [[Paper](https://arxiv.org/pdf/2106.05091.pdf)]

### 研究问题

提升对于人类偏好的采样和反馈效率

### 方法

通过无监督的预训练和通过重标记（re-labelling）经验的基于偏好的学习，该方法学习很大程度上是自主的，并由专家提供的二元标签进行补充修正

1. Accelerating Learning via Unsupervised Pre-training

 通过无监督预训练，仅仅通过内在的动机（奖励）来收集多样化的样本数据，在训练开始时就产生信息丰富的查询。方法上利用**状态熵**作为驱动的内奖励来鼓励智能体探索，通过最大化内在奖励和来探索环境并学习产生多样化的行为

2. Selecting Informative Queries

 理论上应该征求专家偏好最大化EVOI (expected value of information)，但是计算EVOI是困难的，因为涉及到对更新的策略产生的所有的可能的轨迹进行期望求和。以往的工作中探索了一些近似方法来采样那些能够改变奖励模型的查询（query）（**这里应该是指人类能够更好评分，两者差异比较明显的轨迹片段对**），包括均匀采样和基于集成的采样（它在集成奖励模型中选择具有高方差的片段对）。本文中利用基于熵的采样，选择令熵最大的轨迹片段对

3. Using Off-policy RL with Non-Stationary Reward

训练过程中，由于奖励函数不断更新可能是不平稳的（non-stationary），本文中利用了一种off-policy算法，通过重用在经验缓冲区的过往经验提供样本高效的学习。并且在奖励模型更新之后，都将重新标记智能体的过往经验，这种方法稳定了学习过程

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/PEBBLE_overview.png)

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/PEPPLE_PC.png)

### 创新点

1. 证明了无监督预训练和off-policy能够显著提升采样和反馈效率
2. 在复杂运动任务和机器人任务中优于基于优先偏好的基准RL
3. 避免了reward exploitation，与直接利用奖励（an engineered reward）训练的agent相比，行为更理想

### 不足

人类专家的水平参差不齐，这一点文章中没有体现，作者在之后的工作中对此进行了改进

## 03 B-Pref: Benchmarking Preference-Based Reinforcement Learning

[NeurIPS'21 Dataset Track] [[Paper](https://arxiv.org/pdf/2111.03026.pdf)]

<img src="https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/PbRL_illustration.png" style="zoom: 80%;" />

环境部分主要沿用了PEBBLE的9个环境，与众不同的是，这篇文章对**human feedback中的非理性人类行为进行了建模，探讨了这些情况对实验结果的影响**

其中非理性人类行为包括：随机（stoc），错误（mistake），跳过（skip），均等（equal），短视（myopic）

<img src="https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/B-Pref_irrationality.png" style="zoom:67%;" />

在多个环境的实验结果表明，mistake和stoc会对实验结果产生较大的负面结果，而其他类型的问题随问题不同而产生不同的效果

## 04 The Expertise Problem: Learning from Specialized Feedback

**专业性问题：从专业的反馈中学习**

[NeurIPS'22] [[Paper](https://arxiv.org/pdf/2211.06519.pdf)]

### 摘要

来自人类反馈的强化学习( Reinforcement Learning from Human feedback，RLHF )是**训练智能体执行难以确定任务**的强大技术。然而，人类的反馈分布可能具备很大的噪声（方差），特别是当人类教师缺乏相关知识或经验时。不同教师的专业知识水平不同，同一教师对同一任务的不同组成部分可能具有**不同的专业知识水平**。因此，从多个教师那里学习的RLHF算法面临一个**专业性问题**：给定一段反馈的可靠性既取决于它来自哪位教师，又取决于该教师对任务相关的组成部分的专业程度。**现有的最先进的RLHF算法假设所有的评价来自相同的分布**，模糊了这种人与人之间和人本身存在的差异，并阻止了解释或者利用这种专业知识的差异。我们对这个问题建模。将其作为现有RLHF基准的扩展来实现，**评估了最先进的RLHF算法的性能**，并探索了改进查询和专家选择的技术。我们的主要贡献是展示和描述了所谓专业性问题，并提供了一个开源的实现来测试未来的解决方案。

### 研究问题

改进查询和专家选择

### 方法

1. 问题建模

将专业性定义为教师根据其潜在偏好对给定轨迹进行可靠评估的能力，文中将教师（Teacher）建模为一个**Boltzmann理性决策者**，理性程度随着查询轨迹的变化而变化：

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/EP_a.png)

其中 β -函数 表示教师的专业领域或者专业水平，RLHF算法不能直接访问这些 β -函数，必须学习哪些教师对哪些查询给出最可靠的反馈。这就是专长问题：选择教师 β_i 和查询 (σ1 , σ2)，以最大限度地提高基于事实奖励函数的性能

2. β -函数建模

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/EP_b.png)

## 05 Exploiting Unlabeled Data for Feedback Efficient Human Preference based Reinforcement Learning

**利用未标记数据反馈有效的基于人类偏好的强化学习**

[AAAI Workshop'23] [[Paper](https://arxiv.org/pdf/2302.08738.pdf)]

### 研究问题

改进PbRL方法

### 方法

1. 在乐观假设下提出了一个三重损失，即人类更倾向于选择一个未标记的轨迹

2. 动作距离损失函数试图构建正在学习的奖励模型的嵌入空间，以反映状态对之间的动作距离

### 创新点

提出两个损失函数，提高反馈效率

## 06 How to Query Human Feedback Efficiently in RL

**如何在 RL 中高效查询人类反馈信息**

[ICML Workshop'23] [[Paper](https://openreview.net/pdf?id=kW6siW4EB6)]

### 研究问题

高效查询人类反馈

### 方法

1. 从环境接收无人类反馈的state-action轨迹，进行实验性设计，获得探索性轨迹

2. 收集人类专家的偏好反馈

3. 从1、2中学习潜在奖励函数，利用轨迹求解奖励函数 (optimal policy)，解决离线RL

### 创新点

1. 设计了一种与奖励无关的轨迹收集算法，用于在过渡动态未知的情况下查询人类反馈算法。

2. 在学习具有线性参数化和未知转换的基于偏好的模型的最优策略时，所需的人工反馈更少。具体来说，框架可以包含线性和低阶 MDP。

3. 研究了基于行动比较反馈的 RLHF，并介绍了一种高效的查询算法

### 不足

1. 本文假设偏好标签遵循Bradley-Terry-Luce（BTL）模型，该模型可能不适用于所有场景

2. 文中使用的轨迹收集过程与奖励无关，这可能会限制其在某些RL场景中的适用性

## 07 Preference Transformer: Modeling Human Preferences Using Transformers for RL

**Preference Transformer: 使用Transformer为 RL 建立人类偏好模型**

[ICLR'23] [[Paper](https://openreview.net/pdf?id=Peot1SFDX0)]

> 论文《Preference Transformer: Modeling Human Preferences Using Transformers for RL》提出了一种名为“Preference Transformer”的神经网络架构，用于基于偏好的强化学习 (RL)，提供了一种使用人类偏好训练智能体的框架。然而，基于偏好的 RL 存在缩放困难，因为它需要大量的人类反馈来学习与人类意图一致的奖励函数。与先前的方法不同，假设人类判断基于 Markovian 奖励，这些奖励对决策的贡献相等，作者引入了一种基于非 Markovian 奖励加权求和的新偏好模型。然后，作者使用Transformer架构设计了所提出的偏好模型，其中包含因果自注意力层和双向自注意力层。作者演示了 Preference Transformer 可以使用真实人类偏好解决各种控制任务，而先前的方法无法工作。他们还表明，Preference Transformer 可以诱导出一个明确定义的奖励，并通过自动捕获人类决策中的时间依赖关系来关注轨迹中的关键事件。代码可在项目网站上找到：https://sites.google.com/view/preference-transformer。

### 研究问题

1. 之前的工作均将奖励函数建模成Markovian的（**仅仅依赖于当前state和action**），而有各种任务的奖励函数是Non-Markovian的，尤其是在preference的研究中，提供给人类的video是有先后顺序的

2. 人类总使用**具有相同权重的奖励总和**来评估轨迹的质量，而在轨迹内进行置信分配（credit assignment）是必要的

### 方法

提出了Preference Transformer (PT)，一个基于非马尔可夫奖励加权和的人类偏好建模的神经网络结构。

PT 以轨迹段作为输入，允许提取任务相关的历史信息。通过叠加双向和因果自注意力（self-attention）层，PT 产生非马尔可夫奖励和重要性权重作为输出。作者用其定义偏好模型，发现 PT 可以产生一个更好形状的（better-shaped）奖励，并注意到来自人类生成偏好的关键事件。

其中需要注意奖励在Linear层就已经进行了预测，后续输出是对轨迹（trajectory）奖励的加权。

在实验部分，作者设计了一系列的评估指标，用于确定 PT 确实是学到了中间的关键步骤奖励，也就是确实进行了置信分配。

<img src="https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/PT_fig1.png" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/PT_fig2.png" style="zoom:80%;" />

### 创新点

为PbRL引入了Transformer框架，考虑了一种基于非马尔科夫奖励加权求和的新偏好模型，用于捕捉人类对更为复杂的任务的偏好（解决非马尔科夫性）
