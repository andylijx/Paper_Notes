# Efficient Reinforcement Learning for Autonomous Driving with Parameterized Skills and Priors

> Last update: 2023.11.10
>
> Summarized & reorganized by Jingxiao Li

| Conference               | Robotics: Science and Systems (RSS)   |
| :----------------------- | :------------------------------------ |
| **Published Year**       | **2023**                              |
| **Author (Affiliation)** | **U Toronto & Shanghai AI Lab**       |
| **Source**               | https://arxiv.org/pdf/2305.04412.pdf  |
| **Code**                 | https://github.com/Letian-Wang/asaprl |

[TOC]

## 1 Abstract

When autonomous vehicles are deployed on public roads, they will encounter countless and diverse driving situations. Many manually designed driving policies are difficult to scale to the real world. Fortunately, reinforcement learning has shown great success in many tasks by automatic trial and error. However, when it comes to autonomous driving in interactive dense traffic, RL agents either fail to learn reasonable performance or necessitate a large amount of data. Our insight is that when humans learn to drive, they will 1) make decisions over the high-level skill space instead of the low-level control space and 2) leverage expert prior knowledge rather than learning from scratch. Inspired by this, we propose ASAP-RL, an efficient reinforcement learning algorithm for autonomous driving that simultaneously leverages motion skills and expert priors. We first parameterized motion skills, which are diverse enough to cover various complex driving scenarios and situations. A skill parameter inverse recovery method is proposed to convert expert demonstrations from control space to skill space. A simple but effective double initialization technique is proposed to leverage expert priors while bypassing the issue of expert suboptimality and early performance degradation. We validate our proposed method on interactive dense-traffic driving tasks given simple and sparse rewards. Experimental results show that our method can lead to higher learning efficiency and better driving performance relative to previous methods that exploit skills and priors differently. [Code](https://github.com/Letian-Wang/asaprl) is open-sourced to facilitate further research.

当自动驾驶汽车部署在公共道路上使用时，它们会遇到无数不同的驾驶情况。许多人工设计的驾驶政策都难以扩展到现实世界中。幸运的是，通过自主试错，强化学习在许多任务中都取得了巨大的成功。然而，当涉及到**交互、交通流密集**场景下的自动驾驶时，RL智能体要么无法学习到合理的性能，要么需要大量的数据。我们的想法是，当人类学会驾驶时，他们会 **1)** 在high-level的**skill space**上做出决策，而不是在low-level的control space；**2)** 利用**专家的先验知识**，而不是从零开始学习。受此启发，我们提出了ASAP-RL，这是一种用于自动驾驶的高效强化学习算法，同时利用motion skills和专家先验。我们首先用**参数**表示（建模）motion skills，这种表示（模型）足以涵盖各种复杂的驾驶场景和情况。此外我们提出了skill参数逆恢复 (**skill parameter inverse recovery**)方法，将（专家）演示从控制空间 (control space)转换到技能空间 (skill space)。简单有效的双初始化 (**double initialization**)方法旨在利用专家先验，同时绕过了专家次优性和早期性能下降的问题。我们在简单稀疏奖励的交互式密集交通驾驶任务上验证了我们提出的方法。实验结果表明，与以往不同利用技能和先验的方法相比，我们的方法可以获得更高的学习效率和更好的驾驶性能。代码是开源的，以促进进一步的研究。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/figure1-16989457085311.png)

## 2 Method

利用参数化的技能 (parameterized skills)和专家先验 (expert priors)加速连续空间上的强化学习，以便用于密集交通场景的自动驾驶。

### 2.1 Motion Skill Generation

#### 2.1.1 speed profile generation

在时间范围 [0, T] (T: 技能时间窗口 (skill time window))内通过三次多项式 (cubic polynomial)表示。

起点的速度、加速度由当前状态决定 (状态的定义在第3点会介绍) ；终点 (endpoint)的2个参数：速度v~e~ , 加速度 a~e~ 。

这样的速度表示可覆盖多种时域意图，包括加速、减速、急停。

#### 2.1.2 path generation

在自车坐标系下通过三次样条(cubic splines)生成。

起点由当前状态决定；终点由3个参数决定：1）纵向位移x~e~ (longitude position), 2) 横向位移y~e~ (lateral position), 3) 航向角h~e~ (heading angle)。其中考虑到将速度图投影到路径图，路径的长度应大于速度图的长度，所以将x~e~固定为技能时间窗口T内可以达到的最大位移。

这样的路径表示可覆盖多种驾驶意图和操作，包括车道保持、超车、加塞。

#### 2.1.3 parameterized motion skill generation

将速度图投影到路径图。

每一个motion skill是一个自车状态的序列**X** = [**x**~1~, **x**~2~, ..., **x**~T~]，其中**x**~t~是一个元组：**x**~t~ = {x~t~, y~t~, ϕ~t~, v~t~, a~t~}，包括纵向位移、横向位移、航向角、速度、加速度这5个参数，注意这5个参数都指的是对应时间t下的终点的参数。

为了保证skill片段之间连接光滑，规定：1) 决定生成skill的当前状态，是上一执行skill的末状态; 2) 考虑动力学约束 (加速度、曲率)，把规划的参数限制在合理的范围内。

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/formulation-16992740197201.png)

### 2.2 Skill Parameter Recovery

通过RL expert agent收集数据^[P6]^得到专家示范 (expert demonstration)数据集***D***~u~，***D***~u~ = {(s~i~ , u~i~)} 定义在control space上，s~i~表示状态 (state)，u~i~表示控制量 (control action) （例如每一个time step中的转方向盘，油门等），但这样的专家示范缺少skill和奖励信息。所以首先考虑将其转换到skill space上，定义为***D***~θ~ = {(s~i~ , ***θ***~i~)}，θ~i~表示skill的参数。我们要做的是从一个示范的运动技能 (motion skill) **X**~d~，学习得到skill的参数***θ*** 

将motion skill generation中由参数定义得到skill看作是正向 (forward) 生成 (generation)，那么从skill反推得到参数则可理解为是逆向 (inverse) 恢复 (recovery)。

从专家示范中遍历每一条轨迹，将轨迹按（时间）顺序分为长度为T的轨迹片段，得到demonstrated motion skill **X**~d~，再从该motion skill中恢复对应的参数。具体优化方法如下：

采用序列二次规划 (sequential quadratic programming, SQP) 优化下式***θ***，使得**X** (由***θ***组成，可见2.1.3) 逼近**X**~d~

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/actor_pretraining.png)

其中 *f*~s~ 指的是2.1的motion skill generation过程。

同时为了缓解局部最优解问题并提高精确度，在设置不同初始skill参数的情况下多次运行优化过程。

### 2.3 Expert Prior Learning - Actor and Critic Pretraining

在得到***D***~θ~ = {(s~i~ , ***θ***~i~)}后，采用Soft Actor-Critic (SAC) 框架，对actor network和critic network进行预训练。

#### 2.3.1 actor pretraining

首先预训练actor π(***θ***|s) ，获得当前状态下对应的专家示范 (skill)先验。训练目标是将下式最大化：

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/objective_function.png)

其中，(s, θ) 由专家示范 ***D***~θ~ 决定；π(***θ***|s) 的输入是当前状态s，输出skill参数***θ***的高斯分布 (Gaussian distribution)；*H*(**θ**) 指的是熵的正则化项；β指的是熵的权重。

经过预训练的actor可以提供先验知识——哪些skill是更倾向于explore的。

#### 2.3.2 critic pretraining

因为actor与critic在RL训练中是交互的（分别进行policy improvement (PI), policy evaluation (PE)），只训练actor不足以充分利用先验知识，而这点在之前的研究中往往被忽视。目前因为在专家示范中的奖励信息的缺失，critic  Q(s, ***θ***) 并不能通过计算得到，但我们可以从目前已经学习到专家先验的actor入手。从actor中roll out 策略π~φ~，收集携带skill和奖励信息的专家示范 ***D^′^***~θ~ = {(s~i~ , ***θ***~i~ , s^′^~i~ , r)}，奖励r参考稀疏奖励的设置，定义如下：

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/reward_definition.png)

R~progress~: 自车每前进十米，将会得到数值为1的稀疏奖励；

R~destination~: 自车达到目标，将会得到数值为1的奖励；

R~crash~: 自车与他车或与路缘发生碰撞，将会得到数值为-5的惩罚；

R~overtaking~: 自车每超过一辆他车，将会得到数值为0.1的奖励。

之后从***D^′^***~θ~采样得到 (s, ***θ***, r, s′ )，按照SAC的PE预训练critic，更新策略π~φ~ 。

### 2.4  RL over parameterized skill with priors

用预训练得到的权重双初始化actor, critic

借用最大熵强化学习 (maximum-entropy RL) 的目标函数并稍加修改，得到下式：

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/fig3_motion_skill_generation.png)

其中第一项表示在一个长度为T的motion skill里累积的discounted return；*H*(π(***θ***|s)) 指的是熵项，α指的是温度参数（模拟退火）。

通过以上可以学习到一个输出skill参数的策略π(***θ***|s)，再按照2.1 motion skill generation的定义，利用输出的skill参数生成motion skill。

在下一个skill生成前，每一个skill都被跟踪了T time steps——遵循半马尔科夫过程 (semi-MDP process) 。

### 2.5 Pipeline & Pseudo Code

#### 2.5.1 Pipeline

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/pipeline.png)

#### 2.5.2 Pseudo Code

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/pseudo_code-16993468565502.png)

## 3 Experiments

### 3.1 Experiment Setup

1. 仿真平台：MetaDrive 仿真器

自车需要在密集交通流中自主导航，在规定时间内到达目标点

输入为MetaDrive默认的5通道BEV图像（如下图所示）

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/metadrive_default.png)

2. 场景设置：公路、交叉路口、环岛

### 3.2 Results

与基线方法的比较：表现评估分为**三个阶段**，以越来越具体的指标逐步区分方法之间的差异：1）奖励； 2）成功率、道路完成率； 3）碰撞率，以及每episode的超车情况。ASAP-RL **优于**所有其他方法，并且随着从第 1 阶段进入第 3 阶段，ASAP-RL 与其他方法之间的差距不断增加。

#### 3.2.1 Comparison Analysis

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/ablation_study-16993464494121.png)

#### 3.2.2 Ablation Analysis

分别对skill长度T和expert prior进行消融分析

![](https://cdn.jsdelivr.net/gh/andylijx/picGo@main/img/comparison_with_baselines.png)

### 3.3 Visualization

[Demo Video](https://github.com/Letian-Wang/asaprl/blob/main/assets/demo.gif)

## 4 Discussion

### 4.1 Innovation

1. 用参数定义motion skill；
1. 把专家示范（先验）从控制空间转换到skill空间；
2. 强调了actor和critic的双初始化的重要性。

### 4.2 Limitation & Future Work

1. 没有采用真实的驾驶数据；
1. 仅考虑motion skill的长度T (time window/ horizon) 固定的情况；
1. 自车运动轨迹有不“类人”的迹象（not human-like）——或许可考虑在2.1.3加入航向角变化频率的约束。

https://github.com/andylijx/picGo/blob/main/alg1vs2.jpg



