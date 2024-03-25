# 具身智能 Embodied Artificial Intelligence (EAI)

注：本篇整理主要基于文章 [具身智能 | CCF专家谈术语]( https://mp.weixin.qq.com/s/lA9q6-hCIBldaEbWYHvkRQ)

整理人：李京晓  Jingxiao Li



具身智能并非是一个新概念——1950年，图灵在其为人工智能奠基、提出图灵测试的经典论文 *Computing Machinery and Intelligence* ^[1]^的结尾展望了人工智能可能的两条发展道路：一条路是聚焦抽象计算（比如下棋）所需的智能，另一条路则是为机器配备最好的传感器、使其可以与人类交流、像婴儿一样地进行学习。这两条道路逐渐演变成了非具身和具身智能。原文如下。

> We may hope that machines will eventually compete with men in all purely intellectual fields. But which are the best ones to start with? Even this is a difficult decision. Many people think that a very abstract activity, like the playing of chess, would be best. It can also be maintained that it is best to provide the machine with the best sense organs that money can buy, and then teach it to understand and speak English. This process could follow the normal teaching of a child. Things would be pointed out and named, etc. Again I do not know what the right answer is, but I think both approaches should be tried. 



在具身智能的发展道路上，人们思考和探讨人工智能系统是否需要拥有与人类相似的身体和感知能力，以及身体如何影响智能和认知。早期的具身智能研究主要集中在机器人学和仿生学领域，逐渐发展并融合了跨学科的方法和技术。近年来，随着深度学习等技术的快速发展，具身智能研究进入了一个新的阶段。研究人员利用虚拟物理环境和强大的计算能力，设计和训练具备感知和行动能力的智能系统，并将这种交互能力迁移到真实世界、使智能体进行自主决策和执行物理交互任务^[2]^。



## 定义

根据中国计算机协会计算机术语审定委员会，具身智能是指一种基于物理身体进行感知和行动的智能系统，其通过智能体与环境的交互获取信息、理解问题、做出决策并实现行动，从而产生智能行为和适应性^[2]^。



## 与其他人工智能概念的差别

非具身智能聚焦于智能中表征与计算的部分。早在符号主义大行其是的六七十年代，非具身智能就占据了绝对的优势。不需要物理交互、不考虑具体形态、专注抽象算法的开发这一系列有利条件使得非具身智能得以迅速地发展。今天在算力和数据的支持下，深度学习这一强有力的工具大大推进了人工智能研究，非具身智能已经如图灵所愿、近乎完美地解决了下棋、预测蛋白质结构等抽象的独立任务。互联网上充沛的图片和语义标注也使得一系列视觉问题取得了突出的成果。然而这样的智能显然是有局限的。非具身智能没有自己的眼睛，因此只能被动地接受人类已经采集好的数据。非具身智能没有自己的四肢等执行器官，无法执行任何物理任务，也缺乏相关的任务经验。即使是可以辨识万物的视觉大模型也不知道如何倒一杯水，而缺乏身体力行的过程，使得非具身智能体永远也无法理解事物在物理交互中真实的意义。相比而言，具身智能具有支持感觉和运动的物理身体，可以进行主动式感知，也可以执行物理任务，没有非具身智能的诸多局限性。更重要的是，具身智能强调“感知—行动回路” (perception-action loop) 的重要性，即感受世界、对世界进行建模、进而采取行动、进行验证并调整模型的过程。这一过程正是“纸上得来终觉浅，绝知此事要躬行”，与我们人类的学习和认知过程一致^[3]^。



文章[2]进一步指出具身智能任务与其他线上AI (Internet AI) 任务具有完全不同的范式，即：基于一个具身智能体（如机器人），通过该智能体的“看”、“说”、“听”、“动”、“推理”等方式，与环境进行交互和探索任务目标，从而解决环境中的各项挑战性任务。论文[4]指出具身智能理论框架包括：感知(perception)，行动(action)，记忆(memory)，学习(learning)。



## 关键任务或技术

完稿于21年的综述论文[5]将具身智能的基本研究任务归纳为视觉探索 (visual exploration)、视觉导航 (visual navigation)、具身问答(embodied question answering)。视觉探索通过收集关于 3D 环境的信息，通过运动和愿感知，以更新其内部环境模型；视觉导航是在有无外部先验或自然语言指令的情况下，将三维环境导航到目标。具身问答则聚焦于如何在物理实体的状态下完成QA任务。



文章[2]对具身智能的重点任务作了进一步的整理：具身智能一条主要的研究思路是在虚拟物理世界设计和开发具身智能算法，并将其迁移到真实世界 (Sim2Real)。现阶段的重点任务主要包括具身导航、问答和包括物体重排列在内的、多种多样的物体操纵任务等。这些任务的研究内容和已有学科存在重叠但又有所侧重。具身智能范式下的导航^[6]^侧重从交互中完成导航目标，包括点目标、物体目标、指令目标、声音导航等，需要智能体通过看、听、语言理解等方式主动探索周围物理环境完成目标，针对具身导航的虚拟环境主要有iGibson^[7,8]^系列环境、Habitat^[9,10]^、MultiON^[11]^、BEHAVIOR^[12,13]^等。具身问答是导航任务的升级，侧重从交互中探索和理解周围环境，并关联语言指令和回答特定问题，主要虚拟环境有ALFRED^[14]^。具身重排^[15,16]^则是智能体将物理环境中的物体从初始构型转移到目标构型，一般以家居场景为主。这类任务不关注机器人和物品的接触交互控制等底层机器人技术，更加关注对场景的理解^[17]^、物品整体状态感知和任务规划，主要虚拟环境包括AI2-THOR^[18]^、ThreeDWorld^[19,20]^、Habitat 2.0^[21]^等。机器人物体操纵是机器人领域的重要研究内容，具身智能视角下的机器人操作侧重以学习的方式解决如何交互并从接触交互中理解、控制和改变外界状态，实现机器人操作的任务可迁移性、环境适应性和技能可扩展性^[22]^，主要虚拟环境包括SAPIEN^[23-25]^、RLBench^[26]^、VLMbench^[27]^、RFUniverse^[28]^、ARNOLD^[29]^等，物体抓取及操纵信息数据集包括：GraspNet^[30]^、SuctionNet^[31]^、DexGraspNet^[32]^和GAPartNet^[33]^等。此外，具身智能的另一条研究思路是在真实世界中采集具身交互数据和学习。在真实环境中，人类可以遥操作机器人来采集专家数据，通过包括behavior cloning （行为克隆）等模仿学习算法来训练机器人习得技能或交互策略^[35,36]^。借用高采样率的强化学习算法（如基于模型的强化学习或离线强化学习^[34]^），机器人也可以直接与真实世界交互、获得奖励而习得交互策略。在模仿学习数据方面RH20T^[37]^提供了一个20TB级别的大规模多模态模仿学习数据。



未来具身智能的研究将主要关注：1) 训练具身智能体的仿真平台搭建；2) 从仿真环境到真实世界的迁移(Sim2Real)；3) 真实世界中的实时交互学习；4) 与大语言模型的结合。



## 参考资料

[1] Turing A M. Computing machinery and intelligence[M]. Springer Netherlands, 2009.

[2] [具身智能 | CCF专家谈术语]( https://mp.weixin.qq.com/s/lA9q6-hCIBldaEbWYHvkRQ)  2023-07-22 《中国计算机学会》公众号

[3] [冬至 | 浅谈具身人工智能](https://mp.weixin.qq.com/s/9H7zHofxGFQ2rpKRvxNx2Q) 2022-12-22 《北京大学前沿计算研究中心》公众号

[4] Paolo G, Gonzalez-Billandon J, Kégl B. A call for embodied AI[J]. arXiv preprint arXiv:2402.03824, 2024.

[5] Duan J, Yu S, Tan H L, et al. A survey of embodied ai: From simulators to research tasks[J]. IEEE Transactions on Emerging Topics in Computational Intelligence, 2022, 6(2): 230-244.

[6] Driess D, Xia F, Sajjadi M S M, Lynch C, Chowdhery A, Ichter B, Wahid A, Tompson J, Vuong Q, Yu T, others. Palm-e: An embodied multimodal language model[J]. arXiv preprint arXiv:2303.03378, 2023. 

[7] Shen B, Xia F, Li C, Martín-Martín R, Fan L, Wang G, Pérez-D’Arpino C, Buch S, Srivastava S, Tchapmi L, Tchapmi M, Vainio K, Wong J, Fei-Fei L, Savarese S. iGibson 1.0: A Simulation Environment for Interactive Tasks in Large Realistic Scenes[C]//2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). , 2021: 7520–7527.

[8] Li C, Xia F, Martín-Martín R, Lingelbach M, Srivastava S, Shen B, Vainio K E, Gokmen C, Dharan G, Jain T, others. iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks[C]//Conference on Robot Learning. , 2022: 455–465.

[9] Savva M, Kadian A, Maksymets O, Zhao Y, Wijmans E, Jain B, Straub J, Liu J, Koltun V, Malik J, others. Habitat: A platform for embodied ai research[C]//Proceedings of the IEEE/CVF international conference on computer vision. , 2019: 9339–9347.

[10] Ramakrishnan S K, Gokaslan A, Wijmans E, Maksymets O, Clegg A, Turner J, Undersander E, Galuba W, Westbury A, Chang A X, others. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai[J]. arXiv preprint arXiv:2109.08238, 2021.

[11] Wani S, Patel S, Jain U, Chang A, Savva M. Multion: Benchmarking semantic map memory using multi-object navigation[J]. Advances in Neural Information Processing Systems, 2020, 33: 9700–9712.

[12] Li C, Zhang R, Wong J, Gokmen C, Srivastava S, Martín-Martín R, Wang C, Levine G, Lingelbach M, Sun J, others. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation[C]//Conference on Robot Learning. , 2023: 80–93.

[13] Srivastava S, Li C, Lingelbach M, Mart\’\in-Mart\’\in R, Xia F, Vainio K E, Lian Z, Gokmen C, Buch S, Liu K, others. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments[C]//Conference on Robot Learning. , 2022: 477–490.

[14] Shridhar M, Thomason J, Gordon D, Bisk Y, Han W, Mottaghi R, Zettlemoyer L, Fox D. Alfred: A benchmark for interpreting grounded instructions for everyday tasks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. , 2020: 10740–10749.

[15] Weihs L, Deitke M, Kembhavi A, Mottaghi R. Visual room rearrangement[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. , 2021: 5922–5931.

[16] Batra D, Chang A X, Chernova S, Davison A J, Deng J, Koltun V, Levine S, Malik J, Mordatch I, Mottaghi R, others. Rearrangement: A challenge for embodied ai[J]. arXiv preprint arXiv:2011.01975, 2020.

[17] Ma X, Yong S, Zheng Z, Li Q, Liang Y, Zhu S-C, Huang S. SQA3D: Situated Question Answering in 3D Scenes[J]. arXiv preprint arXiv:2210.07474, 2022.

[18] Kolve E, Mottaghi R, Han W, VanderBilt E, Weihs L, Herrasti A, Deitke M, Ehsani K, Gordon D, Zhu Y, others. Ai2-thor: An interactive 3d environment for visual ai[J]. arXiv preprint arXiv:1712.05474, 2017.

[19] Gan C, Schwartz J, Alter S, Schrimpf M, Traer J, De Freitas J, Kubilius J, Bhandwaldar A, Haber N, Sano M, others. ThreeDWorld: A platform for interactive multi-modal physical simulation[J]. Advances in Neural Information Processing Systems (NeurIPS), 2021.

[20] Gan C, Zhou S, Schwartz J, Alter S, Bhandwaldar A, Gutfreund D, Yamins D L K, DiCarlo J J, McDermott J, Torralba A, others. The threedworld transport challenge: A visually guided task-and-motion planning benchmark towards physically realistic embodied ai[C]//2022 International Conference on Robotics and Automation (ICRA). , 2022: 8847–8854.

[21] Szot A, Clegg A, Undersander E, Wijmans E, Zhao Y, Turner J, Maestre N, Mukadam M, Chaplot D S, Maksymets O, others. Habitat 2.0: Training home assistants to rearrange their habitat[J]. Advances in Neural Information Processing Systems, 2021, 34: 251–266.

[22] Cewu Lu, Shiquan Wang. General Purpose Intelligent Agent, [J]. Engineering, 2020, 6(03): 40–52.

[23] Xiang F, Qin Y, Mo K, Xia Y, Zhu H, Liu F, Liu M, Jiang H, Yuan Y, Wang H, Yi L, Chang A X, Guibas L J, Su H. SAPIEN: A SimulAted Part-Based Interactive ENvironment[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). , 2020: 11097–11107.

[24] Mu T, Ling Z, Xiang F, Yang D C, Li X, Tao S, Huang Z, Jia Z, Su H. ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations[C]//Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). , 2021.

[25] Gu J, Xiang F, Li X, Ling Z, Liu X, Mu T, Tang Y, Tao S, Wei X, Yao Y, Yuan X, Xie P, Huang Z, Chen R, Su H. ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills[C]//The Eleventh International Conference on Learning Representations, 2023.

[26] James S, Ma Z, Arrojo D R, Davison A J. IEEE, 2020. Rlbench: The robot learning benchmark & learning environment[J]. IEEE Robotics and Automation Letters, 2020, 5(2): 3019–3026.

[27] Zheng K, Chen X, Jenkins O C, Wang X. Vlmbench: A compositional benchmark for vision-and-language manipulation[J]. Advances in Neural Information Processing Systems, 2022, 35: 665–678.

[28] Fu H, Xu W, Xue H, Yang H, Ye R, Huang Y, Xue Z, Wang Y, Lu C. Rfuniverse: A physics-based action-centric interactive environment for everyday household tasks[J]. arXiv preprint arXiv:2202.00199, 2022.

[29] Gong R, Huang J, Zhao Y, Geng H, Gao X, Wu Q, Ai W, Zhou Z, Terzopoulos D, Zhu S-C, others. ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes[J]. arXiv preprint arXiv:2304.04321, 2023.

[30] Asif U, Tang J, Harrer S. GraspNet: An Efficient Convolutional Neural Network for Real-time Grasp Detection for Low-powered Devices[C]//IJCAI. 2018, 7: 4875-4882.

[31] Cao H, Fang H S, Liu W, et al. Suctionnet-1billion: A large-scale benchmark for suction grasping[J]. IEEE Robotics and Automation Letters, 2021, 6(4): 8718-8725.

[32] Wang R, Zhang J, Chen J, et al. Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation[C]//2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023: 11359-11366.

[33] Geng H, Xu H, Zhao C, et al. Gapartnet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 7081-7091.

[34] Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643.

[35] Mandlekar, A., Xu, D., Martín-Martín, R., Zhu, Y., Fei-Fei, L. and Savarese, S., 2020. Human-in-the-loop imitation learning using remote teleoperation. arXiv preprint arXiv:2012.06733.

[36] Zhang, T., McCarthy, Z., Jow, O., Lee, D., Chen, X., Goldberg, K. and Abbeel, P., 2018, May. Deep imitation learning for complex manipulation tasks from virtual reality teleoperation. In 2018 IEEE International Conference on Robotics and Automation (ICRA) (pp. 5628-5635). IEEE.

[37] Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Junbo Wang, Haoyi Zhu, Cewu Lu ,RH20T: A Robotic Dataset for Learning Diverse Skills in One-Shot, RSS workshop 2023.

