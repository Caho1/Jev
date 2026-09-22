
---
displayName: RiSAWOZ
labelTypes:
- SemanticSegMap
- Chinese Corpus
license:
- CC BY-NC 4.0
mediaTypes:
- Text
paperUrl: https://arxiv.org/pdf/2010.08738v1.pdf
publishDate: "2020"
publishUrl: https://terryqj0107.github.io/RiSAWOZ_webpage/
publisher:
- Soochow University
- Tianjin University
tags:
- Annotation
taskTypes:
- Natural Language Generation
- Slot Filling
- Dialogue State Tracking

---
# 数据集介绍
  ## 简介
   为了缓解多领域数据的短缺并为面向任务的对话建模捕获话语现象，我们提出了 RiSAWOZ，这是一个具有丰富语义注释的大型多领域中文绿野仙踪数据集。 RiSAWOZ 包含 11.2K 人对人 (H2H) 多轮语义注释对话，超过 150K 话语跨越 12 个域，比以前所有带注释的 H2H 对话数据集都要大。单域对话和多域对话都构建，分别占65%和35%。每个对话都带有全面的对话注释，包括自然语言描述形式的对话目标、领域、对话状态以及用户和系统方面的行为。除了传统的对话注释外，我们还特别提供了对话中话语现象的语言注释，例如省略号和共指，这对于对话共指和省略号解析任务很有用。除了完全注释的数据集外，我们还详细描述了数据集的数据收集过程、统计和分析。报告了一系列基准模型和结果，包括自然语言理解（意图检测和槽填充）、对话状态跟踪和对话上下文到文本生成，以及共指和省略号解析，有助于未来研究的基线比较在这个语料库上。 
  ## 引文
  
```
"@article{quan2020risawoz,
title={Risawoz: A large-scale multi-domain wizard-of-oz dataset with rich semantic annotations for task-oriented dialogue modeling},
author={Quan, Jun and Zhang, Shian and Cao, Qian and Li, Zizhong and Xiong, Deyi},
journal={arXiv preprint arXiv:2010.08738},
year={2020}
}"
```

  
## Download dataset
:modelscope-code[]{type="git"}