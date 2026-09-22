---
configs:
- config_name: default
  data_files:
  - path: train.csv
    split: train
license: Apache License 2.0
tags:
- product description generation
text:
  text-generation:
    language:
    - zh
    type:
    - data-to-text
---


## 概述：

客服对话数据集

## 数据集描述：

本数据集包括简单客服对话测试集。其中，每一条数据有三个属性，分别是输入句子、输出句子和历史。

## 范例：

{"prompt":"你好","response":"您好，有什么可以帮您？",history:[]}
## Clone with HTTP
* http://www.modelscope.cn/datasets/newhwa/dialog_test_tmall.git