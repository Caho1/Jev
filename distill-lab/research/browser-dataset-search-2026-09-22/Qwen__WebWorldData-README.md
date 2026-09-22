---
license: apache-2.0
task_categories:
- text-generation
language:
- en
- zh
tags:
- WebWorld
- world-model
- web-agent
- browser-simulation
- a11y
- html
- xml
- markdown
- trajectories
- agent-training
- synthetic-data
pretty_name: WebWorldData
size_categories:
- 1M<n<10M
---

# WebWorldData 🌐

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/LICENSE-2.0) 
[![GitHub](https://img.shields.io/badge/GitHub-WebWorld-4b32c3?logo=github)](https://github.com/QwenLM/WebWorld) 
[![Dataset](https://img.shields.io/badge/HF%20Dataset-WebWorldData-yellow?logo=huggingface)](https://huggingface.co/datasets/Qwen/WebWorldData) 
[![MS Dataset](https://img.shields.io/badge/ModelScope-Dataset-7B42BC)](https://modelscope.cn/datasets/Qwen/WebWorldData) 
[![8B](https://img.shields.io/badge/Model-8B-green?logo=huggingface)](https://huggingface.co/Qwen/WebWorld-8B) 
[![MS 8B](https://img.shields.io/badge/ModelScope-8B-7B42BC)](https://modelscope.cn/models/Qwen/WebWorld-8B) 
[![14B](https://img.shields.io/badge/Model-14B-green?logo=huggingface)](https://huggingface.co/Qwen/WebWorld-14B) 
[![MS 14B](https://img.shields.io/badge/ModelScope-14B-7B42BC)](https://modelscope.cn/models/Qwen/WebWorld-14B) 
[![32B](https://img.shields.io/badge/Model-32B-green?logo=huggingface)](https://huggingface.co/Qwen/WebWorld-32B) 
[![MS 32B](https://img.shields.io/badge/ModelScope-32B-7B42BC)](https://modelscope.cn/models/Qwen/WebWorld-32B)

## Overview

**WebWorldData** is a large-scale dataset of **1.06M web interaction trajectories** collected from the open web, designed for training browser world models. It is the training data behind the [WebWorld](https://github.com/QwenLM/WebWorld) model series.

Each trajectory consists of sequences of `(state, action, next_state)` transitions, where states are represented as A11y Trees extracted from real websites using Playwright.

## Dataset Statistics

| | |
|---|---|
| **Total Trajectories** | 1,059,348 |
| **Total Size** | ~50.9 GB |
| **Languages** | English, Chinese |
| **Max Context Length** | 30K tokens |
| **Max Trajectory Turns** | 30+ steps |
| **State Format** | A11y Tree (primary), HTML, XML, Markdown, Natural Language |
| **Source Websites** | 680K+ URLs from FineWeb, CCI 3.0, and curated lists |

## Data Sources

The dataset is collected through a **scalable hierarchical pipeline**:

| Source | Strategy | Scale | Description |
|---|---|---|---|
| **Level 1: Randomized Crawling** | Rule-based crawlers | 293K | Randomized exploration on websites from pre-training corpora (FineWeb, CCI 3.0), aligned with the base model's linguistic priors |
| **Level 2: Autonomous Exploration** | LLM-driven agents | 38K | Agents autonomously explore websites by generating their own objectives, producing long-horizon trajectories up to 30 steps |
| **Level 3: Task-Oriented Execution** | Synthetic tasks | 94K | Agents execute synthesized web tasks through seed extraction, diversification, and paraphrasing |
| **Open Source** | AgentTrek, etc. | 38K | Reformatted open-source agent trajectories |
| **Multi-Format** | Format conversion | 48K | Trajectories converted to HTML, XML, Markdown, Playwright formats |
| **Interaction** | General QA + chat | 548K | General instruction-following and QA data to preserve conversational abilities |

## Data Format

Each sample is a multi-turn conversation in JSONL format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a web world model. I will provide you with an initial page state and a sequence of actions. For each action, predict the resulting page state.\nStrictly maintain the original format. Output only the full page state without explanations, code, or truncation."
    },
    {
      "role": "user",
      "content": "Initial Page State:\nRootWebArea 'Example Site'\n\t[1] banner ...\n\nFirst Action: 'click([32])'\n\nNext Page State:"
    },
    {
      "role": "assistant",
      "content": "RootWebArea 'Example Site - News'\n\t[1] banner ...\n\t[50] main ..."
    },
    {
      "role": "user",
      "content": "Continue the trajectory. Given the previous state, predict the next page state after this action.\n\nAction: 'fill([19], \"weather today\")'\n\nNext Page State:"
    },
    {
      "role": "assistant",
      "content": "RootWebArea 'Example Site - News'\n\t[1] banner ...\n\t[19] textbox ..., value='weather today' ..."
    }
  ]
}
```

## Domain Distribution

The dataset covers diverse web domains:

| Domain | Share |
|---|---|
| Technology | 15.2% |
| E-Commerce / Shopping | 13.8% |
| News & Media | 12.1% |
| Education | 10.5% |
| Entertainment | 9.3% |
| Lifestyle | 8.7% |
| Business & Finance | 7.9% |
| Government & Public Services | 6.4% |
| Health | 5.8% |
| Other | 10.3% |

## Action Space

Trajectories use a unified action space as Python-style function calls:

| Category | Actions |
|---|---|
| **Element** | `click`, `fill`, `select_option`, `hover` |
| **Mouse** | `mouse_move`, `mouse_click`, `mouse_down`, `mouse_up` |
| **Keyboard** | `keyboard_press`, `keyboard_type` |
| **Browser** | `scroll`, `goto`, `go_back`, `go_forward`, `tab_new`, `tab_close`, `tab_focus` |
| **Meta** | `send_msg_to_user`, `noop`, `infeasible` |

Action distribution: Element interactions (83.4%), Browser & navigation (11.9%), Meta & control (2.4%), Keyboard (1.2%), Coordinate & mouse (1.1%).

## Filtering & Safety

The dataset undergoes rigorous dual-stage filtering:

1. **Rule-based filtering**: Website reachability checks, banned keyword filtering (pornography, gambling, violence), trajectory pruning for no-op transitions
2. **LLM-based URL filtering**: Each URL scored across accessibility, content suitability, interactivity, and engineering quality
3. **Trajectory-level filtering**: Max 30K tokens, max 30 turns, keyword safety checks

All data is collected from publicly accessible webpages in compliance with `robots.txt` protocols.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Qwen/WebWorldData")
```

## Intended Use

- Training browser world models for web simulation
- Generating synthetic trajectories for web agent fine-tuning
- Research on world modeling, environment simulation, and agent learning

## Limitations

- Data is collected from publicly accessible webpages; residual PII may exist despite filtering
- Web content is inherently non-deterministic (ads, A/B tests, dynamic widgets) — some trajectories may not be perfectly reproducible
- Domain distribution reflects the composition of FineWeb and CCI 3.0 pre-training corpora

## Associated Models

| Model | Link |
|---|---|
| WebWorld-8B | [🤗 HuggingFace](https://huggingface.co/Qwen/WebWorld-8B) |
| WebWorld-14B | [🤗 HuggingFace](https://huggingface.co/Qwen/WebWorld-14B) |
| WebWorld-32B | [🤗 HuggingFace](https://huggingface.co/Qwen/WebWorld-32B) |

## Citation

```bibtex
@misc{xiao2026webworldlargescaleworldmodel,
      title={WebWorld: A Large-Scale World Model for Web Agent Training}, 
      author={Zikai Xiao and Jianhong Tu and Chuhang Zou and Yuxin Zuo and Zhi Li and Peng Wang and Bowen Yu and Fei Huang and Junyang Lin and Zuozhu Liu},
      year={2026},
      eprint={2602.14721},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.14721}, 
}
```

## License

This dataset is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).