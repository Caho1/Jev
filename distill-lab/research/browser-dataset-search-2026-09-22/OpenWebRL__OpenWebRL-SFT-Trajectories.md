---
pretty_name: OpenWebRL SFT Trajectories
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
tags:
- openwebrl
- web-agents
- visual-web-agents
- supervised-fine-tuning
- browser-automation
- llamafactory
- playwright
- orchard
---

# OpenWebRL SFT Trajectories

## Dataset Summary

OpenWebRL SFT Trajectories contains successful browser-agent trajectory data collected by [Qwen3-VL-235B-A22B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking). These trajectories are used to supervised fine-tune the initial OpenWebRL visual web agent. The SFT checkpoint is then used as the initialization checkpoint for OpenWebRL's subsequent online multi-turn reinforcement learning stage.

OpenWebRL trains visual web agents on live websites. The full training recipe first performs supervised fine-tuning with LLaMAFactory on successful browser interaction data, then continues with online RL using browser rollouts, format rewards, and VLM-as-a-judge task-completion rewards.

This dataset is stored as turn-level prompt/response examples. Each row contains a browser-agent prompt, the target assistant response for that turn, rollout metadata, screenshots, task information, and reward annotations from the successful rollout.

Project repository: [OpenWebRL](https://github.com/OpenWebRL/OpenWebRL)

## Dataset Details

- **Dataset repository:** `OpenWebRL/OpenWebRL-SFT-Trajectories`
- **Data file:** `OpenWebRL_SFT_trajectories.jsonl`
- **Format:** JSON Lines
- **Number of examples:** 3,085 turn-level SFT examples
- **Number of unique browser tasks:** 412
- **Language:** English
- **Source benchmark:** `pae-webvoyager`
- **Training framework:** LLaMAFactory supervised fine-tuning
- **Role in OpenWebRL:** trains the SFT checkpoint used to initialize OpenWebRL online RL
- **Associated framework:** OpenWebRL, a browser-agent RL framework built on Megatron/SGLang-based `slime`, Playwright browser environments, and optional Orchard sandbox environments for scalable isolated browser rollouts.

All examples in this release are successful rollout turns:

- `status` is `completed` for all examples.
- Top-level `reward` is `1.0` for all examples.
- `metadata.terminate_reason` is `task_completed` for all examples.
- `metadata.is_last_turn` is `true` for 412 examples, corresponding to the final turn of each completed task trajectory.

## Dataset Structure

Each line in `OpenWebRL_SFT_trajectories.jsonl` is a JSON object with the following top-level fields:

| Field | Type | Description |
| --- | --- | --- |
| `rollout_idx` | integer | Rollout index. |
| `group_index` | integer | Group/sample index within the rollout set. Values range from 0 to 29 in this release. |
| `index` | null | Reserved index field. This release uses `null` for all examples. |
| `prompt` | string | Serialized browser-agent prompt used as the SFT input. |
| `response` | string | Target assistant response for supervised fine-tuning. |
| `response_length` | integer | Length of the target response according to the preprocessing pipeline. |
| `reward` | float | Top-level success reward. This release contains only `1.0` examples. |
| `status` | string | Rollout status. This release contains only `completed` examples. |
| `metadata` | object | Task, trajectory, screenshot, message, and reward metadata. |

The `metadata` object contains:

| Field | Type | Description |
| --- | --- | --- |
| `intent` | string | Natural-language task instruction. |
| `start_url` | string | Starting website URL. |
| `sites` | list | Additional site metadata. |
| `task_id` | string | Browser task identifier. |
| `require_login` | boolean | Whether the task requires login. This release uses `false`. |
| `storage_state` | null | Browser storage state. This release uses `null`. |
| `require_reset` | boolean | Whether browser reset is required. |
| `intent_template_id` | integer | Task template identifier. |
| `benchmark_name` | string | Source benchmark name. |
| `domain` | string | High-level task domain. |
| `subdomain` | string | More specific task category. |
| `definite_answer` | string | Direct-answer field. |
| `difficulty` | integer | Difficulty score assigned to the task. |
| `evaluator_reference` | list | Structured evaluator references for task completion. |
| `turn_index` | integer | Turn index within the trajectory. |
| `is_last_turn` | boolean | Whether this row is the final turn of the trajectory. |
| `images` | list | Screenshot image data associated with the turn. This release has one image per row. |
| `messages` | list | Conversation/message history for the browser-agent turn. |
| `terminate_reason` | string | Final trajectory termination reason. |
| `total_steps` | integer | Total number of steps in the completed trajectory. |
| `reward` | object | Detailed reward components, including format reward, judge reward, combined reward, and judge text. |

Simplified example:

```json
{
  "rollout_idx": 0,
  "group_index": 0,
  "index": null,
  "prompt": "<serialized browser-agent prompt>",
  "response": "<assistant reasoning and browser tool call>",
  "response_length": 250,
  "reward": 1.0,
  "status": "completed",
  "metadata": {
    "intent": "Find the latest news on the latest developments in science, technology, and engineering, including new patents, research breakthroughs, and industry news.",
    "start_url": "https://bbc.com/news",
    "task_id": "252667",
    "benchmark_name": "pae-webvoyager",
    "domain": "Science & Research",
    "subdomain": "Technology & Science",
    "difficulty": 12,
    "turn_index": 0,
    "is_last_turn": false,
    "terminate_reason": "task_completed",
    "total_steps": 7,
    "reward": {
      "format": 1.0,
      "judge": 1.0,
      "combined": 1.0
    }
  }
}
```

## Using This Dataset with LLaMAFactory

This dataset was prepared for supervised fine-tuning with LLaMAFactory. The main supervised fields are:

- `prompt`: SFT input, serialized with the browser-agent system prompt, task context, observation/history, and image placeholders.
- `response`: SFT target output, including the assistant's reasoning/action response for the browser turn.

The remaining fields provide metadata for filtering, auditing, debugging, multimodal input construction, and downstream analysis. Training configurations may choose to consume only `prompt` and `response`, or additionally use image/message metadata depending on the local LLaMAFactory multimodal setup.

The resulting SFT checkpoint is used as the starting checkpoint for OpenWebRL online RL, where the model continues training through live browser interaction and judge-based rewards.

The OpenWebRL repository provides a reproducible [SFT helper](https://github.com/OpenWebRL/OpenWebRL/tree/main/sft) under `sft/`:

```bash
LLAMAFACTORY_ROOT=/root/LlamaFactory \
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Thinking \
OUTPUT_DIR=/path/to/openwebrl_sft_ckpt \
bash sft/run_sft_with_llamafactory.sh
```

Before running the wrapper, users should set up the uv environment under the
LLaMAFactory root. A quick environment check is:

```bash
cd /root/LlamaFactory
uv run python -c "import llamafactory; print('ok')"
```

By default, this wrapper downloads this Hugging Face dataset file,
`OpenWebRL_SFT_trajectories.jsonl`, converts it into LLaMAFactory's
ShareGPT-style multimodal format, extracts screenshots to PNG files, generates
`dataset_info.json` and a training YAML, and launches SFT through `uv run` from
`LLAMAFACTORY_ROOT`. Users can run a data-preparation smoke test without
training via:

```bash
RUN_TRAIN=0 MAX_ROWS=10 bash sft/run_sft_with_llamafactory.sh
```

The generated LLaMAFactory dataset uses:

```yaml
formatting: sharegpt
columns:
  messages: messages
  images: images
tags:
  role_tag: role
  content_tag: content
  user_tag: user
  assistant_tag: assistant
  system_tag: system
```

The recommended SFT configuration uses `template: qwen3_vl` and
`mask_history: true`, so the model is supervised on the current assistant turn
while earlier assistant turns remain context.

## Dataset Statistics

### Task and Turn Counts

| Statistic | Value |
| --- | ---: |
| SFT examples | 3,085 |
| Unique browser tasks | 412 |
| Final-turn examples | 412 |
| Non-final-turn examples | 2,673 |
| Screenshots per example | 1 |

### Source Benchmark

| Benchmark | Examples |
| --- | ---: |
| `pae-webvoyager` | 3,085 |

### Domains

| Domain | Examples |
| --- | ---: |
| Lifestyle & Leisure | 903 |
| Science & Research | 847 |
| Misc. | 410 |
| Entertainment | 372 |
| Career & Education | 308 |
| Travel & Transportation | 245 |

### Common Starting Websites

| Website | Examples |
| --- | ---: |
| `https://amazon.com` | 249 |
| `https://allrecipes.com` | 243 |
| `https://apple.com` | 233 |
| `https://coursera.org` | 188 |
| `https://bbc.com/news` | 172 |
| `https://huggingface.co` | 172 |
| `https://github.com` | 159 |
| `https://arxiv.org` | 138 |
| `https://espn.com` | 131 |
| `https://dictionary.cambridge.org` | 127 |

### Lengths

| Field | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| `response_length` | 70 | 307 | 327.43 | 1,324 |
| `prompt` characters | 10,542 | 16,946 | 20,494.09 | 71,136 |

## Intended Uses

This dataset is intended for:

- Supervised fine-tuning browser agents before online RL.
- Reproducing or analyzing the OpenWebRL SFT initialization stage.
- Training visual web agents with LLaMAFactory-style prompt/response supervision.
- Studying successful browser-agent reasoning, tool-call formatting, and action selection.
- Bootstrapping online RL runs that continue from a behaviorally competent browser-agent checkpoint.

## Out-of-Scope Uses

This dataset should not be treated as the online RL task dataset. It contains supervised successful trajectory turns, while OpenWebRL online RL uses task prompts, live browser rollouts, and judge-based rewards.

This dataset should not be used as a source of factual truth about current website contents. It contains snapshots and responses from prior browser interactions, and live websites may change over time.

The dataset should not be used to automate abusive traffic, bypass website access controls, scrape private information, or violate the terms of service of target websites.

## Data Collection and Processing

The examples were collected from successful OpenWebRL browser rollouts on `pae-webvoyager` tasks. Browser episodes were run in the OpenWebRL browser environment, which supports Playwright-based local browser execution and [Orchard](https://github.com/microsoft/Orchard) sandbox execution for scalable isolated rollouts.

Rows were converted into turn-level supervised examples for LLaMAFactory SFT. Each row pairs the serialized browser-agent prompt with the target assistant response for that turn. Reward metadata records successful task completion according to the OpenWebRL reward pipeline.

## Relationship to OpenWebRL Online RL

OpenWebRL uses this SFT data before the online RL stage:

1. Train a supervised browser-agent checkpoint with LLaMAFactory on successful trajectory turns.
2. Use the SFT checkpoint as the initialization checkpoint for OpenWebRL online RL.
3. Continue training with live browser rollouts, format rewards, VLM-as-a-judge rewards, and MM-GRPO optimization.

The online RL stage is implemented in the OpenWebRL repository with browser training launchers such as:

```text
scripts/run_browser_Qwen3VL_4B_Instruct.sh
scripts/run_browser_Qwen3VL_8B_Instruct.sh
```

## Limitations

- The dataset contains only successful examples, so it is not balanced across successful and failed behavior.
- Rows are turn-level SFT examples rather than complete standalone trajectory files.
- The dataset includes screenshots and browser state/history metadata, which can be large and may reflect website content at collection time.
- Website content may vary across regions, sessions, dates, devices, cookies, and personalization settings.
- SFT behavior learned from this data is intended to be improved further by OpenWebRL online RL.

## Privacy and Safety

The dataset contains browser-task prompts, model responses, screenshots, and metadata from web interactions. It is not intended to contain private user data. Users should still review screenshots, message histories, and generated responses for privacy, copyright, security, and safety concerns before redistributing derived datasets or models.

When using this dataset to train browser agents, downstream users should ensure their agents respect website terms of service, robots/access policies, rate limits, and user privacy.

## Acknowledgements

OpenWebRL builds on `slime`, SGLang, Megatron-LM, Megatron-Bridge, Playwright, Qwen VLMs, LLaMAFactory, and the broader open-source VLM/web-agent ecosystem. The OpenWebRL project also acknowledges WebGym for providing the initial browser-task data source.

## Citation

If you use this dataset, please cite OpenWebRL:

```bibtex
@article{yang2026openwebrl,
  title   = {OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents},
  author  = {Rui Yang and Qianhui Wu and Yuxi Chen and Hao Bai and Wenlin Yao and Hao Cheng and Baolin Peng and Huan Zhang and Tong Zhang and Jianfeng Gao},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Maintenance

For questions, corrections, or updates, please contact the OpenWebRL maintainers through the Hugging Face dataset repository or the OpenWebRL project repository.
