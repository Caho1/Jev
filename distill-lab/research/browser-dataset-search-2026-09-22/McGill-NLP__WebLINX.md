---
language:
- en
size_categories:
- 10K<n<100K
configs:
- config_name: chat
  default: true
  data_files:
  - split: train
    path: data/chat/train.json.gz
  - split: validation
    path: data/chat/valid.json.gz
  - split: test
    path: data/chat/test_iid.json.gz
  - split: test_iid
    path: data/chat/test_iid.json.gz
  - split: test_geo
    path: data/chat/test_geo.json.gz
  - split: test_vis
    path: data/chat/test_vis.json.gz
  - split: test_cat
    path: data/chat/test_cat.json.gz
  - split: test_web
    path: data/chat/test_web.json.gz
- config_name: reranking
  data_files:
  - split: validation
    path: data/reranking/valid.json.gz
  - split: test
    path: data/reranking/test_iid.json.gz
  - split: test_iid
    path: data/reranking/test_iid.json.gz
  - split: test_geo
    path: data/reranking/test_geo.json.gz
  - split: test_vis
    path: data/reranking/test_vis.json.gz
  - split: test_web
    path: data/reranking/test_web.json.gz
  - split: test_cat
    path: data/reranking/test_cat.json.gz
tags:
- image-to-text
- vision
- convAI
task_categories:
- image-to-text
- text-generation
- text2text-generation
- sentence-similarity
pretty_name: weblinx
license: cc-by-nc-sa-4.0
---

<div align="center">
  <h1 style="margin-bottom: 0.5em;">WebLINX: Real-World Website Navigation with Multi-Turn Dialogue</h1>
  <em>Xing Han Lù*, Zdeněk Kasner*, Siva Reddy</em>
</div>

<div style="margin-bottom: 2em"></div>


| [**💾Code**](https://github.com/McGill-NLP/WebLINX) | [**📄Paper**](https://arxiv.org/abs/2402.05930) | [**🌐Website**](https://mcgill-nlp.github.io/weblinx) | [**📓Colab**](https://colab.research.google.com/github/McGill-NLP/weblinx/blob/main/examples/WebLINX_Colab_Notebook.ipynb) |
| :--: | :--: | :--: | :--: |
| [**🤖Models**](https://huggingface.co/collections/McGill-NLP/weblinx-models-65c57d4afeeb282d1dcf8434) | [**💻Explorer**](https://huggingface.co/spaces/McGill-NLP/weblinx-explorer) | [**🐦Tweets**](https://twitter.com/sivareddyg/status/1755799365031965140) | [**🏆Leaderboard**](https://paperswithcode.com/sota/conversational-web-navigation-on-weblinx) |


<video width="100%" controls autoplay muted loop>
    <source src="https://huggingface.co/datasets/McGill-NLP/WebLINX/resolve/main/WeblinxWebsiteDemo.mp4?download=false" type="video/mp4">
    Your browser does not support the video tag.
</video>


> [!IMPORTANT]
> WebLINX is now available as a benchmark through [BrowserGym](https://github.com/ServiceNow/BrowserGym), allowing you to access demonstration steps in the same way you would access a web agent environment like [WebArena](https://webarena.dev/) or [MiniWoB](https://miniwob.farama.org/index.html). This also allows you to run agents from the [Agentlab](https://github.com/ServiceNow/AgentLab) library, including agents that achieve SOTA performance through Claude-3.5-Sonnet. To enable this integration, we are releasing the `weblinx-browsergym` extension for BrowserGym on PyPi, as well as a [new dataset, WebLINX 1.1, derived from WebLINX on Huggingface](https://huggingface.co/datasets/McGill-NLP/weblinx-browsergym). In WebLINX 1.1, a small number of demonstrations were removed after processing, but no new demonstration was added. There are substantial changes to the steps being evaluated, with the inclusion of tab actions. Please report your results as "WebLINX-1.1", "WebLINX-BrowserGym" or "WebLINX-BG" in your work, to differentiate from the [initial release of weblinx (1.0)](https://huggingface.co/datasets/McGill-NLP/WebLINX/tree/v1.0).


## Quickstart

To get started, simply install `datasets` with `pip install datasets` and load the chat data splits:

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Load the validation split
valid = load_dataset("McGill-NLP/weblinx", split="validation")

# Download the input templates and use the LLaMA one
snapshot_download(
    "McGill-NLP/WebLINX", repo_type="dataset", allow_patterns="templates/*", local_dir="."
)
with open('templates/llama.txt') as f:
    template = f.read()

# To get the input text, simply pass a turn from the valid split to the template
turn = valid[0]
turn_text = template.format(**turn)
```

You can now use `turn_text` as an input to LLaMA-style models. For example, you can use Sheared-LLaMA:

```python
from transformers import pipeline

action_model = pipeline(
    model="McGill-NLP/Sheared-LLaMA-2.7B-weblinx", device=0, torch_dtype='auto'
)
out = action_model(turn_text, return_full_text=False, max_new_tokens=64, truncation=True)
pred = out[0]['generated_text']

print("Ref:", turn["action"])
print("Pred:", pred)
```

## Raw Data

To use the raw data, you will need to use the `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

# If you want to download the complete dataset (may take a while!)
snapshot_download(repo_id="McGill-NLP/WebLINX-full", repo_type="dataset", local_dir="./wl_data")

# You can download specific demos, for example
demo_names = ['saabwsg', 'ygprzve', 'iqaazif']  # 3 random demo from valid
patterns = [f"demonstrations/{name}/*" for name in demo_names]
snapshot_download(
    repo_id="McGill-NLP/WebLINX-full", repo_type="dataset", local_dir="./wl_data", allow_patterns=patterns
)
```

For more information on how to use this data using our [official library](https://github.com/McGill-NLP/WebLINX), please refer to the [WebLINX documentation](https://mcgill-nlp.github.io/weblinx/docs).

## Reranking Data

You can also access the data processed for reranking tasks. To do that:

```python
from datasets import load_dataset

path = 'McGill-NLP/WebLINX'

# validation split:
valid = load_dataset(path=path, name='reranking', split='validation')
# test-iid split
test_iid = load_dataset(path, 'reranking', split='test_iid')
# other options: test_cat, test_geo, test_vis, test_web

print("Query:")
print(valid[0]['query'])

print("\nPositive:")
print(valid[0]['positives'][0])

print("\nNegative #1:")
print(valid[0]['negatives'][0])

print("\nNegative #2:")
print(valid[0]['negatives'][1])
```

## License and Terms of Use

License: The Dataset is made available under the terms of the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

By downloading this Dataset, you agree to comply with the following terms of use:
- Restrictions: You agree not to use the Dataset in any way that is unlawful or would infringe upon the rights of others.
- Acknowledgment: By using the Dataset, you acknowledge that the Dataset may contain data derived from third-party sources, and you agree to abide by any additional terms and conditions that may apply to such third-party data.
- Fair Use Declaration: The Dataset may be used for research if it constitutes "fair use" under copyright laws within your jurisdiction. You are responsible for ensuring your use complies with applicable laws.

Derivatives must also include the terms of use above.

## Citation

If you use our dataset, please cite our work as follows:

```bibtex
@misc{lu-2024-weblinx,
      title={WebLINX: Real-World Website Navigation with Multi-Turn Dialogue}, 
      author={Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      year={2024},
      eprint={2402.05930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```