"""发布前检查：独立加载器与原本地实现的输入、参数和预测一致。"""
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import torch

lab = Path(__file__).resolve().parents[1]
release = lab / 'releases/laya-banking77-v1'
sys.path.insert(0, str(release))
from predict import Banking77Classifier, sha256

torch.set_num_threads(4)
torch.manual_seed(42)
texts = ['I lost my card yesterday. How do I freeze it?', 'I was charged twice for the same card payment.', 'I need help with a bank transfer which has not arrived in my account yet.', 'My [MASK] card is missing.']
model = Banking77Classifier(base_dir=lab/'checkpoints/banking77-lora-v1/base', device='cpu')
labels = model.labels
encoded = [model.encode(t) for t in texts]
actual = model.logits(texts, batch_size=4).numpy()
solo = model.logits([texts[0]]).numpy()
assert np.allclose(actual[:1], solo, atol=5e-5, rtol=1e-5)
assert model.predict([]) == []
try:
    model.encode('hello ' * 9000)
except ValueError:
    overflow = True
else:
    raise AssertionError('超长文本未被拒绝')
assert actual.shape == (4, 77)
assert all(len(x['markers']) == 77 for x in encoded)
summary = [{'text': t, 'label': labels[int(row.argmax())]} for t,row in zip(texts,actual)]
del model
for obj in [None]: gc.collect()
sys.path.insert(0, str(lab/'scripts'))
from local_laya import LocalLaya
from banking77_data import BankingEncoder
reference = LocalLaya('tuned', 'cpu')
encoder = BankingEncoder(reference.tokenizer, labels, 8192)
assert encoded == [encoder.encode(encoder.tokenize_state(t)) for t in texts]
expected, info = reference.logits(texts, {label:'' for label in labels})
np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
report = {'device':'cpu', 'dtype':'float32', 'synthetic_smoke_examples':len(texts), 'all_77_candidates_preserved':True,
 'training_encoder_input_parity':True, 'max_absolute_logit_difference_from_original_loader':float(np.max(np.abs(actual-expected))),
 'batch_vs_single_max_absolute_logit_difference':float(np.max(np.abs(actual[:1]-solo))), 'overflow_rejected':overflow,
 'checkpoint_sha256':sha256(release/'trainable.safetensors'), 'predictions':summary,
 'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','safetensors','huggingface-hub','numpy']},
 'note':'Publication packaging smoke check, not a new evaluation of test accuracy.'}
(release/'packaging_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
