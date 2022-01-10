# Embedding4BERT

![Stable version](https://img.shields.io/pypi/v/embedding4bert)
![Python3](https://img.shields.io/pypi/pyversions/embedding4bert)![wheel:embedding4bert](https://img.shields.io/pypi/wheel/embedding4bert)
![MIT License](https://img.shields.io/pypi/l/embedding4bert)

<!--![Download](https://img.shields.io/pypi/dm/embedding4bert)-->

Table of Contents
=================

- [User Guide](https://github.com/cyk1337/embedding4bert/#user-guide)
    - [Installation](https://github.com/cyk1337/embedding4bert/#installation)
    - [Usage](https://github.com/cyk1337/embedding4bert/#usage)
- [Citation](https://github.com/cyk1337/embedding4bert/#citation)
- [References](https://github.com/cyk1337/embedding4bert/#references)

This is a python library for extracting word embeddings from pre-trained language models. 

## User Guide
### Installation
```bash
pip install --upgrade embedding4bert
```

### Usage

Extract word embeddings of pretrained language models, such as BERT or XLNet.
The `extract_word_embeddings` function of `Embedding4BERT` class has following arguments:
- `mode`: `str`. `"sum"` (default) or`"mean"`. Take the sum or average representations of the specficied layers. 
- `layers`: `List[int]`. default: `[-1,-2,-3,-4]`, indicating take the last four layers. Take the word representation of specifed layers from the given list.

1. Extract BERT word embeddings.
```python
from embedding4bert import Embedding4BERT
emb4bert = Embedding4BERT("bert-base-cased") # bert-base-uncased
tokens, embeddings = emb4bert.extract_word_embeddings('This is a python library for extracting word representations from BERT.', mode="sum", layers=[-1,-2,-3,-4]) # Take the sum of last four layers
print(tokens)
print(embeddings.shape)
```

Expected output:
```bash
14 tokens: [CLS] This is a python library for extracting word representations from BERT. [SEP], 19 word-tokens: ['[CLS]', 'This', 'is', 'a', 'p', '##yt', '##hon', 'library', 'for', 'extract', '##ing', 'word', 'representations', 'from', 'B', '##ER', '##T', '.', '[SEP]']
['[CLS]', 'This', 'is', 'a', 'python', 'library', 'for', 'extracting', 'word', 'representations', 'from', 'BERT', '.', '[SEP]']
(14, 768)
```

2. Extract XLNet word embeddings.
```python
from embedding4bert import Embedding4BERT
emb4bert = Embedding4BERT("xlnet-base-cased")
tokens, embeddings = emb4bert.extract_word_embeddings('This is a python library for extracting word representations from XLNet.', mode="mean", layers=[-1,-2,-3,]) # Take the mean embeddings of last three layers
print(tokens)
print(embeddings.shape)
```

Expected output:
```bash
11 tokens: This is a python library for extracting word representations from BERT., 16 word-tokens: ['▁This', '▁is', '▁a', '▁', 'py', 'thon', '▁library', '▁for', '▁extract', 'ing', '▁word', '▁representations', '▁from', '▁B', 'ERT', '.']
['▁This', '▁is', '▁a', '▁python', '▁library', '▁for', '▁extracting', '▁word', '▁representations', '▁from', '▁BERT.']
(11, 768)
```


## Citation
For attribution in academic contexts, please cite this work as:
```
@misc{chai2020-embedding4bert,
  author = {Chai, Yekun},
  title = {embedding4bert: A python library for extracting word embeddings from pre-trained language models},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cyk1337/embedding4bert}}
}
```


## References
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
