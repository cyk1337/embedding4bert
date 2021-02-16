# BERT-Word-Embeddings-Pytorch
Extract word embeddings of pretrained BERT models.
- Sum the representations of the last four layers. 
- Take the mean of the representation of subword pieces as the word representations.

1. Extract BERT word embeddings.
```bash
sh run.py bert
```

2. Extract XLNet word embeddings.
```bash
sh run.py xlnet
```

## References
1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)