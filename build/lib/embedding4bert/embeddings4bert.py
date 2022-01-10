#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  //    ~    /  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
@author: Yekun Chai
@email: chaiyekun@baidu.com
@license: (C)Copyright Baidu NLP
@file: embeddings4bert.py
@time: 2022/01/10 11:18:11
@desc: Extract word embeddings from bert models
'''

import torch
# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM


class Embedding4BERT:
    def __init__(self, model_name, **kwargs):
        if kwargs.get('force') != True:
            assert model_name.startswith('bert') or model_name.startswith('xlnet'), "model_name should be ('bert-base-uncased', 'bert-base-cased', 'xlnet-base-cased', ...). Unless set force=True"
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name.startswith('xlnet'):
            AutoModel = AutoModelForCausalLM 
        elif model_name.startswith('bert'):
            AutoModel = AutoModelForMaskedLM
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

    def extract_word_embeddings(self, text:str, mode="sum", layers=[-1,-2,-3,-4],  **kwargs):
        assert mode in ["sum", "mean"]
        merge_step_queue = list()  # save merge_steps
        marked_text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(marked_text)[:512]
        ret_tokens = []
        if self.model_name.startswith('xlnet'):
            for i, token in enumerate(tokenized_text):
                if token.startswith("▁"):
                    merge_step_queue.append(1)
                    ret_tokens.append(token)
                else:
                    merge_step_queue[-1] += 1
                    ret_tokens[-1] += token[2:]
        elif self.model_name.startswith('bert'):
            for i, token in enumerate(tokenized_text):
                if token.startswith("##"):
                    merge_step_queue[-1] += 1
                    ret_tokens[-1] += token[2:]
                else:
                    merge_step_queue.append(1)
                    ret_tokens.append(token)
        else:
            raise ValueError(f"{self.model_name} not in bert/xlnet! Check your model_name.")

        print(f"{len(merge_step_queue)} tokens: {text}, {len(tokenized_text)} word-tokens: {tokenized_text}")

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # for tup in zip(tokenized_text, indexed_tokens):
        #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

        # segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        # segments_tensors = torch.tensor([segments_ids])

        bert_embeddings = []
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            specified_hidden_states = [outputs.hidden_states[i] for i in layers]
            specified_embeddings = torch.stack(specified_hidden_states, dim=0)
            if mode == "sum":
                token_embeddings = torch.squeeze(torch.sum(specified_embeddings, dim=0))
            elif mode == "mean":
                token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))
            else:
                raise ValueError(f"Invalid mode, not support {mode}!")
            start = 0
            while start < len(indexed_tokens):
                cur_step = merge_step_queue.pop(0)
                if cur_step > 1:
                    embedding = torch.mean(token_embeddings[start: start+cur_step, :], dim=0)
                else:
                    embedding = token_embeddings[start]
                bert_embeddings.append(embedding) # torch >=1.8 => vstack
                # bert_embeddings.append(embedding.unsqueeze(0)) # torch <=1.8 => cat
                start += cur_step
            # bert_embedding = torch.vstack(bert_embeddings) # torch >=1.8 => vstack
            bert_embeddings = torch.stack(bert_embeddings, dim=0)
            # bert_embedding = torch.cat(bert_embeddings, dim=0) # torch <=1.8 => cat 
            bert_embeddings = bert_embeddings.cpu().numpy()
            # print(bert_embeddings.shape)
        return ret_tokens, bert_embeddings

if __name__ == "__main__":
    emb4bert = Embedding4BERT("bert-base-cased")
    tokens, embeddings = emb4bert.extract_word_embeddings('This is a python library for extracting word representations from BERT.')
    print(tokens)
    print(embeddings.shape)