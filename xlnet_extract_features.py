#!/usr/bin/env python

# -*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

@author: Yekun Chai
@license: CYK
@email: chaiyekun@gmail.com
@file: extract_features.py
@time: @Time : 1/4/21 10:10 PM 
@desc： save xlnet embeddings on specific layers.

'''

import torch
from transformers import XLNetTokenizer, XLNetModel
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_name = "dataset1"

name = "train"  # dev / test

data_dir = f"./data/{data_name}"
save_dir = f"./bert_embedding/{data_name}"
os.makedirs(save_dir, exist_ok=True)

file_paths = {
    "train": f"{data_dir}/train.csv",
    "dev": f"{data_dir}/dev.csv",
    "test": f"{data_dir}/test.csv",
}

file_path = file_paths[name]
save_path = os.path.join(save_dir, f"{data_name}_bert.pkl")

df = pd.read_csv(file_path, names=['X', "y"])
X = df["X"]

# XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Load pre-trained model (weights)
model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states=True)
model.to(device)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

all_embeddings = []

for text in X:
    # Tokenize our sentence with the BERT tokenizer.
    merge_step_queue = list()  # save merge_steps
    # TODO: merge
    tokenized_text = tokenizer.tokenize(text)
    for i, token in enumerate(tokenized_text):
        if token.startswith("▁"):
            merge_step_queue.append(1)
        else:
            merge_step_queue[-1] += 1

    print(f"{len(merge_step_queue)} tokens: {text}, {len(tokenized_text)} word-tokens: {tokenized_text}")

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indices.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    # segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    bert_embeddings = []
    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_four_layer_hidden_states = outputs.hidden_states[-4:]
        last_four_layer_embeddings = torch.stack(last_four_layer_hidden_states, dim=0)
        token_embeddings = torch.squeeze(torch.sum(last_four_layer_embeddings, dim=0))
        start = 0
        while start < len(indexed_tokens):
            cur_step = merge_step_queue.pop(0)
            if cur_step > 1:
                embedding = torch.mean(token_embeddings[start: start + cur_step, :], dim=0)
            else:
                embedding = token_embeddings[start]
            bert_embeddings.append(embedding)
            start += cur_step
        bert_embedding = torch.vstack(bert_embeddings)
        all_embeddings.append(bert_embeddings)
        print(bert_embedding.size())

# save to file
import pickle

pickle.dump(all_embeddings, open(save_path, 'wb'))
print(f"Saved to {save_path} !")
