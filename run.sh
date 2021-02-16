#!/bin/sh

if [ "$1" = "bert" ]
then
  echo 'Extract bert word embeddings ...'
  python extract_bert_embedding.py
elif [ "$1" = "xlnet" ]
then
  echo 'Extract XLNet word embeddings ...'
  python extract_xlnet_embedding.py
fi