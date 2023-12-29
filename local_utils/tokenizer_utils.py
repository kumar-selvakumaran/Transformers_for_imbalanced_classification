import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import pandas as pd
import numpy as np


#IMPORTING TOKENIZERS OF PROPOSED APPROACH'S MODEL (DeBERTa) AND ALL OTHER COMPARED MODELS
from transformers import (
    ElectraTokenizerFast,
    BertTokenizerFast,
    DebertaTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
    AdamW)

from tqdm import tqdm
import torch
import json
import pickle 

def init_tokenizer(name):
    tokenizer_dict = {
        "Electra" : (ElectraTokenizerFast,'google/electra-base-discriminator'),
        "Bert" : (BertTokenizerFast,'bert-base-uncased'),
        "Deberta" : (DebertaTokenizerFast,'microsoft/deberta-base'), 
        "DistilBert" : (DistilBertTokenizerFast,'distilbert-base-uncased'),
        "Roberta" : (RobertaTokenizerFast, 'roberta-base')
    }
    tokennizer_base = tokenizer_dict[name][0]
    tokenizer =  tokennizer_base.from_pretrained(tokenizer_dict[name][1]) 
    return tokenizer

def batch_encode(texts, tokenizer, max_seq_len = 512, batch_size = 10000):
    input_ids = []
    attention_masks = []
    for i in tqdm(range(0, len(texts), batch_size)):
        if (len(texts) - i) < batch_size:
          batch_text = texts[i:]
        else:
          batch_text = texts[i:i+batch_size]  

        encoded_sent = tokenizer.batch_encode_plus(
            batch_text,
            max_length=max_seq_len,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='max_length',
            truncation=True
        )

        input_ids += encoded_sent["input_ids"]
        attention_masks += encoded_sent["attention_mask"]

    return input_ids, attention_masks
