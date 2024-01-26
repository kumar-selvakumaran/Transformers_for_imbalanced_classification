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
    """
    The `init_tokenizer` function initializes a tokenizer based on the specified model name.
    
    :param name: The `name` parameter is a string that represents the name of the tokenizer. It is used
    to select the appropriate tokenizer from the `tokenizer_dict` dictionary
    :return: a tokenizer object based on the input name.
    """
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
    """
    The function `batch_encode` takes a list of texts, a tokenizer, and optional parameters for maximum
    sequence length and batch size, and returns the input IDs and attention masks for the encoded texts.
    
    :param texts: The `texts` parameter is a list of strings containing the texts that you want to
    encode
    :param tokenizer: The tokenizer is an object that is used to convert text into numerical tokens. It
    is typically used in natural language processing tasks such as text classification or language
    generation. The tokenizer takes care of tasks such as splitting text into words, converting words to
    lowercase, and mapping words to numerical indices
    :param max_seq_len: The `max_seq_len` parameter specifies the maximum length of the input sequences.
    If a sequence is longer than this value, it will be truncated to fit. If it is shorter, it will be
    padded with special tokens to match the specified length, defaults to 512 (optional)
    :param batch_size: The batch_size parameter determines the number of texts that will be processed in
    each batch. It is used to split the texts into smaller batches for efficient processing, defaults to
    10000 (optional)
    :return: The function `batch_encode` returns two lists: `input_ids` and `attention_masks`.
    """
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
