#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:44:12 2023

@author: guoxing.lan
"""

from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
model_name_or_path="xx/models/xlm-roberta-large/"

import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer=XLMRobertaTokenizer(vocab_file=model_name_or_path)
model=AutoModelForSequenceClassification(model_name_or_path)


model.half()


model=model.to(device)
text_list=["I love shopping"]

inputs_t=tokenizer(
    text_list,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
    )
outputs=model(**inputs_t)

logits=outputs.logits

dummy_input=inputs_t["input_ids"]

# In[]
onnx_fp16_path="xx/models/xlm-roberta-large-onnx-fp16"
torch.onnx.export(model,dummy_input,onnx_fp16_path,opset_version=11)
