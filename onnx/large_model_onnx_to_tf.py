#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:46:20 2023

@author: guoxing.lan
"""

import onnx
from onnx_tf.backend import prepare

onnx_model=onnx.load("xx/models/xlm-roberta-large-onnx-fp16/model.onnx")
tf_rep=prepare(onnx_model, auto_cast=True)
tf_rep.export_graph("xx/models/xlm-roberta-large-tf-fp16")
