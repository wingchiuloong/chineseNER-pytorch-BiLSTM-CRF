#!/usr/bin/python
# coding=utf-8

import pickle

with open('../data/renmindata.pkl', 'rb') as inp:
	word2id = pickle.load(inp) 
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs 
# from BiLSTM_CRF import BiLSTM_CRF
from resultCal import calculate

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

model_data_filepath = './model/model2.pkl'
model = torch.load(model_data_filepath, map_location=torch.device('cpu'))

sentence_input= raw_input('Enter your sentence:\n')

sentence_input = unicode(sentence_input, 'utf-8')
sentence_id = []
for word in sentence_input:
	if word not in [u'，', u'。', u'！', u'？', u'-', u'“', u'”', u',', u'.', u'?', u'!', u' ', u'、']:
		if word in word2id:
			sentence_id.append(word2id[word])
		else:
			sentence_id.append(0)

entityres=[]
sentence = torch.tensor(sentence_id, dtype=torch.long)
score,predict = model(sentence)
entityres = calculate(sentence,predict,id2word,id2tag,entityres)

print '\nentity:', len(entityres)

for entity_code in entityres:
	entity = ''
	for i in range(len(entity_code)-1):
		entity += entity_code[i].split('/')[0]
	print(entity.encode('utf-8'))