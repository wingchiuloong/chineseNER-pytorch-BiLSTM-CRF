#!/usr/bin/python
# coding=utf-8
import pickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from resultCal import calculate
with open('../data/renmindata.pkl', 'rb') as inp:
	word2id = pickle.load(inp) 
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)

# 将训练好的模型数据导入
model_data_filepath = './model/model2.pkl' # 逐个句子输入模型进行分析
model = torch.load(model_data_filepath, map_location=torch.device('cpu'))

sentence_raw = raw_input('Enter your sentence:\n')
sentence_raw = unicode(sentence_raw, 'utf-8') # 因为word2id中的word是unicode，所以先把输入的字符串变成unicode

sentence_split = []
sentence = ''
for word in sentence_raw: # 根据标点符号，把句子分割
	if word not in [u'，', u'。', u'！', u'？', u'-', u'“', u'”', u',', u'.', u'?', u'!']:
		sentence += word
	else:
		sentence_split.append(sentence)
		sentence = ''

print '\nresult(人名)[组织名]<地名>:'
# 逐个句子输入模型进行分析
for sentence in sentence_split: 
	sentence_id = []
	for word in sentence: # 根据word2id把每个字转成id
		if word in word2id:
			sentence_id.append(word2id[word])
		else: # 在word2id内找不到该字，就置0
			sentence_id.append(word2id[u'unknow'])
	# 将句子id放进模型测试
	entityres=[]
	sentence_tensor = torch.tensor(sentence_id, dtype=torch.long)
	score,predict = model(sentence_tensor)
	entityres = calculate(sentence_tensor,predict,id2word,id2tag,entityres)
	# 打印结果
	print sentence, ':',
	for entity_code in entityres:
		entity = ''
		entity_type = ''
		for i in range(len(entity_code)-1):
			e_split = entity_code[i].split('/')
			entity += e_split[0]
			entity_type = e_split[1][2:4]
		if entity_type == u'nr':
			print '(', entity.encode('utf-8').strip(), ')',
		elif entity_type == u'nt':
			print '[', entity.encode('utf-8').strip(), ']',
		if entity_type == u'ns':
			print '<', entity.encode('utf-8').strip(), '>',
	print ''