# -*- coding: utf-8 -*-

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader
import random
import json
import argparse

# args=argparse.

seq_length=71

"""
使用SimCLUE数据集，进一步调优sentence-transformer
定义训练和验证、加载的模型、损失函数，训练并保存模型
"""
def read_txt(input_file):
    """
    读取数据
    :param input_file: txt文件
    :return: [[text1, text2, int(label)], [text1, text2, int(label)]]
    """
    with open(input_file, 'r', encoding='utf8') as f:
        reader = f.readlines()
    lines = []
    for line in reader:
        json_data=json.loads(line.strip()) # {"sentence1": "英德是哪个省", "sentence2": "英德是哪个市的", "label": "0"}
        text1, text2, label = json_data['sentence1'],json_data['sentence2'],json_data['label']
        lines.append([text1, text2, int(label)])
    random.shuffle(lines)
    return lines

# 定义加载的模型
# model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model = SentenceTransformer("bert-base-chinese")

train_datas = read_txt('/cephfs/luoyifei/work/SimCLUE/datasets/full/train_pair.json') # 大规模训练可以改成train_pair.json
test_datas = read_txt('/cephfs/luoyifei/work/SimCLUE/datasets/full/dev.json')
test_public_datas = read_txt('/cephfs/luoyifei/work/SimCLUE/datasets/full/test_public.json')
train_size = len(train_datas)
eval_size = len(test_datas)
train_data = []
for idx in range(train_size):
    if len(train_datas[idx][0]) > seq_length or len(train_datas[idx][1])>seq_length: continue
    train_data.append(InputExample(texts=[train_datas[idx][0], train_datas[idx][1]], label=float(train_datas[idx][2])))

print("train avg:")
print(sum([len(x.texts[0]) for x in train_data])/len(train_data))
print(sum([len(x.texts[1]) for x in train_data])/len(train_data))

print("train max:")
print(max([len(x.texts[0]) for x in train_data]))
print(max([len(x.texts[1]) for x in train_data]))

print("95% data:")
import numpy as np
print(np.percentile(np.array([len(x.texts[1]) for x in train_data]), 95))


print("train size:" )
print(train_size)


# 设置验证集 Define your evaluation examples
sentences1 = []
sentences2 = []
scores = []
labels = []
for ss in test_datas:
    if len(ss[0]) > seq_length or len(ss[1])>seq_length: continue
    sentences1.append(ss[0])
    sentences2.append(ss[1])
    labels.append(int(ss[2]))
evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, labels)

# 设置测试集 
sentences1 = []
sentences2 = []
scores = []
labels = []
for ss in test_public_datas:
    if len(ss[0]) > seq_length or len(ss[1]) > seq_length: continue
    sentences1.append(ss[0])
    sentences2.append(ss[1])
    labels.append(int(ss[2]))
test_evaluator = evaluation.BinaryClassificationEvaluator(sentences1, sentences2, labels)



# 定义训练集、加载数据集和训练的损失 Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32) #64
train_loss = losses.CosineSimilarityLoss(model)


# 训练模型
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100, evaluator=evaluator,
          evaluation_steps=100, output_path='./sentence_transformer_checkpoint',save_best_model=True,use_amp=True,checkpoint_path="./sentence_transformer_checkpoint/checkpoint",checkpoint_save_total_limit=5)


##Test
model=SentenceTransformer('./sentence_transformer_checkpoint')
# evaluator = evaluation.BinaryClassificationEvaluator.from_input_examples()
test_evaluator(model,output_path='./sentence_transformer_checkpoint/Test')