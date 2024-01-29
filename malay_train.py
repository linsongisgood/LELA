# coding=gbk
import os
import random
import re
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import WEIGHTS_NAME, CONFIG_NAME
from models_o.xlm_roberta5 import xlm_robertamodel

import pandas as pd

nclass= {"Negative ": 0, 'Negative': 0, 'Positive ': 1, 'Positive': 1, "Mixed_feelings ": 2, "Mixed_feelings": 2, "unknown_state ": 3, "unknown_state": 3, "not-Tamil ": 4, "not-Tamil": 4}
path = ''
path0_1 = ''
path0_11 = ''
path1 = ''
path1_1 = ''
path1_11 = ''

sentences0_1 = []
sentences0_11 = []
sentences1_1 = []
sentences1_11 = []
data = pd.read_csv(path, sep='\t')
sentences = data.text
labels_t = data.category
labels = []
for label in labels_t:
    temp = int(nclass[label])
    labels.append(int(temp))
with open(path0_1, "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=2)
        x = a[2]
        sentences0_1.append(x)
with open(path0_11, "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=2)
        x = a[2]
        sentences0_11.append(x)


data1 = pd.read_csv(path1, sep='\t')
sentences2 = data1.text
labels_t2 = data1.category
labels2 = []
for label in labels_t2:
    temp = nclass[label]
    labels2.append(int(temp))
with open(path1_1, "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=2)
        x = a[2]
        sentences1_1.append(x)
with open(path1_11, "r") as f:
    lines = f.readlines()
    for line in lines:
        a = re.split(r"\s+", line, maxsplit=2)
        x = a[2]
        sentences1_11.append(x)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../../../root/autodl-tmp/xlm-roberta-large")

input_ids = []
attention_masks = []
p1 = []
input_ids_1 = []
attention_masks_1 = []
p2 = []
input_ids_2 = []
attention_masks_2= []
p3 = []
input_ids2 = []
attention_masks2 = []
p4 = []
input_ids2_2 = []
attention_masks2_2 = []
p5 = []
input_ids2_3 = []
attention_masks2_3 = []
p6 = []

for sent in sentences:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
from language_tag import lang_tag
p1 = torch.tensor(lang_tag(sentences))

for sent in sentences0_1:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids_1.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks_1.append(encoded_dict['attention_mask'])
p2 = torch.tensor(lang_tag(sentences0_1))

for sent in sentences0_11:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids_2.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks_2.append(encoded_dict['attention_mask'])
p3 = torch.tensor(lang_tag(sentences0_11))

for sent in sentences2:
    encoded_dict2 = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids2.append(encoded_dict2['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks2.append(encoded_dict2['attention_mask'])
p4 = torch.tensor(lang_tag(sentences2))

for sent in sentences1_1:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids2_2.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks2_2.append(encoded_dict['attention_mask'])
p5 = torch.tensor(lang_tag(sentences1_1))

for sent in sentences1_11:
    encoded_dict = tokenizer(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=40,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    # �ѱ���ľ��Ӽ���list.
    input_ids2_3.append(encoded_dict['input_ids'])
    # ���� attention mask (simply differentiates padding from non-padding).
    attention_masks2_3.append(encoded_dict['attention_mask'])
p6 = torch.tensor(lang_tag(sentences1_11))
# ��lists תΪ tensors.

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
input_ids_1 = torch.cat(input_ids_1, dim=0)
attention_masks_1 = torch.cat(attention_masks_1, dim=0)
input_ids_2 = torch.cat(input_ids_2, dim=0)
attention_masks_2 = torch.cat(attention_masks_2, dim=0)
labels = torch.tensor(labels)

input_ids2 = torch.cat(input_ids2, dim=0)
attention_masks2 = torch.cat(attention_masks2, dim=0)
input_ids2_2 = torch.cat(input_ids2_2, dim=0)
attention_masks2_2 = torch.cat(attention_masks2_2, dim=0)
input_ids2_3 = torch.cat(input_ids2_3, dim=0)
attention_masks2_3 = torch.cat(attention_masks2_3, dim=0)
labels2 = torch.tensor(labels2)

from torch.utils.data import TensorDataset
# ��input ���� TensorDataset��
dataset = TensorDataset(input_ids, attention_masks,p1, input_ids_1, attention_masks_1,p2,input_ids_2, attention_masks_2,p3, labels)
val_dataset = TensorDataset(input_ids2, attention_masks2,p4, input_ids2_2, attention_masks2_2,p5,input_ids2_3, attention_masks2_3,p6, labels2)

from torch.utils.data import DataLoader, RandomSampler

batch_size = 32
train_dataloader = DataLoader(
    dataset,  # ѵ������.
    sampler=RandomSampler(dataset),  # ����˳��
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,  # ��֤����.
    # sampler = RandomSampler(val_dataset), # ����˳��
    batch_size=batch_size
)
from transformers import AdamW, AutoConfig
config = AutoConfig.from_pretrained('../../../root/autodl-tmp/xlm-roberta-large', trust_remote_code=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = xlm_robertamodel(config=config)
model.to('cuda:0')

# AdamW ��һ�� huggingface library ���࣬'W' ��'Weight Decay fix"����˼��
optimizer = AdamW(model.parameters(),
                  lr=1e-5,  # args.learning_rate - Ĭ���� 5e-5
                  eps=1e-8  # args.adam_epsilon  - Ĭ���� 1e-8�� ��Ϊ�˷�ֹ˥���ʷ�ĸ����0
                  )

from transformers import get_linear_schedule_with_warmup

# bert �Ƽ� epochs ��2��4֮��Ϊ�á�
epochs = 12

# training steps ������: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# ��� learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=50,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def flat_accuracy(preds,labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # ���� hh:mm:ss ��ʽ��ʱ��
    return str(datetime.timedelta(seconds=elapsed_rounded))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


output_dir = "./binary3"
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)



# ��¼training ,validation loss ,validation accuracy and timings.
training_stats = []

# ������ʱ��.
total_t0 = time.time()
best_val_accuracy = 0

for epoch_i in range(0, epochs):
    print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

    # ��¼ÿ�� epoch ���õ�ʱ��
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    tmp  = 10
    for step, batch in enumerate(train_dataloader):
        # ÿ��40��batch ���һ������ʱ��.
        if step % 50 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.  loss:{:}'.format(step, len(train_dataloader), elapsed,loss))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        p1 = batch[2].to(device)
        b_input_ids_1 = batch[3].to(device)
        b_input_mask_1 = batch[4].to(device)
        p2 = batch[5].to(device)
        b_input_ids_2 = batch[6].to(device)
        b_input_mask_2 = batch[7].to(device)
        p3 = batch[8].to(device)
        b_labels = batch[9].to(device)

        # ����ݶ�
        model.zero_grad()
        # forward
        outputs = model(input_ids = b_input_ids,attention_mask = b_input_mask,p1= p1,
                        b_input_ids_1  = b_input_ids_1, b_input_mask_1 = b_input_mask_1,p2 = p2,
                        b_input_ids_2 =  b_input_ids_2, b_input_mask_2 =  b_input_mask_2,p3 = p3,
                        b_labels = b_labels,tmp = tmp)
        loss, logits = outputs[:2]
        total_train_loss += loss.item()
        # backward ���� gradients.
        loss.backward()
        # ��ȥ����1 ���ݶȣ�������Ϊ 1.0, �Է��ݶȱ�ը.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # ����ģ�Ͳ���
        optimizer.step()
        # ���� learning rate.
        scheduler.step()

        logit = logits.detach().cpu().numpy()
        label_id = b_labels.to('cpu').numpy()
        # ����training ���ӵ�׼ȷ��.
        total_train_accuracy += flat_accuracy(logit, label_id)

        # ����batches��ƽ����ʧ.
    avg_train_loss = total_train_loss / len(train_dataloader)
    # ����ѵ��ʱ��.
    training_time = format_time(time.time() - t0)

    # ѵ������׼ȷ��.
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    #print("  ѵ��׼ȷ��: {0:.2f}".format(avg_train_accuracy))
    print("  ƽ��ѵ����ʧ loss: {0:.2f}".format(avg_train_loss))
    print("  ѵ��ʱ��: {:}".format(training_time))


    t0 = time.time()
    # ���� model Ϊvaluation ״̬����valuation״̬ dropout layers ��dropout rate�᲻ͬ
    model.eval()
    # ���ò���
    labels_pred1 = []
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        p1 = batch[2].to(device)
        b_input_ids_1 = batch[3].to(device)
        b_input_mask_1 = batch[4].to(device)
        p2 = batch[5].to(device)
        b_input_ids_2 = batch[6].to(device)
        b_input_mask_2 = batch[7].to(device)
        p3 = batch[8].to(device)
        b_labels = batch[9].to(device)

        # ��valuation ״̬��������Ȩֵ�����ı����ͼ
        with torch.no_grad():
             outputs2 = model(input_ids = b_input_ids,attention_mask = b_input_mask,p1= p1, b_input_ids_1  = b_input_ids_1,
                              b_input_mask_1 = b_input_mask_1,p2 = p2,b_input_ids_2 =  b_input_ids_2,
                              b_input_mask_2 =  b_input_mask_2,p3 = p3, b_labels = b_labels,tmp = tmp)

        loss2, logits2 = outputs2[:2]
        # ���� validation loss.
        total_eval_loss += loss2.item()
        logit = logits2.detach().cpu().numpy()

        label_id = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_pred1 = np.append(labels_pred1, pred_flat)
        # ���� validation ���ӵ�׼ȷ��.
        total_eval_accuracy += flat_accuracy(logit, label_id)

    # ���� validation ��׼ȷ��.
    print(classification_report(labels2, labels_pred1, digits=4))
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("")
    print("  ����׼ȷ��: {0:.2f}".format(avg_val_accuracy))

    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), output_model_file)

    # ����batches��ƽ����ʧ.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # ����validation ʱ��.
    validation_time = format_time(time.time() - t0)

    print("  ƽ��������ʧ Loss: {0:.2f}".format(avg_val_loss))
    print("  ����ʱ��: {:}".format(validation_time))


print("ѵ��һ������ {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))






