import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from models import *


class xlm_robertamodel(nn.Module):
    def __init__(self, config):
        super(xlm_robertamodel, self).__init__()
        self.h_size = config.hidden_size
        self.encoder = XLMRobertaForSequenceClassification.from_pretrained('../../../root/autodl-tmp/xlm-roberta-large', return_dict=True,
                                                 output_hidden_states=True)
        self.peft_config = PLoraConfig(task_type="SEQ_CLS",
                                  inference_mode=False,
                                  r=64,
                                  lora_alpha=128,
                                  target_modules=['query',"key","value" ],
                                  lora_dropout=0.1,
                                  num_virtual_users=3,
                                  user_token_dim=1024)
        self.model = get_peft_model(self.encoder, self.peft_config)
        self.out_proj = nn.Linear(config.hidden_size, 5)
        self.out_proj1 = nn.Linear(config.hidden_size, 5)
        self.out_proj2 = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids ,attention_mask ,p1,
                        b_input_ids_1, b_input_mask_1,p2,
                        b_input_ids_2 , b_input_mask_2 ,p3,
                        b_labels,tmp):

        outputs = self.model(input_ids =input_ids, attention_mask = attention_mask,p =p1)
        outputs1 = self.model(input_ids = b_input_ids_1, attention_mask = b_input_mask_1,p = p2)
        outputs2 = self.model(input_ids = b_input_ids_2, attention_mask = b_input_mask_2,p = p3)

        last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state1 = outputs1.hidden_states[-1]
        last_hidden_state2 = outputs2.hidden_states[-1]

        cls_embeddings = last_hidden_state[:, 0]
        cls_embeddings1 = last_hidden_state1[:, 0]
        cls_embeddings2 = last_hidden_state2[:, 0]

        #labels = torch.eye(3)[labels]
        labels = b_labels.to('cuda:0')

        logits = self.out_proj(cls_embeddings)
        #logits = torch.nn.functional.softmax(logits, dim=1)

        logits1 = self.out_proj1(cls_embeddings1)
        #logits1 = torch.nn.functional.softmax(logits1, dim=1)

        logits2 = self.out_proj2(cls_embeddings2)
        #logits2 = torch.nn.functional.softmax(logits2, dim=1)

        soft_loss  = torch.nn.KLDivLoss(reduction='batchmean')
        loss1 = soft_loss(torch.log_softmax(logits1 / tmp, dim=-1), torch.softmax(logits / tmp, dim=-1))
        loss2 = soft_loss(torch.log_softmax(logits2 / tmp, dim=-1), torch.softmax(logits / tmp, dim=-1))
        loss3 = soft_loss(torch.log_softmax(logits2 / tmp, dim=-1), torch.softmax(logits1 / tmp, dim=-1))

        loss_fct = CrossEntropyLoss()
        loss4 = loss_fct(logits.view(-1, 5), labels.view(-1))
        loss5 = loss_fct(logits1.view(-1, 5), labels.view(-1))
        loss6 = loss_fct(logits2.view(-1, 5), labels.view(-1))

        logits = logits+logits1+logits2

        loss  = 0.6*(loss4+loss5+loss6)+0.4*(loss1+loss2+loss3)*tmp*tmp
        #print(0.1*(loss1+loss2),0.9*loss3)
        return loss, logits


