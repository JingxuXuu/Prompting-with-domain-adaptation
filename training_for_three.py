import os
import sys
import torch
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import random
from tqdm import tqdm
import time
import os

sys.path.append("/home/ubuntu/efs/PADA")
from src.utils.constants import DATA_DIR, EXP_DIR
from src.data_processing.absa.pada import AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset
from src.data_processing.rumor.base import RumorDataProcessor, RumorDataset
from src.data_processing.rumor.base_unlabelled import RumorunlabelledDataProcessor, RumorunlabelledDataset
from src.data_processing.rumor.pada import RumorPadaDataProcessor, RumorPadaDataset
from src.data_processing.mnli.base import MNLIDataProcessor, MNLIDataset
from src.data_processing.mnli.base_unlabelled import MNLIunlabelledDataProcessor, MNLIunlabelledDataset
from src.data_processing.review.base import ReviewDataProcessor, ReviewDataset
from src.data_processing.review.base_normal import ReviewDataProcessor_normal, ReviewDataset_normal
from src.data_processing.review.base_unlabelled import ReviewunlabelledDataProcessor, ReviewunlabelledDataset
from src.modeling.token_classification.pada_seq2seq_token_classifier import PadaSeq2SeqTokenClassifierGeneratorMulti
from src.modeling.text_classification.t5_text_classifier import T5TextClassifier
from src.modeling.text_classification.t5_domain_classifier import T5DomainClassifier

from src.modeling.token_classification.pada_seq2seq_token_classifier import PadaSeq2SeqTokenClassifierGeneratorMulti
from src.modeling.text_classification.t5_text_classifier import T5TextClassifier
from src.modeling.text_classification.pada_text_classifier import PadaTextClassifierMulti
from src.utils.train_utils import set_seed, ModelCheckpointWithResults, LoggingCallback
from pathlib import Path
from argparse import Namespace, ArgumentParser
from pytorch_lightning import Trainer
import random
import os
import json
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--param0') #use unlabelled data or not
parser.add_argument('--param1') #source domain
parser.add_argument('--param2') #target domain
parser.add_argument('--param3') #datasets
parser.add_argument('--param4') #freeze or not
parser.add_argument('--param5') #seed
parser.add_argument('--param6') #do few-shot or not
 #template
args = parser.parse_args()
data = args.param3
seed = int(args.param5)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
seed_everything(seed)

few_shot = args.param6
unlabelled=args.param0

if args.param4 == "True":
    freeze = True
else:
    freeze = False

args_dict = dict(
    model_name="T5classification",
    src_domains=args.param1,
    trg_domain=args.param2,
    data_dir=str(DATA_DIR),  # path to data files
    experiment_dir=str(EXP_DIR),  # path to base experiment dir
    output_dir=str(EXP_DIR),  # path to save the checkpoints
    t5_model_name='t5-base',
    max_seq_len=512,
    learning_rate=5e-5,
    weight_decay=1e-5,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=5,
    gradient_accumulation_steps=1,
    n_gpu=1,
    fast_dev_run=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    beam_size=10,
    repetition_penalty=2.0,
    length_penalty=1.0,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
    num_beam_groups=5,
    diversity_penalty=0.2,
    eval_metrics=[ "micro_f1", "macro_f1", "weighted_f1"],
)
set_seed(seed)
model_hparams_dict = args_dict
model_hparams_dict['src_domains'] = args.param1.split(",")


amp_level=model_hparams_dict.pop("opt_level"),
train_args = dict(
            default_root_dir=model_hparams_dict["output_dir"],
            accumulate_grad_batches=model_hparams_dict["gradient_accumulation_steps"],
            gpus=model_hparams_dict["n_gpu"],
            max_epochs=model_hparams_dict["num_train_epochs"],
            precision=16 if model_hparams_dict.pop("fp_16") else 32,
            gradient_clip_val=model_hparams_dict.pop("max_grad_norm"),
            fast_dev_run=model_hparams_dict.pop("fast_dev_run"),
            deterministic=True,
            benchmark=False
        )


model_name = model_hparams_dict.pop("model_name")

if data == "Rumor":
    if few_shot == "False":
        model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, RumorunlabelledDataProcessor, RumorunlabelledDataset
    else:
        if unlabelled=="True":
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, RumorunlabelledDataProcessor, RumorunlabelledDataset
        else:
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, RumorDataProcessor, RumorDataset
if data == "MNLI":
    if few_shot == "False":
        model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, MNLIunlabelledDataProcessor, MNLIunlabelledDataset
    else:
        if unlabelled=="True":
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, MNLIunlabelledDataProcessor, MNLIunlabelledDataset
        else:
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, MNLIDataProcessor, MNLIDataset
if data == "Review":
    if few_shot == "False":
        model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, ReviewunlabelledDataProcessor, ReviewunlabelledDataset
    else:
        if unlabelled=="True":
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, ReviewunlabelledDataProcessor, ReviewunlabelledDataset
        else:
            model_obj, data_procesor_obj, dataset_obj = T5DomainClassifier, ReviewDataProcessor, ReviewDataset

model_hparams_dict["data_procesor_obj"] = data_procesor_obj
model_hparams_dict["dataset_obj"] = dataset_obj

model = model_obj(**model_hparams_dict)
trainer = Trainer(**train_args)
trainer.fit(model)

# we will not train the domain classifier further and will freeze the parameters
for n,p in model.named_parameters():
    p.requires_grad = False


from numpy import percentile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
from src.utils.constants import DATA_DIR
import pickle
from PADA.src.data_processing.rumor.base import *
from PADA.src.data_processing.mnli.base import *
from transformers import BertTokenizer
if data == "Rumor":
    if few_shot == "True":
        Review_data =  RumorDataProcessor(src_domains=args.param1.split(","),
                                    trg_domain=args.param2,data_dir=DATA_DIR).data
    else:
        Review_data = RumorunlabelledDataProcessor(src_domains=args.param1.split(","),
                                         trg_domain=args.param2, data_dir=DATA_DIR).data
if data == "MNLI":
    if few_shot == "True":
        Review_data = MNLIDataProcessor(src_domains=args.param1.split(","),
                                     trg_domain=args.param2, data_dir=DATA_DIR).data
    else:
        Review_data = MNLIunlabelledDataProcessor(src_domains=args.param1.split(","),
                                        trg_domain=args.param2, data_dir=DATA_DIR).data
if data == "Review":
    if few_shot == "True":
        Review_data = ReviewDataProcessor(src_domains=args.param1.split(","),
                                        trg_domain=args.param2, data_dir=DATA_DIR).data
    else:
        Review_data = ReviewDataProcessor_normal(src_domains=args.param1.split(","),
                                          trg_domain=args.param2, data_dir=DATA_DIR).data
X_train = Review_data['train']['input_str']
X_val = Review_data['dev']['input_str']
X_test = Review_data['test']['input_str']

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
import numpy as np
device=torch.device('cuda')
tokenizer = T5Tokenizer.from_pretrained("t5-base",do_lower_case=True)


class Fix_embed():
    def __init__(self, input_ids, attention_mask, domain_classification_model):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.domain_classification_model = domain_classification_model

        with torch.no_grad():
            output1 = torch.mean(self.domain_classification_model.model.encoder(torch.unsqueeze(self.input_ids,0),torch.unsqueeze(self.attention_mask,0))[0],
                               dim=1)
            output2 = self.domain_classification_model.classifier(
                self.domain_classification_model.model.encoder(torch.unsqueeze(self.input_ids, 0),
                                                               torch.unsqueeze(self.attention_mask, 0))[0])

        self.output = [output2[2],output1]


#candidatetext is the candidate set for templates and varies with respect to different datasets
if data == "Rumor":
    text1 = '{"placeholder":"text_a"} Is it a rumor? The answer is {"mask"}.'
    text2 = '{"soft":None, "fixed_id":1} {"placeholder":"text_a"} Is it a rumor? The answer is {"mask"}.'
    text3 = '{"soft":None, "fixed_id":2} {"placeholder":"text_a"} Is it a rumor? The answer is {"mask"}.'
    text4 = '{"soft":None, "fixed_id":1} {"soft":None} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text5 = '{"soft":None, "fixed_id":2} {"soft":None} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text6 = '{"soft":None} {"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}.'
    text7 = '{"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} {"placeholder":"text_a"} Is it a rumor? The answer is {"mask"}.'
    text8 = '{"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} {"soft":None} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text9 = '{"soft":None} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text10 = '{"soft":None} {"soft":None, "fixed_id":1} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text11 = '{"soft":None} {"soft":None, "fixed_id":2} {"placeholder":"text_a"} Is it a rumor? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text12 = '{"placeholder":"text_a"} Is it a rumor {"soft":None, "fixed_id":1} ? The answer is {"mask"}.'
    text13 = '{"placeholder":"text_a"} {"soft":None, "fixed_id":1} Is it a rumor? The answer is {"mask"}.'
    text14 = '{"placeholder":"text_a"} {"soft":None, "fixed_id":2} Is it a rumor? The answer is {"mask"}.'
    text15 = '{"placeholder":"text_a"} Is it a rumor {"soft":None, "fixed_id":2} ? The answer is {"mask"}.'
    if unlabelled == "True":
        candidatetext=[text2,text3,text4,text5,text7,text8]
    else:

        candidatetext = [text1]
elif data == "MNLI":
    text1 = 'Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    text2 = '{"soft":None, "fixed_id":1} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}?  Is it correct? {"mask"}.'
    text3 = '{"soft":None, "fixed_id":2} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}?  Is it correct? {"mask"}.'
    text4 = '{"soft":None, "fixed_id":1} {"soft":None} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text5 = '{"soft":None, "fixed_id":2} {"soft":None} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text6 = '{"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    text7 = '{"soft":None} {"placeholder":"text_a"} {"soft"} {"placeholder":"text_b"} {"soft"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}'
    text8 = '{"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} {"soft":None} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text9 = '{"soft":None} Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"soft":"The"} {"soft":"answer"} {"soft":"is"} {"mask"}.'
    text10 = 'Premise: {"placeholder":"text_a"} {"soft":None, "fixed_id":1} Question: {"placeholder":"text_b"}?  Is it correct? {"mask"}.'
    text11 = 'Premise: {"placeholder":"text_a"} {"soft":None, "fixed_id":2} Question: {"placeholder":"text_b"}?  Is it correct? {"mask"}.'
    text12 = 'Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? {"soft":None, "fixed_id":1} Is it correct? {"mask"}. '
    text13 = 'Premise: {"placeholder":"text_a"} Question: {"placeholder":"text_b"}? {"soft":None, "fixed_id":2} Is it correct? {"mask"}. '
    if unlabelled=="True":
        candidatetext = [text2,text3,text4,text5,text6,text8]
    else:
        candidatetext=[text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11,text12,text13]
elif data == "Review":
    text1 = '{"placeholder":"text_a"} In summary, it was {"mask"}.'
    text2 = '{"soft":None} {"placeholder":"text_a"} {"soft": "It"} {"soft": "was"} {"mask"}.'
    text3 = '{"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text4 = '{"soft":None, "fixed_id":1} {"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text6 = '{"soft":None, "fixed_id":2} {"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text7 = '{"soft":"Domain"} {"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} {"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text8 = '{"soft":None, "fixed_id":1} {"soft":None, "fixed_id":2} {"placeholder":"text_a"} In summary, it was {"mask"}.'

    text10 = '{"soft":None, "fixed_id":1} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text11 = '{"soft":None, "fixed_id":2} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text13 = '{"soft":"Domain"} {"soft":None, "fixed_id":1} {"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text14 = '{"soft":"Domain"} {"soft":None, "fixed_id":2} {"soft":None} {"placeholder":"text_a"} In summary, it was {"mask"}.'
    text15 = '{"placeholder":"text_a"} {"soft":None, "fixed_id":1} In summary, it was {"mask"}.'
    text16 = '{"placeholder":"text_a"} {"soft":None, "fixed_id":2} In summary, it was {"mask"}.'
    text17 = '{"placeholder":"text_a"} It is {"soft":None, "fixed_id":1} and {"mask"}.'
    text18 = '{"placeholder":"text_a"} It is {"soft":None, "fixed_id":2} and {"mask"}.'
    if unlabelled == "True":
        candidatetext = [text7, text8, text10, text11, text13, text14]
    else:
        candidatetext = [text10, text11,text4,text6,text8,text7]

from torch import nn
import os
sys.path.append("/home/ubuntu/efs/OpenPrompt")
from openprompt.prompts import MixedTemplate
from openprompt.plms import load_plm

from openprompt.data_utils import InputExample

dataset = {}
if data != "MNLI":
    for split in ['train', 'test','dev']:
        dataset[split] = []
        if split == 'train':
            for i in range(len(X_train)):
                input_example = InputExample(text_a = X_train[i], label=Review_data['train']['output_label'][i])
                dataset[split].append(input_example)
        if split == 'test':
            for i in range(len(X_test)):
                input_example = InputExample(text_a = X_test[i], label=Review_data['test']['output_label'][i])
                dataset[split].append(input_example)
        if split == 'dev':
            for i in range(len(X_val)):
                input_example = InputExample(text_a = X_val[i], label=Review_data['dev']['output_label'][i])
                dataset[split].append(input_example)
else:
    for split in ['train', 'test','dev']:
        dataset[split] = []
        if split == 'train':
            for i in range(len(X_train)):
                input_example = InputExample(text_a=X_train[i][0], text_b=X_train[i][1],
                                             label=Review_data['train']['output_label'][i])
                dataset[split].append(input_example)
        if split == 'test':
            for i in range(len(X_test)):
                input_example = InputExample(text_a=X_test[i][0], text_b=X_test[i][1],
                                             label=Review_data['test']['output_label'][i])
                dataset[split].append(input_example)
        if split == 'dev':
            for i in range(len(X_val)):
                input_example = InputExample(text_a=X_val[i][0], text_b=X_val[i][1],
                                             label=Review_data['dev']['output_label'][i])
                dataset[split].append(input_example)

from openprompt import PromptDataLoader
import torch
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer
from openprompt.prompts import ManualVerbalizer
import torch

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    return acc

for i in range(len(candidatetext)):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    plm, tokenizer, model_config, WrapperClass = load_plm("t5","t5-base")
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=candidatetext[i],fixed_embed=Fix_embed,domain_classification_model=model)
    from openprompt import PromptDataLoader
    train_dataloader = PromptDataLoader(dataset=dataset['train'], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=512,decoder_max_length=3,
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")
    val_dataloader = PromptDataLoader(dataset=dataset['dev'], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                        truncate_method="head")







    if data == "Rumor":
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                                    label_words=[["no"], ["yes"]])
    elif data == "MNLI":
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=3,
                                        label_words=[["maybe"], ["no"], ["yes"]])
    elif data == "Review":
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2,
                                        label_words=[["bad"], ["good"]])
    # we can also use soft verbalizer, for example
    # myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2,
    #          label_words=["no","yes"])








    use_cuda = True
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=freeze)
    if use_cuda:
        prompt_model=  prompt_model.cuda()
    prompt_model.parallelize()
    from transformers import AdamW, get_linear_schedule_with_warmup

    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters
    optimizer_grouped_parameters2 = [
        {'params': [p for n, p in prompt_model.template.named_parameters() if "raw_embedding" not in n and p.requires_grad==True]}
    ]

    # if we use a soft verbalizer we can use a seperate optimizer for verbalizer parameters
    #optimizer_grouped_parameters3 = [
    #    {'params': prompt_model.verbalizer.group_parameters_1, "lr": 3e-5},
    #    {'params': prompt_model.verbalizer.group_parameters_2, "lr": 3e-4},
    #]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=500, num_training_steps=20000)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=0.1)
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=500, num_training_steps=20000)  # usually num_warmup_steps is 500


    tot_loss = 0
    log_loss = 0
    best_val_acc = 0
    glb_step = 0
    actual_step = 0
    leave_training = False

    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10
    prompt_model.train()

    pbar = tqdm(total=20000, desc="Train")
    for epoch in range(2):
        print(f"Begin epoch {epoch}")
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            tot_train_time -= time.time()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            actual_step += 1

            if actual_step % 4 == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1
                if glb_step % pbar_update_freq == 0:
                    aveloss = (tot_loss - log_loss) / pbar_update_freq
                    pbar.update(10)
                    pbar.set_postfix({'loss': aveloss})
                    log_loss = tot_loss


            optimizer1.step()
            optimizer1.zero_grad()
            scheduler1.step()
            optimizer2.step()
            optimizer2.zero_grad()
            scheduler2.step()

            tot_train_time += time.time()

            if actual_step % 8 == 0 and glb_step > 0 and glb_step % 50 == 0:
                val_acc = evaluate(prompt_model, val_dataloader, desc="Valid")
                if val_acc >= best_val_acc:

                    best_val_acc = val_acc

                acc_traces.append(val_acc)
                print(
                    "Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time / actual_step),
                    flush=True)
                prompt_model.train()

            if glb_step > 20000:
                leave_training = True
                break

        if leave_training:
            break



    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=512,decoder_max_length=3,
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")


    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())



    import sklearn

    print(sklearn.metrics.f1_score(alllabels,allpreds,average='weighted'))
    print(sklearn.metrics.f1_score(alllabels,allpreds,average='macro'))
    print(sklearn.metrics.f1_score(alllabels,allpreds,average='micro'))


    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(acc)

    random_file_name = os.path.join('results_position',f"{random.randint(0,100000000)}.json")
    while(os.path.exists(random_file_name)):
        random_file_name = os.path.join('results_position', f"{random.randint(0, 100000000)}.json")
    with open(random_file_name,'w') as nf:
        output ={"unlabelled":args.param0,"freeze":freeze,"fewshot":few_shot, "date":20220829,"param1":args.param1, "param2":args.param2, "template":candidatetext[i],"param3":data, "param4":seed,'result':[sklearn.metrics.f1_score(alllabels,allpreds,average='weighted'),sklearn.metrics.f1_score(alllabels,allpreds,average='macro'),sklearn.metrics.f1_score(alllabels,allpreds,average='micro')]}
        nf.write(json.dumps(output, indent=2))
