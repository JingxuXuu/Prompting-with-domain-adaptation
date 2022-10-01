
import sys
sys.path.append("/home/ubuntu/efs/prompt/PADA")

from src.utils.constants import DATA_DIR, EXP_DIR
from src.data_processing.absa.pada import AbsaSeq2SeqPadaDataProcessor, AbsaSeq2SeqPadaDataset
from src.data_processing.rumor.pada import RumorPadaDataProcessor, RumorPadaDataset
from src.data_processing.mnli.pada import MNLIPadaDataProcessor, MNLIPadaDataset
from src.data_processing.review.pada import ReviewPadaDataProcessor, ReviewPadaDataset
from src.modeling.token_classification.pada_seq2seq_token_classifier import PadaSeq2SeqTokenClassifierGeneratorMulti
from src.modeling.text_classification.pada_text_classifier import PadaTextClassifierMulti
from src.utils.train_utils import set_seed, ModelCheckpointWithResults, LoggingCallback
from pathlib import Path
from argparse import Namespace, ArgumentParser
from pytorch_lightning import Trainer

import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--param0')
parser.add_argument('--param1') #source domain
parser.add_argument('--param2') #target domain
args = parser.parse_args()

data = args.param0

args_dict = dict(
    model_name="T5classificationPADA",
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
seed = 30
set_seed(seed)
model_hparams_dict = args_dict
model_hparams_dict['src_domains'] =args.param1.split(",")


train_args = dict(
            default_root_dir=model_hparams_dict["output_dir"],
            accumulate_grad_batches=model_hparams_dict["gradient_accumulation_steps"],
            gpus=model_hparams_dict["n_gpu"],
            max_epochs=model_hparams_dict["num_train_epochs"],
            precision=16 if model_hparams_dict.pop("fp_16") else 32,
            gradient_clip_val=model_hparams_dict.pop("max_grad_norm"),
            fast_dev_run=model_hparams_dict.pop("fast_dev_run"),
            deterministic=False,
            benchmark=False
        )


model_name = model_hparams_dict.pop("model_name")
if data == "review":
    model_obj, data_procesor_obj, dataset_obj= PadaTextClassifierMulti, ReviewPadaDataProcessor, ReviewPadaDataset
elif data == "rumor":
    model_obj, data_procesor_obj, dataset_obj = PadaTextClassifierMulti, RumorPadaDataProcessor, RumorPadaDataset
elif data == "mnli":
    model_obj, data_procesor_obj, dataset_obj = PadaTextClassifierMulti, MNLIPadaDataProcessor, MNLIPadaDataset
model_hparams_dict["data_procesor_obj"] = data_procesor_obj
model_hparams_dict["dataset_obj"] = dataset_obj
print(model_hparams_dict['experiment_dir'])
model = model_obj(**model_hparams_dict)
trainer = Trainer(**train_args)
trainer.fit(model)
test_ckpt = "best"
trainer.test(ckpt_path=test_ckpt)