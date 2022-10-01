from collections import defaultdict
from numpy import percentile
from random import random
from pathlib import Path
from typing import List, Union, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast, BatchEncoding
from src.data_processing.rumor.base import RumorDataProcessor
from src.data_processing.review.base_normal import ReviewDataProcessor_normal
from src.data_processing.rumor.base_unlabelled import RumorunlabelledDataProcessor
from src.utils.constants import DATA_DIR, EXP_DIR
import torch


class ReviewPadaDataProcessor(ReviewDataProcessor_normal):

    def __init__(self, src_domains: List[str], trg_domain: str, data_dir: Union[str, Path], experiment_dir: Union[str, Path]):
        super().__init__(src_domains, trg_domain, data_dir)
        self.experiment_dir = experiment_dir
        for split in ReviewDataProcessor_normal.ALL_SPLITS:
            self.data[split]["drfs"] = self._read_drfs(split)

    def _read_drfs(self, mode: str = 'train'):
        domains = self.src_domains if mode != 'test' else [self.trg_domain]
        if mode == 'test' or mode == 'dev':
            return ['' for _ in range(len(self.data[mode]["input_str"]))]
        all_drfs = list()
        for domain in domains:
            prompts_path = Path(self.experiment_dir) / "review" / self.trg_domain / "prompt_annotations" / domain
            with open(prompts_path / f"annotated_prompts_train.pt", "rb") as f:
                drfs = torch.load(f)
            all_drfs.extend(drfs)
        return all_drfs




class ReviewPadaDataset(Dataset):

    DOMAIN_PROMPT = "domain"
    Review_PROMPT = "review"
    DRF_DELIMITER = ", "

    def __init__(self, split: str, data_processor: ReviewPadaDataProcessor, tokenizer: T5TokenizerFast,
                 max_seq_len: int = 512, max_drf_seq_len: int = 20, mixture_alpha: float = 0.2):
        self.split = split
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_drf_seq_len = max_drf_seq_len
        self.mixture_alpha = mixture_alpha
        self.tokenized_data = self._init_tokenized_data(split, data_processor)

    def __len__(self):
        if self.split == "train":
            return self.tokenized_data["num_examples"]
        else:
            return len(self.tokenized_data["domain_label"])

    def __getitem__(self, index):
        if self.split == "train":
            if random() <= self.mixture_alpha:
                tokenized_data = self.tokenized_data["domain"]
            else:
                tokenized_data = self.tokenized_data["drf"]
        else:
            tokenized_data = self.tokenized_data
        return {
            k: v[index]
            for k, v in tokenized_data.items()
        }

    def tokenize_input_str(self, input_str: List[List[str]]) -> BatchEncoding:
        return self.tokenizer(input_str, is_split_into_words=False,
                              padding='max_length', truncation=True,
                              max_length=self.max_seq_len, return_tensors="pt", return_attention_mask=True)

    def tokenize_output_str(self, output_str: List[List[str]]) -> BatchEncoding:
        return self.tokenizer(output_str, is_split_into_words=False, padding='max_length', truncation=True,
                              max_length=self.max_drf_seq_len, return_tensors="pt")["input_ids"]

    @staticmethod
    def _build_training_tokenized_data(data):
        domain_data = defaultdict(list)
        drf_data = defaultdict(list)
        for i in range(len(data["domain_label"])):
            drfs_str = f"{data['domain_label'][i]} - " + f"{ReviewPadaDataset.DRF_DELIMITER.join(data['drfs'][i])}:"
            prompt = f"{ReviewPadaDataset.DOMAIN_PROMPT}: "
            domain_data["input_str"].append(prompt + data["input_str"][i])
            domain_data["output_str"].append(drfs_str)
            domain_data["output_labels"].append(data["output_label"][i])
            domain_data["prompt"].append(prompt)

            prompt = f"{drfs_str}: "
            drf_data["input_str"].append(prompt + data["input_str"][i])
            drf_data["output_str"].append("")
            drf_data["output_labels"].append(data["output_label"][i])
            drf_data["prompt"].append(prompt)
        return {"domain": domain_data, "drf": drf_data}

    def _init_tokenized_data(self, split, data_processor) -> Union[BatchEncoding, Dict[str, BatchEncoding]]:
        data = data_processor.get_split_data(split)
        if split == "train":
            tokenized_data_dict = dict(num_examples=len(data["domain_label"]))
            for key, data_dict in self._build_training_tokenized_data(data).items():
                tokenized_data = self.tokenize_input_str(data_dict["input_str"])
                tokenized_data["prompt_output_ids"] = self.tokenize_output_str(data_dict["output_str"])
                tokenized_data["output_label_id"] = data_dict["output_labels"]
                tokenized_data["prompt"] = data_dict["prompt"]
                tokenized_data["input_str"] = data["input_str"]
                tokenized_data["domain_label"] = data["domain_label"]
                tokenized_data["drfs"] = data["drfs"]

                tokenized_data_dict[key] = tokenized_data
            return tokenized_data_dict
        else:
            tokenized_data = self.tokenize_input_str(data["input_str"])
            tokenized_data["output_label_id"] = data["output_label"]
            tokenized_data["input_str"] = data["input_str"]
            tokenized_data["domain_label"] = data["domain_label"]
            tokenized_data["drfs"] = data["drfs"]

        return tokenized_data

