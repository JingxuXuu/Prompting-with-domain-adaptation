from numpy import percentile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
from PADA.src.utils.constants import DATA_DIR
import pickle
import json
import xml.etree.ElementTree as ET


class MNLIunlabelledDataProcessor:
    ALL_SPLITS = ("train", "dev", "test")
    WORD_DELIMITER = " "

    def __init__(self, src_domains: List[str], trg_domain: str, data_dir: Union[str, Path]):
        self.data_dir = data_dir
        self.src_domains = src_domains
        self.trg_domain = trg_domain
        self.reduce_label_dict = {
            1: 1,
            0: 0,
            2: 2
        }
        self.labels_dict = {
            0: 0,
            1: 1,
            2: 2
        }
        self.data = self.load_data()

    def read_data_from_file(self, mode: str = 'train') -> Dict[str, List[Union[str, List[str], Tuple[int]]]]:
        domains = self.src_domains if mode != 'test' else [self.trg_domain]
        data_dict = defaultdict(list)
        for domain_idx, domain in enumerate(domains):
            data_path = Path(self.data_dir) / "mnli_data" / domain / mode
            with open(data_path, 'rb') as f:
                (text, labels) = pickle.load(f)
            for i, (txt, lbl) in enumerate(zip(text, labels)):
                data_dict["input_str"].append(txt)
                data_dict["output_label"].append(self.reduce_label_dict[lbl])
                data_dict["domain_label"].append(domain)
                data_dict["domain_label_id"].append(domain_idx)
                data_dict["example_id"].append(f"{domain}_{i + 1}")
        return data_dict


    def load_data(self) -> Dict[str, Dict[str, List[Union[str, List[str], List[int]]]]]:
        return {split: self.read_data_from_file(mode=split) for split in MNLIunlabelledDataProcessor.ALL_SPLITS}


    def get_split_data(self, split: str) -> Dict[str, List[Union[str, List[str], List[int]]]]:
        assert split in MNLIunlabelledDataProcessor.ALL_SPLITS
        return self.data[split]


# def get_split_domain_data(self, split: str, domain: str) -> Dict[str, List[Union[str, List[str], List[int]]]]:
#     assert split in RumorDataProcessor.ALL_SPLITS
#     assert domain in self.src_domains + [self.trg_domain]
#     l, r = 0, len(self.data[split]["domain_label"]) - 1
#     while self.data[split]["domain_label"][l] != domain:
#         l += 1
#     while self.data[split]["domain_label"][r] != domain:
#         r -= 1
#     split_domain_data = defaultdict(list)
#     for k, v in self.data[split].items():
#         split_domain_data[k] = v[l:r+1]
#     return split_domain_data


class MNLIunlabelledDataset(Dataset):
    def __init__(self, split: str, data_processor: MNLIunlabelledDataProcessor, tokenizer: T5TokenizerFast, max_seq_len: int):
        self.split = split
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenized_data = self._init_tokenized_data(split, data_processor, tokenizer, max_seq_len)

    def __len__(self):
        return len(self.tokenized_data["domain_label_id"])

    def __getitem__(self, index):
        return {
            k: v[index]
            for k, v in self.tokenized_data.items()
        }

    @staticmethod
    def _init_tokenized_data(split, data_processor, tokenizer, max_seq_len):
        data = data_processor.get_split_data(split)
        sentence_lst = []
        for sentence_pair in data["input_str"]:
            sentence_lst.append(
                 sentence_pair[0] + tokenizer.eos_token  + sentence_pair[1])

        tokenized_data = tokenizer(sentence_lst, is_split_into_words=False,
                                   padding="max_length", truncation=True,
                                   max_length=max_seq_len, return_tensors="pt", return_attention_mask=True)

        tokenized_data["input_str"] = data["input_str"]
        tokenized_data["domain_label"] = data["domain_label"]

        tokenized_data["domain_label_id"] = data["domain_label_id"]
        return tokenized_data
