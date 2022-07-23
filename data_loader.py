import os
import shutil
from typing import Union
import sentencepiece as spm
from hydra.utils import get_original_cwd
from abc import ABCMeta, abstractmethod, ABC

import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:
    def __init__(self,
                 save_path: str,
                 bos_id: int = 0,
                 eos_id: int = 1,
                 unk_id: int = 2,
                 pad_id: int = 3
                 ):
        self.save_path = os.path.join(get_original_cwd(), save_path)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.pad_id = pad_id

    def load_spm_voca(self,
                      src_path: str,
                      trg_path: str,
                      src_vocab_size: int = 8000,
                      trg_vocab_size: int = 8000
                      ) -> Union[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        """
        create or load vocabulary
        Args:
            src_path(str): source data file path
            trg_path(str): target data file path
            src_vocab_size(int): source data vocabulary size
            trg_vocab_size(int): target data vocabulary size

        Returns:
            Union[spm.SentencePieceProcessor, spm.SentencePieceProcessor]: source, target sentencepiece Processor

        """
        src_corpus_prefix = f"spm_src_{src_vocab_size}"
        trg_corpus_prefix = f"spm_trg_{trg_vocab_size}"
        src_path = os.path.join(get_original_cwd(), src_path)
        trg_path = os.path.join(get_original_cwd(), trg_path)

        if not (os.path.isfile(os.path.join(self.save_path, f"{src_corpus_prefix}.model"))
                and os.path.isfile(os.path.join(self.save_path, f"{trg_corpus_prefix}.model"))):
            src_model_train_cmd = f"--input={src_path} --model_prefix={src_corpus_prefix} --vocab_size={src_vocab_size} " \
                                  f"--bos_id={self.bos_id} --eos_id={self.eos_id} " \
                                  f"--unk_id={self.unk_id} --pad_id={self.pad_id}"
            trg_model_train_cmd = f"--input={trg_path} --model_prefix={trg_corpus_prefix} --vocab_size={trg_vocab_size} " \
                                  f"--bos_id={self.bos_id} --eos_id={self.eos_id} " \
                                  f"--unk_id={self.unk_id} --pad_id={self.pad_id}"

            spm.SentencePieceTrainer.Train(src_model_train_cmd)  # Train Source Data
            spm.SentencePieceTrainer.Train(trg_model_train_cmd)  # Train Target Data

            shutil.move(f"{src_corpus_prefix}.model", self.save_path)
            shutil.move(f"{src_corpus_prefix}.vocab", self.save_path)
            shutil.move(f"{trg_corpus_prefix}.model", self.save_path)
            shutil.move(f"{trg_corpus_prefix}.vocab", self.save_path)

        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.load(os.path.join(self.save_path, f"{src_corpus_prefix}.model"))
        trg_sp.load(os.path.join(self.save_path, f"{trg_corpus_prefix}.model"))
        return src_sp, trg_sp


class TranslationDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 src_path: str,
                 trg_path: str,
                 src_vocab,
                 trg_vocab,
                 max_seq_size: int
                 ):
        # src_path = os.path.join(get_original_cwd(), src_path)
        # trg_path = os.path.join(get_original_cwd(), trg_path)
        src_path = os.path.join("/home/ubuntu/Workspace/KoDialect", src_path)
        trg_path = os.path.join("/home/ubuntu/Workspace/KoDialect", trg_path)
        self.x = open(src_path, encoding="utf-8-sig").readlines()
        self.y = open(trg_path, encoding="utf-8-sig").readlines()
        self.src_voca = src_vocab
        self.trg_voca = trg_vocab
        self.max_seq_size = max_seq_size

        self.bos_id = src_vocab["<s>"]
        self.eos_id = src_vocab["</s>"]
        self.pad_id = src_vocab["<pad>"]

    def __len__(self):  # data size를 넘겨주는 파트
        if len(self.x) != len(self.y):
            raise IndexError('not equal x_path, y_path line size')
        return len(self.x)

    @abstractmethod
    def encoder_input_to_vector(self, sentence: str):
        pass

    @abstractmethod
    def decoder_input_to_vector(self, sentence: str):
        pass

    @abstractmethod
    def decoder_output_to_vector(self, sentence: str):
        pass

    @abstractmethod
    def padding(self, idx_list):
        pass


class TransformerDataset(TranslationDataset, ABC):
    def __init__(self, src_path: str, trg_path: str, src_vocab, trg_vocab, max_seq_size: int):
        super().__init__(src_path, trg_path, src_vocab, trg_vocab, max_seq_size)
        pass

    def __getitem__(self, idx):
        encoder_input = self.encoder_input_to_vector(self.x[idx])
        decoder_input = self.decoder_input_to_vector(self.y[idx])
        decoder_output = self.decoder_output_to_vector(self.y[idx])
        return encoder_input, decoder_input, decoder_output

    def encoder_input_to_vector(self, sentence: str):
        idx_list = self.src_voca.EncodeAsIds(sentence)
        idx_list = self.padding(idx_list)
        return torch.tensor(idx_list).to(device)

    def decoder_input_to_vector(self, sentence: str):
        idx_list = self.trg_voca.EncodeAsIds(sentence)  # str -> idx
        idx_list.insert(0, self.bos_id)  # Start Token 삽입
        idx_list = self.padding(idx_list)  # padding 삽입
        return torch.tensor(idx_list).to(device)

    def decoder_output_to_vector(self, sentence: str):
        idx_list = self.trg_voca.EncodeAsIds(sentence)
        idx_list.append(self.eos_id)
        idx_list = self.padding(idx_list)
        return torch.tensor(idx_list).to(device)

    def padding(self, idx_list):
        if len(idx_list) < self.max_seq_size:
            idx_list = idx_list + [self.pad_id for _ in range(self.max_seq_size - len(idx_list))]
        else:
            idx_list = idx_list[:self.max_seq_size]
        return idx_list