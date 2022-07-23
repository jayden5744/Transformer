import math
import logging
import sentencepiece as spm

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from src.transformer import Transformer
from src.utils.loss import CrossEntropyLoss
from src.utils.weight_initialize import xavier_uniform_initialize, xavier_normal_initialize, \
    he_uniform_initialize, he_normal_initialize
from data_loader import TransformerDataset, Vocabulary


logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def select_weight_initialize_method(
        method: str,
        distribution: str,
        model: Transformer
) -> None:
    """
    Initialize weight method
        - weight initialization of choice [he, xavier]
        - weight distribution of choice [uniform, normal]
    Args:
        method(str): weight initialization method
        distribution: weight distribution
        model: Transformer Model

    """
    if method == "xavier" and distribution == "uniform":
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            model.apply(xavier_uniform_initialize)

    elif method == "xavier" and distribution == "normal":
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            model.apply(xavier_normal_initialize)

    elif method == "he" and distribution == "uniform":
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            model.apply(he_uniform_initialize)

    elif method == "he" and distribution == "normal":
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            model.apply(he_normal_initialize)

    else:
        raise ValueError("weight initialization of choice [he, xavier] and "
                         "Weight distribution of choice [uniform, normal]")


class TransformerPL(pl.LightningModule):
    def __init__(self, cfg):
        super(TransformerPL, self).__init__()
        self.hp_data = cfg.data
        self.hp_model = cfg.model
        self.hp_trainer = cfg.trainer
        self.src_voca, self.trg_voca = self.get_vocabulary()
        self.model = Transformer(cfg)
        select_weight_initialize_method(self.hp_trainer.weight_init, self.hp_trainer.weight_distribution, self.model)
        self.criterion = CrossEntropyLoss(ignore_index=self.hp_data.pad_id,
                                          smooth_eps=self.hp_trainer.label_smoothing_value,
                                          from_logits=False)

    def forward(self, enc_inputs, dec_inputs):
        outputs = self.model(enc_inputs, dec_inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, ppl = self._shared_eval_step(batch, batch_idx)
        metrics = {"loss": loss, "ppl": ppl}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss, ppl = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss, "val_ppl": ppl}
        self.log_dict(metrics)
        return metrics, batch

    def training_epoch_end(self, training_step_outputs):
        matrix = training_step_outputs[-1]
        loss = round(matrix['loss'].item(), 4)
        ppl = round(matrix['ppl'], 4)
        if self.global_rank == 0:
            print("")
            print("=" * 30)
            logger.info(f"[Train] loss : {loss}  Perplexity : {ppl}")

    def validation_epoch_end(self, validation_step_outputs):
        metrics, batch = validation_step_outputs[0][0], validation_step_outputs[0][1]
        loss = round(metrics['val_loss'].item(), 4)
        ppl = round(metrics['val_ppl'], 4)

        src_input, _, tar_output = batch

        input_ids = torch.tensor([src_input[0].cpu().numpy()]).type_as(src_input.data)

        tar_output = torch.tensor([tar_output[0].cpu().numpy()]).to(device)
        enc_outputs, _ = self.model.encode(input_ids)

        target_ids = torch.zeros(1, self.hp_model.max_sequence_len).type_as(src_input.data)
        next_token = self.trg_voca.bos_id()

        for i in range(self.hp_model.max_sequence_len):
            target_ids[0][i] = next_token
            decoder_output = self.model.decode(target_ids, input_ids, enc_outputs)
            prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_token = prob.data[i].item()
            if next_token == self.trg_voca.eos_id():
                break
        input_sentence = self.src_voca.DecodeIds(input_ids[0].tolist())
        output_sentence = self.trg_voca.DecodeIds(target_ids[0].tolist())
        target_sentence = self.trg_voca.DecodeIds(tar_output[0].tolist())
        print("")
        print("=" * 30)
        logger.info(f"Source sentence : {input_sentence}")
        logger.info(f"Predict sentence : {output_sentence}")
        logger.info(f"Target sentence : {target_sentence}")

        if self.global_rank == 0:
            print("")
            print("=" * 30)
            logger.info(f"[Val] loss : {loss}   Perplexity : {ppl}")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        seed_val = 42
        torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        train_dataset = TransformerDataset(self.hp_data.src_train_path, self.hp_data.trg_train_path,
                                           self.src_voca, self.trg_voca, self.hp_model.max_sequence_len)
        point_sampler = torch.utils.data.RandomSampler(train_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        train_loader = DataLoader(train_dataset, batch_size=self.hp_trainer.batch_size, sampler=point_sampler)

        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        seed_val = 42
        torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        val_dataset = TransformerDataset(self.hp_data.src_val_path, self.hp_data.trg_val_path,
                                         self.src_voca, self.trg_voca, self.hp_model.max_sequence_len)
        point_sampler = torch.utils.data.RandomSampler(val_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        val_loader = DataLoader(val_dataset, batch_size=self.hp_trainer.batch_size, sampler=point_sampler)

        return val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        seed_val = 42
        torch.manual_seed(seed_val)
        # path를 불러와서 train_loader를 만드는 함수
        test_dataset = TransformerDataset(self.hp_data.src_test_path, self.hp_data.trg_test_path,
                                         self.src_voca, self.trg_voca, self.hp_model.max_sequence_len)
        point_sampler = torch.utils.data.RandomSampler(test_dataset)  # data의 index를 반환하는 함수, suffle를 위한 함수
        # dataset을 인자로 받아 data를 뽑아냄
        test_loader = DataLoader(test_dataset, batch_size=self.hp_trainer.batch_size, sampler=point_sampler)

        return test_loader

    def get_vocabulary(self) -> Union[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        vocab = Vocabulary(self.hp_data.dictionary_path)
        src_vocab, trg_vocab = vocab.load_spm_voca(self.hp_data.src_train_path, self.hp_data.trg_train_path,
                                                   self.hp_model.enc_vocab_size, self.hp_model.dec_vocab_size)
        return src_vocab, trg_vocab

    def _shared_eval_step(self, batch, batch_idx):
        src_input, tar_input, tar_output = batch
        y_hat = self.model(src_input, tar_input)
        predict = y_hat.contiguous().view(-1, y_hat.size(-1))
        tar_output = tar_output.contiguous().view(-1)
        loss = self.criterion(predict, tar_output)
        ppl = math.exp(loss)
        return loss, ppl

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp_trainer.learning_rate,
                                betas=(self.hp_trainer.optimizer_b1, self.hp_trainer.optimizer_b1),
                                eps=self.hp_trainer.optimizer_e)

    def optimizer_step(self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
    ):
        d_model = self.hp_model.d_embedding
        step_num = self.trainer.global_step
        warm_up_step = self.hp_trainer.warmup_steps
        if step_num != 0:
            lr = d_model ** (-0.5) * min(step_num ** (-0.5), step_num * (warm_up_step ** (-1.5)))
            lr = self.hp_trainer.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            self.log_dict({"learning rate": lr})
        optimizer.step(closure=optimizer_closure)

    def translate(self, input_sentence):
        self.eval()
        input_ids = self.src_voca.EncodeAsIds(input_sentence)
        if len(input_ids) <= self.hp_model.max_sequence_len:
            input_ids = input_ids + [self.hp_data.pad_id] * (self.hp_model.max_sequence_len - len(input_ids))
        else:
            input_ids = input_ids[:self.hp_model.max_sequence_len]
        input_ids = torch.tensor([input_ids]).to(device)
        enc_outputs, _ = self.model.encode(input_ids)
        target_ids = torch.zeros(1, self.hp_model.max_sequence_len).type_as(input_ids.data)
        next_token = self.trg_voca.bos_id()

        for i in range(self.hp_model.max_sequence_len):
            target_ids[0][i] = next_token
            decoder_output = self.model.decode(target_ids, input_ids, enc_outputs)
            prob = decoder_output.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_token = prob.data[i].item()
            if next_token == self.trg_voca.eos_id():
                break
        output_sentence = self.trg_voca.DecodeIds(target_ids[0].tolist())
        return output_sentence