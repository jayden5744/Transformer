# https://github.com/PyTorchLightning/lightning-transformers
import os
import logging
from typing import Dict

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.trainer import TransformerPL
from src.utils.bleu import bleu_score

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_logger():
    """
    날짜, 시간을 logging 해주는 함수
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def make_config(cfg: DictConfig) -> Dict:
    """

    Args:
        cfg(DictConfig):

    Returns:
        Dict :

    """
    result = {"src_name": cfg.data.src_name, "trg_name": cfg.data.trg_name,
              "model_path": cfg.data.model_path}
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))

    return result


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    model = TransformerPL(cfg)
    init_logger()
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(get_original_cwd(), "./SavedModel/"),
                                          filename=cfg.data.file_name,
                                          save_top_k=True,
                                          verbose=True,
                                          monitor="val_loss",
                                          mode="min")

    wandb_logger = WandbLogger(name=cfg.data.file_name, project=cfg.data.project_name)
    config = make_config(cfg)
    wandb_logger.log_hyperparams(config)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=cfg.trainer.early_stopping,
                                        verbose=False, mode='min')
    # trainer = Trainer(gpus=1)
    trainer = Trainer(gpus=[0, 1], accelerator="ddp", callbacks=[early_stop_callback, checkpoint_callback],
                      logger=wandb_logger,
                      checkpoint_callback=True, gradient_clip_val=1, max_epochs=cfg.trainer.epochs)
    trainer.fit(model)


@hydra.main(config_path="conf", config_name="config")
def inference(cfg: DictConfig) -> None:
    src_test_data = []
    trg_test_data = []
    src_path = os.path.join(get_original_cwd(), cfg.data.src_test_path)
    trg_path = os.path.join(get_original_cwd(), cfg.data.trg_test_path)
    with open(src_path, "r", encoding="utf-8-sig") as f:
        for i in f.readlines():
            src_test_data.append(i.strip())
    with open(trg_path, "r", encoding="utf-8-sig") as f:
        for i in f.readlines():
            trg_test_data.append(i.strip())
    model_path = os.path.join(get_original_cwd(), cfg.data.model_path)
    model = TransformerPL.load_from_checkpoint(model_path, cfg=cfg)
    total_bleu = 0
    f = open(os.path.join(get_original_cwd(), cfg.data.result_path), "w", encoding="utf-8-sig")
    for input, target in zip(src_test_data, trg_test_data):
        predict = model.translate(input)
        bleu = bleu_score(predict, target)
        total_bleu += bleu
        f.write("=" * 30 + "\n")
        f.write(f"Input : {input}" + "\n")
        f.write(f"Predict : {predict}" + "\n")
        f.write(f"Target : {target}" + "\n")
        f.write(f"BLEU : {bleu}" + "\n")
        print("=" * 30)
        print(f"Input : {input}")
        print(f"Predict : {predict}")
        print(f"Target : {target}")
        print(f"BLEU : {bleu}")
    f.write(f"Average BLE : {total_bleu / len(src_test_data)}" + "\n")
    print(f"Average BLEU : {total_bleu / len(src_test_data)}")
    f.close()


if __name__ == '__main__':
    train()
    # inference()
