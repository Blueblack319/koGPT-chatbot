import argparse
import logging
import os

from dataset import ChatbotDataset, koGPT2_TOKENIZER
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pandas as pd

parser = argparse.ArgumentParser(description="Simsimi based on KoGPT-2")

parser.add_argument(
    "--chat",
    action="store_true",
    default=False,
    help="response generation on given user input",
)

parser.add_argument(
    "--sentiment",
    type=str,
    default="0",
    help="sentiment for system. 0 is neutral, 1 is negative, 2 is positive.",
)

parser.add_argument(
    "--model_params",
    type=str,
    default="checkpoint/last.ckpt",
    help="model binary for starting chat",
)

parser.add_argument("--train", action="store_true", default=False, help="for training")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


U_TKN = "<usr>"
S_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
MASK = "<unused0>"
SENT = "<unused1>"
PAD = "<pad>"


class KoGPTChatbot(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPTChatbot, self).__init__()
        self.save_hyperparameters(hparams)
        self.neg = -1e18
        self.model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--max-len",
            type=int,
            default=32,
            help="max sentence length on input (default: 32)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=96,
            help="batch size for training (default: 96)",
        )
        parser.add_argument(
            "--lr", type=float, default=5e-5, help="The initial learning rate"
        )
        parser.add_argument(
            "--warmup-ratio", type=float, default=0.1, help="warmup ratio"
        )

        return parser

    def forward(self, inputs):
        # inputs: (batch_size, input_size)
        # Q. return_dict=False랑 비교해보기
        # A. return_dict=False => return tuple
        #    return_dict=True => return ModelOutput
        output = self.model(inputs, return_dict=True)  
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)  # Q. self가 무슨 function인지 확인하기. sup: 아마도 forward? A: Correct! self() == forward
        # Q. mask의 dimension 확인, repeat_interleave로 뭘 만들고 싶어하는지 확인하기.
        # A. mask dim: (96, 32) -> mask_3d dim: (96, 32, 51200) => 51200은 tokenizer의 embedding dim
        # out: (96, 32, 51200)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[-1], dim=-1)
        # Q. 아마도 mask_3d가 1이면 out 아니면 0이 무슨 뜻?
        # A. mask: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, ...] * 96
        #    mask_3d: [[0]*51200, [0]*51200, [0]*51200, [0]*51200, [1]*51200, [1]*51200, [1]*51200, [1]*51200, [1]*51200, [0]*51200 [0]*51200, [0]*51200, [0]*51200, ...] * 96
        #    mask_out: [[-]*51200, [-]*51200, [-]*51200, [-]*51200, [out]*51200, [out]*51200, [out]*51200, [out]*51200, [out]*51200, [-]*51200 [-]*51200, [-]*51200, [-]*51200, ...] * 96
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        # label: (batch size=96, input size=32)
        loss = self.loss_func(
            mask_out.transpose(2, 1), label
        )  
        loss_avg = loss.sum() / mask.sum()
        self.log("train_loss", loss_avg)
        return loss_avg

    def prepare_data(self):
        data = pd.read_csv(f"{os.getcwd()}/chatbot_data/ChatBotData.csv")
        self.train_set = ChatbotDataset(data, max_len=self.hparams.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def configure_optimizers(self):
        # Prepare optimizer
        # Q. self.named_parameters() 확인
        # A. nn.Linear 등으로 정의한 model의 parameters
        param_optimizer = list(self.named_parameters())
        # Q. no decay? 이게 뭐야?
        # A. Layer name이 no_decay에 포함되어 있는 layer라면 decay False
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )

        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "cosine_schedule_with_warmup",
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        # Q. batch의 item은 python list?
        # A. batch: (batch size, 3(tuple))
        # tuple: [input, mask, input_masked]
        breakpoint()
        print(f"Batch type: {type(batch)}")
        print(f"Batch : {batch}")
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]

        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def chat(self, sent="0"):
        tokenizer = koGPT2_TOKENIZER
        sent_tokens = tokenizer.tokenize(sent)
        print(sent_tokens)
        print(sent)
        with torch.no_grad():
            while True:
                q = input("user > ").strip()
                if q == "quit":
                    break
                a = ""
                while True:
                    input_ids = torch.LongTensor(
                        tokenizer.encode(U_TKN + q + SENT + sent + S_TKN + a)
                    ).unsqueeze(dim=0)
                    pred = self(input_ids)
                    # Q. argmax에 대해 공부
                    # A. max value의 index를 반환
                    
                    gen = tokenizer.convert_ids_to_tokens(
                        torch.argmax(pred, dim=-1).squeeze().numpy().tolist()
                    )[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", ' ')
                    print(gen)
                print(f"Bot > {a.strip()}")
                
parser = KoGPTChatbot.add_model_specific_args(parser)
# Q. add_argparse_args VS from_argparse_args 
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoint",
            filename="{epoch:02d}-{train_loss:.2f}",
            verbose=True,
            save_last=True,
            monitor="train_loss",
            mode="min",
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        model = KoGPTChatbot(args)
        model.train()
        
        trainer = Trainer.from_argparse_args(
            args, callbacks=[checkpoint_callback], gradient_clip_val=1.0
        )
        trainer.fit(model)
        logging.info(f"best model path: {checkpoint_callback.best_model_path}")

    if args.chat:
        model = KoGPTChatbot.load_from_checkpoint(args.model_params)
        model.chat()
