import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

#import esm
from src.model.bert import BERT

from src.data.dataset import MyDataModule
from src.model.config import Config_100M


################ pytorch lightning model ######################
class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.config = Config_100M

        self.model = BERT(self.config)

        self.learning_rate = self.config['learning_rate']
        self.weight_decay = self.config['weight_decay']
        self.beta1 = self.config['beta1']
        self.beta2 = self.config['beta2']
        
        # metrics and loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=1) # ignore pad token in the loss function

   
    def forward(self, inputs):
        output = self.model(inputs['input_ids'], inputs['attention_mask']) 
        return output.logits

    def training_step(self, batch):
        """This function will be called for each batch during training.
        It will compute the loss and log it.
        """
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
 
    def validation_step(self, batch):
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss # this loss is not used, but I could return something else and modify.
    
    def test_step(self, batch):
        logits = self(batch)
        targets = batch['labels']
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        perplexity = torch.exp(loss)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_perplexity", perplexity, prog_bar=True, logger=True, sync_dist=True)
        return loss, perplexity


    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps) # 10% of full train rain all steps * epochs

        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=self.learning_rate*0.1)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])  # milestone is when to switch from warm up to decay.
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }


def main(args):

    print("Initializing model and data module...")
    model      = LitModel(args)
    datamodule = MyDataModule(args)

    logger = CSVLogger(save_dir="logs/", name=args.output)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=0.001)
    checkpoint = ModelCheckpoint(
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        accelerator="gpu",
        strategy="ddp",
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        accumulate_grad_batches=5,
        callbacks=[early_stop, checkpoint],
    )
    

    
    ########################### Training ##########################    
    if args.resume:
        print(f"Resuming training from {args.checkpoint_resume}!")
        trainer.fit(model, datamodule, ckpt_path=args.checkpoint_resume)

    else:
        print("Training from scratch!")
        trainer.fit(model, datamodule)
    
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--dataDir', type=str, default='data/uniref90/uniref90_stage1/')
    #parser.add_argument('-i', '--dataDir', type=str, default='../ViCAM/data/RVDB/processed/CRVDBv30prot_maxlen1600_20aa/')
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_resume', type=str)
    args = parser.parse_args()
    main(args)



###### RUNNING EXAMPLES ######  
# python src/train/train.py -o PLEM/stage1
# python src/train/train.py -o PLEM/stage1 --resume --checkpoint_resume checkpoints/PLEM/stage1/epoch=5-val_loss=1.44.ckpt