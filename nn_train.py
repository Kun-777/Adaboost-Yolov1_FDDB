from nn_dataloader import FDDBDataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from nn_model import Yolov1
import pytorch_lightning as pl
import tensorboard

if __name__ == "__main__":
    batch_size = 8
    img_size = 448
    S = 7
    B = 2
    C = 0
    model_checkpoint_cb = ModelCheckpoint(dirpath="./model/new/", filename="{epoch}-{val_loss:.5f}", save_top_k=-1, mode="min", verbose=True)
    model = Yolov1(1, S, B, C)
    dl = FDDBDataloader("./", batch_size, 4, img_size, S, B, C)
    trainer = pl.Trainer(gpus=1, 
                        log_every_n_steps=10,
                        max_epochs=50,
                        callbacks=[model_checkpoint_cb],
                        num_sanity_val_steps=0,
                        logger=True,
                        )
    trainer.fit(model, dl)