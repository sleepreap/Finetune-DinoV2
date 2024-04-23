import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from DinoV2model import Dinov2Finetuner
from DinoV2.dataset import SegmentationDataModule
import DinoV2.config as config


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=config.DATASET_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    model=Dinov2Finetuner(config.ID2LABEL, config.LEARNING_RATE)

    trainer = pl.Trainer(
        logger=config.LOGGER,
        precision=config.PRECISION,
        accelerator='cuda',
        devices=config.DEVICES,
        callbacks=[config.EARLY_STOPPING_CALLBACK, config.CHECKPOINT_CALLBACK],
        max_epochs=config.EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("dinov2.ckpt")
