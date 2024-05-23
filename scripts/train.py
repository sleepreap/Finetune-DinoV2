import pytorch_lightning as pl
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")
from DinoV2 import ( Dinov2Finetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE, 
                        LOGGER, 
                        DEVICES, 
                        CHECKPOINT_CALLBACK, 
                        EPOCHS )
from pytorch_lightning.strategies import DDPStrategy


if __name__=="__main__":
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model=Dinov2Finetuner(ID2LABEL, LEARNING_RATE)

    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator='cuda',
        devices=DEVICES,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("DinoV2.ckpt")
