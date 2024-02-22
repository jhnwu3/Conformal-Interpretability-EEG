import torch
import torch
import pytorch_lightning as pl
from models import *
from interpret.chefer import *
from models.st_transformer import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.pytorch_lightning import *
from data import *
from uq.conformal import *

# my training hyperparameters
sampling_rate = 200
batch_size = 512
num_workers = 32
lr = 0.01
total_epochs = 50

# my model hyperparameters
emb_size = 256
depth = 4 
dropout = 0.5
num_heads = 8
patch_kernel_length = 11  # cqi = 15 - UNUSED
stride = 11  # cqi = 8 - UNUSED

train_loader, test_loader, val_loader, cal_loader = prepare_TUEV_cal_dataloader(sampling_rate=sampling_rate, batch_size=batch_size, num_workers=num_workers)
   
version = f"TUEV-conf-st_transformer-{lr}-{batch_size}-{sampling_rate}-{num_workers}-{total_epochs}"
logger = TensorBoardLogger(
    save_dir="./",
    version=version,
    name="log",
)
early_stop_callback = EarlyStopping(
    monitor="val_cohen", patience=5, verbose=False, mode="max"
)

# define the model for training - STT transformer
st_transformer = STTransformer(emb_size=emb_size, 
                                depth=depth,
                                n_classes=6, 
                                channel_length=1000,
                                dropout=dropout, 
                                num_heads=num_heads,
                                kernel_size=11, 
                                stride=11,
                                kernel_size2=5,
                                stride2=5)

lightning_model = LitModel(st_transformer)
trainer = pl.Trainer(
    devices=[0],
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=False),
    benchmark=True,
    enable_checkpointing=True,
    logger=logger,
    max_epochs=total_epochs,
    callbacks=[early_stop_callback],
)

trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# test the model
pretrain_result = trainer.test(
    model=lightning_model, ckpt_path="best", dataloaders=test_loader
)[0]

print(pretrain_result)

# save the model
torch.save(lightning_model.model.state_dict(), f"saved_weights/st_transformer_conformal_tuev.pt")

# calibration step after validation and what not.


# train model with pytlightning
