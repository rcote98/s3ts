"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
@version 2022-12
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from s3ts.data_str import AugProbabilities, TaskParameters
from s3ts.data_aux import download_dataset
from s3ts.network import MultitaskModel
from s3ts.data import MTaskDataModule

import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

EXPERIMENT = "multitask_GP"
DATASET = "GunPoint"

TEST_SIZE = 0.8
STS_LENGTH = None

WINDOW_SIZE = 10
BATCH_SIZE  = 128
LEARNING_RATE = 1E-5

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

probs = AugProbabilities()
tasks = TaskParameters(
    # main task
    main=True,
    main_weight=1,
    
    discrete_intervals=5,

    disc=True,
    disc_weight=1,
    
    pred=True,
    pred_time=None,
    pred_weight=1,
    
    areg_ts=True,
    areg_ts_weight=1,

    areg_img=False,
    areg_img_weight=1,
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Downloading data...")
start_time = time.perf_counter()
X, Y, mapping = download_dataset(DATASET)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Computing dataset...")
start_time = time.perf_counter()
dm = MTaskDataModule(
    experiment=EXPERIMENT,
    X=X, Y=Y,
    sts_length=STS_LENGTH,
    window_size=WINDOW_SIZE,
    tasks=tasks,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Creating model...", end="")
start_time = time.perf_counter()
model = MultitaskModel(
    n_labels=dm.n_labels,
    n_patterns=dm.n_patterns,
    patt_length=dm.sample_length,
    window_size=WINDOW_SIZE,
    tasks=tasks,
    max_feature_maps=128,
    learning_rate=LEARNING_RATE)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Setup the trainer...")
start_time = time.perf_counter()
early_stop = EarlyStopping(monitor="val_auroc", mode="max", patience=5)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_checkpoint = ModelCheckpoint(dm.exp_path, save_last=True)
trainer = Trainer(default_root_dir=dm.exp_path,
    callbacks=[lr_monitor, model_checkpoint, early_stop],
    max_epochs=200, check_val_every_n_epoch=1,
    deterministic = True)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Begin training...")
trainer.fit(model, datamodule=dm)
trainer.validate(model, datamodule=dm)
trainer.test(model, datamodule=dm)