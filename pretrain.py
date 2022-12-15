"""
Run the main classification task alongside in two scenarios: 
alone and with shifted discrete label pretrains.

@author Raúl Coterillo
@version 2022-12
"""

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything

from sklearn.model_selection import train_test_split

from s3ts.data_str import AugProbabilities, TaskParameters
from s3ts.data_aux import download_dataset
from s3ts.network import MultitaskModel
from s3ts.data import MTaskDataModule

import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

EXPERIMENT = "pretrain_GP"
DATASET = "GunPoint"

USE_PRETRAIN = True

PRET_SIZE = 0.8
TEST_SIZE = 0.5
PRET_STS_LENGTH = 100 
MAIN_STS_LENGTH = None

WINDOW_SIZE = 10
BATCH_SIZE  = 128
LEARNING_RATE = 1E-5
MAX_FEATURE_MAPS = 128

RANDOM_STATE = 0
seed_everything(RANDOM_STATE)

# does not do anything as of yet
probs = AugProbabilities()

# pretrain tasks
aux_task = TaskParameters(
    # main task
    main=False,
    main_weight=1,
    # discretization intervals
    discrete_intervals=5,
    # discrete classification
    disc=True,
    disc_weight=1,
    # discrete prediction
    pred=True,
    pred_time=None,
    pred_weight=1,
    # time series regression
    areg_ts=False,
    areg_ts_weight=1,
    # similarity frame regression
    areg_img=False,
    areg_img_weight=1,
    )

# main task
main_task = TaskParameters(main=True,
    disc=False, pred=False, areg_ts=False, areg_img=False,
    ) # suppress all other tasks

# DATA ACQUISITION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Downloading data...")
start_time = time.perf_counter()
X, Y, mapping = download_dataset(DATASET)
n_samples = X.shape[0]
X_main, X_pret, Y_main, Y_pret = train_test_split(X, Y, 
    test_size=PRET_SIZE, stratify=Y, random_state=RANDOM_STATE,)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

if USE_PRETRAIN:
    print("Computing dataset for pretrain...")
    start_time = time.perf_counter()
    pretrain_dm = MTaskDataModule(
        experiment=EXPERIMENT / "pretrain",
        X=X_pret, Y=Y_pret,
        sts_length=PRET_STS_LENGTH,
        window_size=WINDOW_SIZE,
        tasks=aux_task,
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE)
    end_time = time.perf_counter()
    print("DONE! ", end_time - start_time, "seconds")

print("Computing dataset for main task...")
start_time = time.perf_counter()
main_dm = MTaskDataModule(
    experiment=EXPERIMENT / "main",
    X=X_main, Y=Y_main,
    sts_length=MAIN_STS_LENGTH,
    window_size=WINDOW_SIZE,
    tasks=main_task,
    batch_size=BATCH_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE)
end_time = time.perf_counter()
print("DONE! ", end_time - start_time, "seconds")

# PRETRAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
if USE_PRETRAIN:
    print("Creating pretrain model...", end="")
    start_time = time.perf_counter()
    pretrain_model = MultitaskModel(
        n_labels=pretrain_dm.n_labels,
        n_patterns=pretrain_dm.n_patterns,
        patt_length=pretrain_dm.sample_length,
        window_size=WINDOW_SIZE,
        tasks=aux_task,
        max_feature_maps=MAX_FEATURE_MAPS,
        learning_rate=LEARNING_RATE)
    end_time = time.perf_counter()
    print(end_time - start_time, "seconds")

    print("Setup the pretrain trainer...")
    start_time = time.perf_counter()
    early_stop = EarlyStopping(monitor="val_auroc", mode="max", patience=5)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(pretrain_dm.exp_path, save_last=True)
    trainer = Trainer(default_root_dir=pretrain_dm.exp_path,
        callbacks=[lr_monitor, model_checkpoint, early_stop],
        max_epochs=100, check_val_every_n_epoch=1,
        deterministic = True)
    end_time = time.perf_counter()
    print(end_time - start_time, "seconds")

    print("Begin pretrain training...")
    trainer.fit(pretrain_model, datamodule=pretrain_dm)
    trainer.validate(pretrain_model, datamodule=pretrain_dm)
    trainer.test(pretrain_model, datamodule=pretrain_dm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

print("Creating main task model...", end="")
start_time = time.perf_counter()
main_model = MultitaskModel(
    n_labels=main_dm.n_labels,
    n_patterns=main_dm.n_patterns,
    patt_length=main_dm.sample_length,
    window_size=WINDOW_SIZE,
    tasks=main_task,
    max_feature_maps=MAX_FEATURE_MAPS,
    learning_rate=LEARNING_RATE)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

if USE_PRETRAIN:
    # copy the decoder from the pretraing
    main_model.conv_encoder = pretrain_model.conv_encoder

print("Setup the main trainer...")
start_time = time.perf_counter()
early_stop = EarlyStopping(monitor="val_auroc", mode="max", patience=5)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_checkpoint = ModelCheckpoint(main_dm.exp_path, save_last=True)
trainer = Trainer(default_root_dir=main_dm.exp_path,
    callbacks=[lr_monitor, model_checkpoint, early_stop],
    max_epochs=100, check_val_every_n_epoch=1,
    deterministic = True)
end_time = time.perf_counter()
print(end_time - start_time, "seconds")

print("Begin main training...")
trainer.fit(pretrain_model, datamodule=pretrain_dm)
trainer.validate(pretrain_model, datamodule=pretrain_dm)
trainer.test(pretrain_model, datamodule=pretrain_dm)