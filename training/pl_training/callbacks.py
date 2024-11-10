from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint


def get_model_checkpoint_callback(dirpath):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_epoch/loss",
        mode="min",
        save_top_k=2,
        dirpath=dirpath,
        filename="best_model",
    )
    return checkpoint_callback


def get_early_stopping_callback():
    early_stopping_callback = EarlyStopping(monitor="val_epoch/loss")
    return early_stopping_callback


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training completed!")
