from qpi_data_model import QPIDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import pytorch_lightning as pl
from lista_model import LISTA


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    qpi_data_module = QPIDataModule()

    # ------------
    # model
    # ------------
    lista_model = LISTA()

    # ------------
    # training
    # ------------

    # Logger: 1st value is the directory name, followed by the name of the run
    logger = TensorBoardLogger(save_dir='classifier_logs', name='linear_log')

    # Stopping conditions:
    early_stop_callback = EarlyStopping(
        monitor="Validation loss",
        stopping_threshold=1,
        min_delta=0,
        patience=5,
        verbose=False,
        mode="min",
        check_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        auto_lr_find=True,
        accelerator='auto',
        auto_select_gpus=True,
        max_epochs=3000,
        logger=logger,
        gpus=-1,
        num_nodes=-1,
        callbacks=[early_stop_callback]
    )
    trainer.tune(lista_model, qpi_data_module)
    trainer.fit(lista_model, qpi_data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=qpi_data_module)
    print(result)


if __name__ == '__main__':
    cli_main()
