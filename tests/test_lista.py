from pytorch_lightning import Trainer, seed_everything
from project.lista_model import LISTA
from project.qpi_data_model import QPIDataModule


def test_lista():
    seed_everything(1234)

    model = LISTA()
    data_module = QPIDataModule()
    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, train, val)

    results = trainer.test(test_dataloaders=test)
    assert results[0]['test_acc'] > 0.7
