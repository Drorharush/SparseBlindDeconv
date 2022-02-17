import torch
from torch import nn
import pytorch_lightning as pl
import model_tools


class LISTA(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, layer_num=5, iter_num=10):
        super().__init__()
        self.learning_rate = learning_rate
        self.iter_num = iter_num  # Number of times the data goes through the layers.
        self.layer_num = layer_num  # Number of layers
        pad = (2, 2)  # These values make sure the image dimensions stay the same
        ker_same = (5, 5)
        self.loss_function = model_tools.ActivationLoss()
        self.x_layers = nn.ModuleList()
        self.y_layers = nn.ModuleList()
        self.slu_layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.x_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
            self.y_layers.append(nn.Conv2d(1, 1, kernel_size=ker_same, padding=pad))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = model_tools.normalize_tensor_0to1(x_in)
        y = torch.clone(x)
        for _ in range(self.iter_num):
            for layer_idx in range(self.layer_num):
                x = self.relu(self.x_layers[layer_idx](x) + self.y_layers[layer_idx](y))
            x = model_tools.normalize_tensor_sumto1(x)
        return x

    def training_step(self, batch, batch_idx):
        eval_dict = self._shared_pred_eval(batch)
        self.log('non-zero diff (%)', round(100 * eval_dict['non zero diff'], 0), logger=False, prog_bar=True)
        return eval_dict

    def validation_step(self, batch, batch_idx):
        return self._shared_pred_eval(batch)

    def _shared_pred_eval(self, batch) -> dict:
        measurement, kernel, activation = batch
        pred_activation = self(measurement)
        loss = self.loss_function(pred_activation, activation, kernel)
        pred_non_zero = torch.count_nonzero(pred_activation, dim=(-2, -1))
        target_non_zero = torch.count_nonzero(activation, dim=(-2, -1))
        non_zero_diff = pred_non_zero - target_non_zero
        return {'loss': loss, 'non zero diff': non_zero_diff / target_non_zero}

    def training_epoch_end(self, outputs) -> None:
        self._shared_logging(outputs, 'Training')

    def validation_epoch_end(self, outputs) -> None:
        self._shared_logging(outputs, 'Validation')

        # The metric by which we stop has to be logged differently (apparently)
        epoch_loss = torch.stack([log['loss'] for log in outputs]).median()
        self.log('Validation loss', epoch_loss, on_epoch=True, prog_bar=True, logger=False)

    def _shared_logging(self, outputs, prefix):
        # Calculating matrices over epoch
        epoch_loss = torch.stack([log['loss'] for log in outputs]).median()
        loss_std = torch.std(torch.stack([log['loss'] for log in outputs]), dim=0)
        epoch_non_zero_diff = torch.stack([log['non zero diff'] for log in outputs]).median()

        self.logger.experiment.add_scalar(f'{prefix} loss', epoch_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix} loss std', loss_std, self.current_epoch)
        self.logger.experiment.add_scalar(f'{prefix} non-zero diff', epoch_non_zero_diff, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

