from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
from src.loss import loss
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import DataLoader, Dataset

EPS = 1e-9


def labels_to_window_mask(
    labels_batch: torch.Tensor, reg_mode: str, half_wnd: int = None
):
    """Utility for std regularization."""

    assert reg_mode in ["window", "right_part"], f"Unknown mode {reg_mode}."

    if reg_mode == "window":
        assert half_wnd is not None, "Specify half window size."

    cp_indexes = labels_batch.argmax(dim=1)
    abnormal_wnd_mask = torch.zeros_like(labels_batch)

    batch_size, seq_len = abnormal_wnd_mask.shape

    for i in range(batch_size):
        row_mask = torch.zeros(seq_len)
        if cp_indexes[i] > 0:
            if reg_mode == "window":
                wnd_start = max(cp_indexes[i] - half_wnd, 0)
                wnd_end = min(cp_indexes[i] + half_wnd, seq_len)
            else:
                wnd_start = cp_indexes[i]
                wnd_end = seq_len
            # abnormal std window mask
            row_mask[wnd_start : wnd_end + 1] = 1
            abnormal_wnd_mask[i] = row_mask
    return abnormal_wnd_mask


################################################################################################
#                                         TOTAL BAYES                                          #
################################################################################################


class BayesLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout: float = 0.0,
        n_layers: int = 1,
        batch_first=True,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

        self.linear_w = nn.ModuleList(
            [
                bnn.BayesLinear(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_features=input_dim,
                    out_features=hidden_dim,
                )
                for _ in range(4)
            ]
        )

        self.linear_u = nn.ModuleList(
            [
                bnn.BayesLinear(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
                for _ in range(4)
            ]
        )

        self.dropout = dropout
        self.n_layers = n_layers

    def forward(self, inputs, init_states=None):
        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(N, self.hidden_dim).to(inputs)
            c_prev = torch.zeros(N, self.hidden_dim).to(inputs)
        else:
            h_prev, c_prev = init_states

        for x_t in inputs:
            x_f, x_i, x_o, x_c_hat = [linear(x_t) for linear in self.linear_w]
            h_f, h_i, h_o, h_c_hat = [linear(h_prev) for linear in self.linear_u]

            f_t = torch.sigmoid(x_f + h_f)
            i_t = torch.sigmoid(x_i + h_i)
            o_t = torch.sigmoid(x_o + h_o)
            c_t_hat = torch.tanh(x_c_hat + h_c_hat)

            c_prev = torch.mul(f_t, c_prev) + torch.mul(i_t, c_t_hat)
            h_prev = torch.mul(o_t, torch.tanh(c_prev))

            if self.dropout > 0:
                F.dropout(h_prev, p=self.dropout, training=self.training, inplace=True)

            outputs.append(
                h_prev
            )  # .clone().detach()) # NOTE fail with detach. check it

        outputs = torch.stack(outputs, dim=0)

        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, (h_prev, c_prev)


class BaseBayesRnn(nn.Module):
    """LSTM-based network for experiments with Synthetic Normal data and Human Activity."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        drop_prob: float,
        prior_mu: float,
        prior_sigma: float,
    ) -> None:
        """Initialize model's parameters.

        :param input_size: size of elements in input sequence
        :param output_size: length of the generated sequence
        :param hidden_dim: size of the hidden layer(-s)
        :param n_layers: number of recurrent layers
        :param drop_prob: dropout probability
        """
        super().__init__()

        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = BayesLSTM(
            input_size,
            hidden_dim,
            dropout=drop_prob,
            n_layers=n_layers,
            batch_first=True,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        self.linear = bnn.BayesLinear(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_features=hidden_dim,
            out_features=1,
        )

        self.activation = nn.Sigmoid()

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param input_seq: batch of generated sunthetic normal sequences
        :return: probabilities of changes for each sequence
        """
        batch_size = input_seq.size(0)
        lstm_out, hidden = self.lstm(input_seq.float())
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(lstm_out)
        out = self.activation(out)
        out = out.view(batch_size, -1)
        return out


class MnistBayesRNN(nn.Module):
    """Recurrent network for MNIST experiments."""

    def __init__(
        self,
        input_size: int,
        hidden_rnn: int,
        rnn_n_layers: int,
        linear_dims: List[int],
        rnn_dropout: float = 0.0,
        dropout: float = 0.5,
        rnn_type: str = "LSTM",
        prior_mu: float = 0.0,
        prior_sigma: float = 0.1,
    ) -> None:
        """Initialize model's parameters.

        :param input_size: number of input features
        :param hidden_rnn: size of recurrent model's hidden layer
        :param rnn_n_layers: number of recurrent layers
        :param linear_dims: list of dimensions for linear layers
        :param rnn_dropout: dropout in recurrent block
        :param dropout: dropout in fully-connected block
        :param rnn_type: type of recurrent block (LSTM, GRU, RNN)
        """
        super().__init__()

        # initialize rnn layer
        if rnn_type == "LSTM":
            self.rnn = BayesLSTM(
                input_size,
                hidden_rnn,
                # rnn_n_layers,
                dropout=rnn_dropout,
                n_layers=rnn_n_layers,
                batch_first=True,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_rnn,
                rnn_n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size,
                hidden_rnn,
                rnn_n_layers,
                dropout=rnn_dropout,
                batch_first=True,
            )

        # initialize linear layers
        if linear_dims[0] != hidden_rnn:
            linear_dims = [hidden_rnn] + linear_dims

        self.linears = nn.ModuleList(
            [
                bnn.BayesLinear(
                    prior_mu=prior_mu,
                    prior_sigma=prior_sigma,
                    in_features=linear_dims[i],
                    out_features=linear_dims[i + 1],
                )
                for i in range(len(linear_dims) - 1)
            ]
        )
        self.output_layer = bnn.BayesLinear(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            in_features=linear_dims[-1],
            out_features=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward propagation through model.

        :param inputs: batch of sequences with MNIST image
        :return: probabilities of changes for each sequence
        """
        batch_size, seq_len = inputs.size()[:2]

        if inputs.type() != "torch.FloatTensor":
            inputs = inputs.float()

        out = inputs.flatten(2, -1)  # batch_size, seq_len, input_size
        out, _ = self.rnn(out)  # batch_size, seq_len, hidden_dim
        out = out.flatten(0, 1)  # batch_size * seq_len, hidden_dim

        for layer in self.linears:
            out = layer(out)
            out = self.relu(self.dropout(out))

        out = self.output_layer(out)
        out = self.sigmoid(out)
        out = out.reshape(batch_size, seq_len, 1)
        return out


class CombinedVideoBayesRNN(nn.Module):
    """LSTM-based network for experiments with videos."""

    def __init__(
        self,
        input_dim: int,
        rnn_hidden_dim: int,
        num_layers: int,
        rnn_dropout: float,
        dropout: float,
        prior_mu: float,
        prior_sigma: float,
    ) -> None:
        """Initialize combined LSTM model for video datasets.

        :param input_dim: dimension of the input data (after feature extraction)
        :param rnn_hidden_dim: hidden dimension for LSTM block
        :param rnn_dropuot: dropout probability in LSTM block
        :param dropout: dropout probability in Dropout layer
        """
        super().__init__()

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.rnn = BayesLSTM(
            input_dim,
            rnn_hidden_dim,
            dropout=rnn_dropout,
            n_layers=num_layers,
            batch_first=True,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        self.fc = bnn.BayesLinear(
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            in_features=rnn_hidden_dim,
            out_features=1,
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x: input torch tensor
        :return: out of the model
        """
        r_out, _ = self.rnn(x)
        r_out = self.dropout(self.fc(r_out))
        out = torch.sigmoid(r_out)
        return out


class BayesCPDModel(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        kl_coeff: float = None,
        n_samples: int = 10,
        std_coeff: float = None,
        half_std_wnd: int = None,
        seed: int = 42,
    ) -> None:
        """Initialize CPD model.

        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param args: dict with supplementary argumemnts
        :param model: base model
        :param train_dataset: train data
        :param test_dataset: test data
        """
        super().__init__()

        self.experiments_name = args["experiments_name"]
        self.model = model

        if self.experiments_name in ["explosion", "road_accidents"]:
            print("Loading extractor...")
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

            # freeze extractor parameters
            for param in self.extractor.parameters():
                param.requires_grad = False
        else:
            self.extractor = None

        self.learning_rate = args["learning"]["lr"]
        self.batch_size = args["learning"]["batch_size"]
        self.num_workers = args["num_workers"]

        self.T = args["loss"]["T"]

        if args["loss_type"] == "indid":
            self.loss = loss.CPDLoss(len_segment=self.T)
        elif args["loss_type"] == "bce":
            self.loss = nn.BCELoss()
        else:
            raise ValueError(
                "Wrong loss_type {}. Please, choose 'indid' or 'bce' loss_type.".format(
                    args["loss_type"]
                )
            )

        self.n_samples = n_samples

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.kl_coeff = kl_coeff
        self.std_coeff = std_coeff

        if self.kl_coeff is not None:
            self.bkl_loss = bnn.BKLLoss()

        if self.std_coeff is not None:
            assert half_std_wnd is not None, "Specify wnd size for std regularization."

        self.half_std_wnd = half_std_wnd

        self.args = args

        fix_seeds(seed)

    def __preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method
        """
        if self.experiments_name in ["explosion", "road_accidents"]:
            inputs = self.extractor(inputs.float())
            inputs = inputs.transpose(1, 2).flatten(
                2
            )  # shape is (batch_size,  C*H*W, seq_len)

        # do nothing for non-video experiments
        return inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(self.__preprocess(inputs))

    def predict_all_models(
        self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:  # change naming for consistency
        preds = []
        for _ in range(self.n_samples):
            preds.append(self.model(inputs))

        preds = torch.stack(preds)
        self.preds = preds

        return preds

    def predict(
        self,
        inputs: torch.Tensor,
        scale: float = None,
        step: int = 1,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.predict_all_models(inputs, scale=scale, step=step, alpha=alpha)

        mean_preds = torch.mean(preds, axis=0)
        std_preds = torch.std(preds, axis=0)

        return mean_preds, std_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        loss = self.loss(pred.squeeze(), labels.float().squeeze())

        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("train_cpd_loss", loss, prog_bar=True)

        if self.kl_coeff is not None:
            bkl_loss = self.bkl_loss(self.model)[0]
            self.log("train_bkl_loss", bkl_loss, prog_bar=True)

            loss += self.kl_coeff * bkl_loss

        if self.std_coeff is not None:
            std_wnd_mask = labels_to_window_mask(
                labels, mode="window", half_wnd=self.half_std_wnd
            )

            _, std_preds = self.predict(inputs)

            target_stds_abnorm = std_preds[std_wnd_mask == 1]
            sigma_loss_abnorm = (
                torch.mean(target_stds_abnorm) if len(target_stds_abnorm) > 0 else 0.0
            )

            target_stds_norm = std_preds[std_wnd_mask == 0]
            sigma_loss_norm = (
                torch.mean(target_stds_norm) if len(target_stds_norm) > 0 else 0.0
            )

            sigma_loss = sigma_loss_norm - sigma_loss_abnorm

            self.log("train_std_loss", sigma_loss, prog_bar=True)

            loss += self.std_coeff * sigma_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        loss = self.loss(pred.squeeze(), labels.float().squeeze())

        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("val_cpd_loss", loss, prog_bar=True)

        if self.kl_coeff is not None:
            bkl_loss = self.bkl_loss(self.model)[0]

            self.log("val_bkl_loss", bkl_loss, prog_bar=True)
            loss += self.kl_coeff * bkl_loss

        if self.std_coeff is not None:
            std_wnd_mask = labels_to_window_mask(
                labels, mode="window", half_wnd=self.half_std_wnd
            )

            _, std_preds = self.predict(inputs)

            target_stds_abnorm = std_preds[std_wnd_mask == 1]
            sigma_loss_abnorm = (
                torch.mean(target_stds_abnorm) if len(target_stds_abnorm) > 0 else 0.0
            )

            target_stds_norm = std_preds[std_wnd_mask == 0]
            sigma_loss_norm = (
                torch.mean(target_stds_norm) if len(target_stds_norm) > 0 else 0.0
            )

            sigma_loss = sigma_loss_norm - sigma_loss_abnorm

            self.log("val_std_loss", sigma_loss, prog_bar=True)

            loss += self.std_coeff * sigma_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
