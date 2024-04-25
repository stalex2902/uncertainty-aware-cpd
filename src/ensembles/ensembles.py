import os
from abc import ABC
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.baselines import klcpd, tscp
from src.datasets import datasets
from src.ensembles import distances
from src.models import model_utils
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import Subset

EPS = 1e-6


class EnsembleCPDModel(ABC):
    """Wrapper for general ensemble models with bootstrapping."""

    def __init__(
        self,
        args: dict,
        n_models: int,
        boot_sample_size: int = None,
        train_anomaly_num: int = None,
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset
                                 (if None, all the models are trained on the original train dataset)
        """
        super().__init__()

        self.args = args

        assert args["experiments_name"] in [
            "synthetic_1D",
            "synthetic_100D",
            "mnist",
            "human_activity",
            "explosion",
            "road_accidents",
            "yahoo",
        ], "Wrong experiments name"

        self.train_dataset, self.test_dataset = datasets.CPDDatasets(
            experiments_name=args["experiments_name"],
            train_anomaly_num=train_anomaly_num,
        ).get_dataset_()

        self.n_models = n_models

        if boot_sample_size is not None:
            assert boot_sample_size <= len(
                self.train_dataset
            ), "Desired sample size is larger than the whole train dataset."
        self.boot_sample_size = boot_sample_size

        self.fitted = False
        self.initialize_models_list()

    def eval(self) -> None:
        """Turn all the models to 'eval' mode (for consistency with our code)."""
        for model in self.models_list:
            model.eval()

    def to(self, device: str) -> None:
        """Move all models to the device (for consistency with our code)."""
        for model in self.models_list:
            model.to(device)

    def bootstrap_datasets(self) -> None:
        """Generate new train datasets if necessary."""
        # No boostrap
        if self.boot_sample_size is None:
            self.train_datasets_list = [self.train_dataset] * self.n_models

        else:
            # for reproducibility of torch.randint()
            torch.manual_seed(42)

            self.train_datasets_list = []
            for _ in range(self.n_models):
                # sample with replacement
                idxs = torch.randint(
                    len(self.train_dataset), size=(self.boot_sample_size,)
                )
                curr_train_data = Subset(self.train_dataset, idxs)
                self.train_datasets_list.append(curr_train_data)

    def initialize_models_list(self) -> None:
        """Initialize cpd models for a particular exeriment."""
        self.bootstrap_datasets()

        self.models_list = []
        for i in range(self.n_models):
            fix_seeds(i)

            curr_model = model_utils.get_models_list(
                self.args, self.train_datasets_list[i], self.test_dataset
            )[
                -1
            ]  # list consists of 1 model as, currently, we do not work with 'combined' models
            self.models_list.append(curr_model)

    def fit(self) -> None:
        """Fit all the models on the corresponding train datasets."""
        logger = TensorBoardLogger(
            save_dir=f'logs/{self.args["experiments_name"]}',
            name=self.args["model_type"],
        )

        if not self.fitted:
            self.initialize_models_list()
            for i, cpd_model in enumerate(self.models_list):
                fix_seeds(i)

                print(f"\nFitting model number {i + 1}.")
                trainer = pl.Trainer(
                    max_epochs=self.args["learning"]["epochs"],
                    accelerator=self.args["learning"]["accelerator"],
                    devices=self.args["learning"]["devices"],
                    benchmark=True,
                    check_val_every_n_epoch=1,
                    logger=logger,
                    # callbacks=EarlyStopping(
                    #    monitor=monitor, min_delta=min_delta, patience=patience
                    # ),
                    callbacks=EarlyStopping(**self.args["early_stopping"]),
                )
                trainer.fit(cpd_model)

            self.fitted = True

        else:
            print("Attention! Models are already fitted!")

    def predict_all_models(
        self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
    ):
        if not self.fitted:
            print("Attention! The model is not fitted yet.")

        self.eval()

        ensemble_preds = []
        for model in self.models_list:
            inputs = inputs.to(model.device)
            if self.args["model_type"] == "seq2seq":
                outs = model(inputs).squeeze()
            elif self.args["model_type"] == "kl_cpd":
                outs = klcpd.get_klcpd_output_scaled(
                    model, inputs, model.window_1, model.window_2, scale=scale
                )
            elif self.args["model_type"] == "tscp":
                # outs = tscp.get_tscp_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale)
                # outs = tscp.get_tscp_output_scaled_padded(
                #    model, inputs, model.window_1, model.window_2, scale=scale, step=step, alpha=alpha
                # )
                outs = tscp.get_tscp_output_padded(
                    model, inputs, model.window_1, model.window_2, step=step
                )
                # outs = tscp.get_tscp_output(
                #    model, inputs, model.window_1, model.window_2, step=step
                # )
                outs = tscp.post_process_tscp_output(outs, scale=scale, alpha=alpha)

            else:
                raise ValueError(
                    f'Wrong or not implemented model type {self.args["model_type"]}.'
                )
            ensemble_preds.append(outs)

        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)
        self.preds = ensemble_preds

        return ensemble_preds

    def predict(
        self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:
        """Make a prediction.

        :param inputs: input batch of sequences
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing predictions of all the models
        """
        ensemble_preds = self.predict_all_models(inputs, scale, step, alpha)

        _, batch_size, seq_len = ensemble_preds.shape

        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

        # if self.args["model_type"] == "tscp":
        #    preds_mean = tscp.post_process_tscp_output(preds_mean, scale=scale, alpha=alpha)

        # TODO: how to fix std post-processing?..
        # preds_std = tscp.post_process_tscp_output(preds_std, scale=scale, alpha=alpha)
        # preds_std = torch.zeros_like(preds_mean)

        return preds_mean, preds_std

    def get_quantile_predictions(
        self, inputs: torch.Tensor, q: float, scale: int = None
    ) -> torch.Tensor:
        """Get the q-th quantile of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param q: desired quantile
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)
        preds_quantile = torch.quantile(self.preds, q, axis=0)
        return preds_quantile, preds_std

    def get_min_max_predictions(
        self, inputs: torch.Tensor, mode: str, scale: int = None
    ) -> torch.Tensor:
        """Get the point-wise minimum/maximum of the predicted CP scores distribution.

        :param inputs: input batch of sequences
        :param scale: scale parameter for KL-CPD and TSCP2 models

        :returns: torch.Tensor containing quantile predictions
        """
        _, preds_std = self.predict(inputs)

        # torch.min() and torch.max() return a tuple (values, indices)
        if mode == "min":
            preds = torch.min(self.preds, axis=0)[0]
        elif mode == "max":
            preds = torch.max(self.preds, axis=0)[0]
        else:
            raise ValueError(f"Wring mode {mode}. Only 'min' and 'max' are available.")
        return preds, preds_std

    def save_models_list(self, path_to_folder: str) -> None:
        """Save trained models.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """

        if not self.fitted:
            print("Attention! The models are not trained.")

        loss_type = (
            self.args["loss_type"] if self.args["model_type"] == "seq2seq" else None
        )

        for i, model in enumerate(self.models_list):
            path = (
                path_to_folder
                + "/"
                + self.args["experiments_name"]
                + "_loss_type_"
                + str(loss_type)
                + "_model_type_"
                + self.args["model_type"]
                + "_sample_"
                + str(self.boot_sample_size)
                + "_model_num_"
                + str(i)
                + ".pth"
            )
            torch.save(model.state_dict(), path)

    def load_models_list(self, path_to_folder: str) -> None:
        """Load weights of the saved models from the ensemble.

        :param path_to_folder: path to the folder for saving, e.g. 'saved_models/mnist'
        """
        # check that the folder contains self.n_models files with models' weights,
        # ignore utility files
        paths_list = [
            path for path in os.listdir(path_to_folder) if not path.startswith(".")
        ]

        assert (
            len(paths_list) == self.n_models
        ), "Number of paths is not equal to the number of models."

        # initialize models list
        self.initialize_models_list()

        # load state dicts
        for model, path in zip(self.models_list, paths_list):
            try:
                model.load_state_dict(torch.load(path_to_folder + "/" + path))
            except RuntimeError:
                model.model.load_state_dict(torch.load(path_to_folder + "/" + path))

        self.fitted = True


# class CusumEnsembleCPDModel(EnsembleCPDModel):
#     """Wrapper for cusum aproach ensemble models."""

#     def __init__(
#         self,
#         args: dict,
#         n_models: int,
#         global_sigma: Optional[
#             Union[float, str]
#         ] = None,  # float / 'local_start' / 'local_whole' / None (for 'non-cond' variants)
#         boot_sample_size: int = None,
#         train_anomaly_num: int = None,
#         cusum_threshold: float = 0.1,
#         cusum_mode: str = "correct",
#         conditional: bool = False,
#         lambda_null: float = None,
#         lambda_inf: float = None,
#         half_wnd: int = None,
#         var_coeff: float = 1.0,
#     ) -> None:
#         """Initialize EnsembleCPDModel.

#         :param args: dictionary containing core model params, learning params, loss params, etc.
#         :param n_models: number of models to train
#         :param boot_sample_size: size of the bootstrapped train dataset
#                                  (if None, all the models are trained on the original train dataset)
#         :param scale_by_std: if True, scale the statistic by predicted std, i.e.
#                                 in cusum, t = series_mean[i] - series_mean[i-1]) / series_std[i],
#                              else:
#                                 t = series_mean[i] - series_mean[i-1]
#         :param susum_threshold: threshold for CUSUM algorithm
#         """
#         super().__init__(args, n_models, boot_sample_size, train_anomaly_num)

#         assert cusum_mode in [
#             "correct",
#             "old",
#             "new_criteria",
#         ], f"Wrong CUSUM mode: {cusum_mode}"

#         if cusum_mode == "new_criteria":
#             assert lambda_null is not None, "Specify lambda_null for 'new_crit'"
#             # assert lambda_inf is not None, "Specify lambda_inf for 'new_crit'"
#             assert half_wnd is not None, "Specify half_wnd for 'new_crit'"

#         self.cusum_threshold = cusum_threshold
#         self.cusum_mode = cusum_mode
#         self.conditional = conditional

#         if not self.conditional:
#             assert (
#                 global_sigma is not None
#             ), "Global sigma is required for non-conditional statisctics."
#             assert isinstance(global_sigma, float) or isinstance(
#                 global_sigma, str
#             ), "If not None, global_sigma should be either str or float"

#             if isinstance(global_sigma, str):
#                 assert global_sigma in [
#                     "local_start",
#                     "local_whole",
#                 ], "Unknow local global_sigma type"
#                 assert half_wnd is not None, "Need window size for local global_sigma"

#         self.global_sigma = global_sigma

#         self.lambda_null = lambda_null
#         self.lambda_inf = lambda_inf
#         self.half_wnd = half_wnd

#         self.var_coeff = var_coeff

#     def cusum_detector(
#         self, series_batch: torch.Tensor, series_std_batch: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len = series_batch.shape

#         normal_to_change_stat = torch.zeros(batch_size, seq_len).to(series_batch.device)
#         change_mask = torch.zeros(batch_size, seq_len).to(series_batch.device)

#         # TODO: check!
#         if self.global_sigma == "local_start":
#             global_sigma = series_std_batch[:, : 2 * self.half_wnd].mean(axis=1)
#         elif self.global_sigma == "local_whole":
#             global_sigma = series_std_batch.mean(axis=1)
#         else:
#             global_sigma = self.global_sigma

#         for i in range(1, seq_len):
#             # old CUSUM
#             if self.cusum_mode == "old":
#                 if self.conditional:
#                     t = (series_batch[:, i] - series_batch[:, i - 1]) / (
#                         series_std_batch[:, i] + EPS
#                     )
#                 else:
#                     t = series_batch[:, i] - series_batch[:, i - 1] / (
#                         global_sigma + EPS
#                     )
#             # new (correct) CUSUM
#             else:
#                 if self.conditional:
#                     t = (series_batch[:, i] - 0.5) / (series_std_batch[:, i] ** 2 + EPS)
#                 else:
#                     t = (series_batch[:, i] - 0.5) / (global_sigma**2 + EPS)

#             normal_to_change_stat[:, i] = torch.maximum(
#                 torch.zeros(batch_size).to(series_batch.device),
#                 normal_to_change_stat[:, i - 1] + t,
#             )

#             is_change = (
#                 normal_to_change_stat[:, i]
#                 > torch.ones(batch_size).to(series_batch.device) * self.cusum_threshold
#             )
#             change_mask[is_change, i:] = True

#         return change_mask, normal_to_change_stat

#     def new_scores_aggregator(
#         self, series_batch: torch.Tensor, series_std_batch: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         batch_size, seq_len = series_batch.shape

#         normal_to_change_stat = torch.zeros(batch_size, seq_len).to(series_batch.device)
#         change_mask = torch.zeros(batch_size, seq_len).to(series_batch.device)

#         # TODO: check!
#         if self.global_sigma == "local_start":
#             global_sigma = series_std_batch[:, : 2 * self.half_wnd].mean(axis=1)
#         elif self.global_sigma == "local_whole":
#             global_sigma = series_std_batch.mean(axis=1)
#         else:
#             global_sigma = self.global_sigma

#         if self.lambda_inf is None:
#             # compute lambda_inf based on the local global_sigma
#             lambda_inf = 1.0 / global_sigma**2
#         else:
#             lambda_inf = self.lambda_inf
#         lambda_null = self.lambda_null

#         for i in range(1, seq_len):
#             if self.conditional:
#                 t = (series_batch[:, i] - 0.5) / (series_std_batch[:, i] ** 2 + EPS)
#             else:
#                 t = (series_batch[:, i] - 0.5) / (global_sigma**2 + EPS)

#             wnd_start = max(0, i - self.half_wnd)
#             wnd_end = min(seq_len, i + self.half_wnd + 1)
#             # wnd_end = i

#             windom_var_sum = self.var_coeff * sum(
#                 [series_std_batch[:, k] ** 2 for k in range(wnd_start, wnd_end)]
#             )

#             normal_to_change_stat[:, i] = torch.maximum(
#                 (lambda_inf - lambda_null) * windom_var_sum,
#                 normal_to_change_stat[:, i - 1] + t,
#             )

#             is_change = (
#                 normal_to_change_stat[:, i]
#                 > torch.ones(batch_size).to(series_batch.device) * self.cusum_threshold
#             )
#             change_mask[is_change, i:] = True

#         return change_mask, normal_to_change_stat

#     def sample_cusum_trajectories(self, inputs):
#         if not self.fitted:
#             print("Attention! The model is not fitted yet.")

#         self.eval()

#         ensemble_preds = []

#         for model in self.models_list:
#             ensemble_preds.append(model(inputs).squeeze())

#         # shape is (n_models, batch_size, seq_len)
#         ensemble_preds = torch.stack(ensemble_preds)

#         _, batch_size, seq_len = ensemble_preds.shape

#         preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

#         cusum_trajectories = []
#         change_masks = []

#         for preds_traj in ensemble_preds:
#             # use one_like tensor of std's, do not take them into account
#             # change_mask, normal_to_change_stat = self.cusum_detector_batch(preds_traj, torch.ones_like(preds_traj))
#             change_mask, normal_to_change_stat = self.cusum_detector(
#                 preds_traj, preds_std
#             )
#             cusum_trajectories.append(normal_to_change_stat)
#             change_masks.append(change_mask)

#         cusum_trajectories = torch.stack(cusum_trajectories)
#         change_masks = torch.stack(change_masks)

#         return change_masks, cusum_trajectories

#     def predict(
#         self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
#     ) -> torch.Tensor:
#         """Make a prediction.

#         :param inputs: input batch of sequences

#         :returns: torch.Tensor containing predictions of all the models
#         """
#         # shape is (n_models, batch_size, seq_len)
#         ensemble_preds = self.predict_all_models(inputs, scale, step, alpha)

#         _, batch_size, seq_len = ensemble_preds.shape

#         preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
#         preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

#         if self.cusum_mode in ["old", "correct"]:
#             change_masks, normal_to_change_stats = self.cusum_detector(
#                 preds_mean, preds_std
#             )
#         else:
#             change_masks, normal_to_change_stats = self.new_scores_aggregator(
#                 preds_mean, preds_std
#             )

#         self.preds_mean = preds_mean
#         self.preds_std = preds_std

#         self.change_masks = change_masks
#         self.normal_to_change_stats = normal_to_change_stats

#         return change_masks

#     def predict_cusum_trajectories(
#         self, inputs: torch.Tensor, q: float = 0.5
#     ) -> torch.Tensor:
#         """Make a prediction.

#         :param inputs: input batch of sequences

#         :returns: torch.Tensor containing predictions of all the models
#         """
#         change_masks, _ = self.sample_cusum_trajectories(inputs)
#         cp_idxs_batch = torch.argmax(change_masks, dim=2).float()
#         cp_idxs_batch_aggr = torch.quantile(cp_idxs_batch, q, axis=0).round().int()
#         _, bs, seq_len = change_masks.shape
#         cusum_quantile_labels = torch.zeros(bs, seq_len).to(inputs.device)

#         for b in range(bs):
#             if cp_idxs_batch_aggr[b] > 0:
#                 cusum_quantile_labels[b, cp_idxs_batch_aggr[b] :] = 1

#         return cusum_quantile_labels

#     def fake_predict(self, series_batch: torch.Tensor, series_std_batch: torch.Tensor):
#         """In case of pre-computed model outputs."""
#         if self.cusum_mode in ["old", "correct"]:
#             return self.cusum_detector(series_batch, series_std_batch)  # [0]
#         elif self.cusum_mode == "new_criteria":
#             return self.new_scores_aggregator(series_batch, series_std_batch)  # [0]


class CusumEnsembleCPDModel(ABC):
    """Wrapper for cusum aproach ensemble models."""

    def __init__(
        self,
        # args: dict,
        # n_models: int,
        ens_model,
        global_sigma: Optional[
            Union[float, str]
        ] = None,  # float / 'local_start' / 'local_whole' / None (for 'non-cond' variants)
        # boot_sample_size: int = None,
        # train_anomaly_num: int = None,
        cusum_threshold: float = 0.1,
        cusum_mode: str = "correct",
        conditional: bool = False,
        lambda_null: float = None,
        lambda_inf: float = None,
        half_wnd: int = None,
        var_coeff: float = 1.0,
    ) -> None:
        """Initialize EnsembleCPDModel.

        :param args: dictionary containing core model params, learning params, loss params, etc.
        :param n_models: number of models to train
        :param boot_sample_size: size of the bootstrapped train dataset
                                 (if None, all the models are trained on the original train dataset)
        :param scale_by_std: if True, scale the statistic by predicted std, i.e.
                                in cusum, t = series_mean[i] - series_mean[i-1]) / series_std[i],
                             else:
                                t = series_mean[i] - series_mean[i-1]
        :param susum_threshold: threshold for CUSUM algorithm
        """
        super().__init__()

        assert cusum_mode in [
            "correct",
            "old",
            "new_criteria",
        ], f"Wrong CUSUM mode: {cusum_mode}"

        if cusum_mode == "new_criteria":
            assert lambda_null is not None, "Specify lambda_null for 'new_crit'"
            # assert lambda_inf is not None, "Specify lambda_inf for 'new_crit'"
            assert half_wnd is not None, "Specify half_wnd for 'new_crit'"

        self.ens_model = ens_model

        self.cusum_threshold = cusum_threshold
        self.cusum_mode = cusum_mode
        self.conditional = conditional

        if not self.conditional:
            assert (
                global_sigma is not None
            ), "Global sigma is required for non-conditional statisctics."
            assert isinstance(global_sigma, float) or isinstance(
                global_sigma, str
            ), "If not None, global_sigma should be either str or float"

            if isinstance(global_sigma, str):
                assert global_sigma in [
                    "local_start",
                    "local_whole",
                ], "Unknow local global_sigma type"
                assert half_wnd is not None, "Need window size for local global_sigma"

        self.global_sigma = global_sigma

        self.lambda_null = lambda_null
        self.lambda_inf = lambda_inf
        self.half_wnd = half_wnd

        self.var_coeff = var_coeff

    def eval(self):
        self.ens_model.eval()

    def to(self, device: str) -> None:
        self.ens_model.to(device)

    def cusum_detector(
        self, series_batch: torch.Tensor, series_std_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = series_batch.shape

        normal_to_change_stat = torch.zeros(batch_size, seq_len).to(series_batch.device)
        change_mask = torch.zeros(batch_size, seq_len).to(series_batch.device)

        # TODO: check!
        if self.global_sigma == "local_start":
            global_sigma = series_std_batch[:, : 2 * self.half_wnd].mean(axis=1)
        elif self.global_sigma == "local_whole":
            global_sigma = series_std_batch.mean(axis=1)
        else:
            global_sigma = self.global_sigma

        for i in range(1, seq_len):
            # old CUSUM
            if self.cusum_mode == "old":
                if self.conditional:
                    t = (series_batch[:, i] - series_batch[:, i - 1]) / (
                        series_std_batch[:, i] + EPS
                    )
                else:
                    t = series_batch[:, i] - series_batch[:, i - 1] / (
                        global_sigma + EPS
                    )
            # new (correct) CUSUM
            else:
                if self.conditional:
                    t = (series_batch[:, i] - 0.5) / (series_std_batch[:, i] ** 2 + EPS)
                else:
                    t = (series_batch[:, i] - 0.5) / (global_sigma**2 + EPS)

            normal_to_change_stat[:, i] = torch.maximum(
                torch.zeros(batch_size).to(series_batch.device),
                normal_to_change_stat[:, i - 1] + t,
            )

            is_change = (
                normal_to_change_stat[:, i]
                > torch.ones(batch_size).to(series_batch.device) * self.cusum_threshold
            )
            change_mask[is_change, i:] = True

        return change_mask, normal_to_change_stat

    def new_scores_aggregator(
        self, series_batch: torch.Tensor, series_std_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = series_batch.shape

        normal_to_change_stat = torch.zeros(batch_size, seq_len).to(series_batch.device)
        change_mask = torch.zeros(batch_size, seq_len).to(series_batch.device)

        # TODO: check!
        if self.global_sigma == "local_start":
            global_sigma = series_std_batch[:, : 2 * self.half_wnd].mean(axis=1)
        elif self.global_sigma == "local_whole":
            global_sigma = series_std_batch.mean(axis=1)
        else:
            global_sigma = self.global_sigma

        if self.lambda_inf is None:
            # compute lambda_inf based on the local global_sigma
            lambda_inf = 1.0 / global_sigma**2
        else:
            lambda_inf = self.lambda_inf
        lambda_null = self.lambda_null

        for i in range(1, seq_len):
            if self.conditional:
                t = (series_batch[:, i] - 0.5) / (series_std_batch[:, i] ** 2 + EPS)
            else:
                t = (series_batch[:, i] - 0.5) / (global_sigma**2 + EPS)

            wnd_start = max(0, i - self.half_wnd)
            wnd_end = min(seq_len, i + self.half_wnd + 1)
            # wnd_end = i

            windom_var_sum = self.var_coeff * sum(
                [series_std_batch[:, k] ** 2 for k in range(wnd_start, wnd_end)]
            )

            normal_to_change_stat[:, i] = torch.maximum(
                (lambda_inf - lambda_null) * windom_var_sum,
                normal_to_change_stat[:, i - 1] + t,
            )

            is_change = (
                normal_to_change_stat[:, i]
                > torch.ones(batch_size).to(series_batch.device) * self.cusum_threshold
            )
            change_mask[is_change, i:] = True

        return change_mask, normal_to_change_stat

    def sample_cusum_trajectories(self, inputs):
        if not self.ens_model.fitted:
            print("Attention! The model is not fitted yet.")

        self.eval()

        ensemble_preds = []

        for model in self.ens_model.models_list:
            ensemble_preds.append(model(inputs).squeeze())

        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = torch.stack(ensemble_preds)

        _, batch_size, seq_len = ensemble_preds.shape

        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

        cusum_trajectories = []
        change_masks = []

        for preds_traj in ensemble_preds:
            # use one_like tensor of std's, do not take them into account
            # change_mask, normal_to_change_stat = self.cusum_detector_batch(preds_traj, torch.ones_like(preds_traj))
            change_mask, normal_to_change_stat = self.cusum_detector(
                preds_traj, preds_std
            )
            cusum_trajectories.append(normal_to_change_stat)
            change_masks.append(change_mask)

        cusum_trajectories = torch.stack(cusum_trajectories)
        change_masks = torch.stack(change_masks)

        return change_masks, cusum_trajectories

    def predict(
        self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:
        """Make a prediction.

        :param inputs: input batch of sequences

        :returns: torch.Tensor containing predictions of all the models
        """
        # shape is (n_models, batch_size, seq_len)
        ensemble_preds = self.ens_model.predict_all_models(inputs, scale, step, alpha)

        _, batch_size, seq_len = ensemble_preds.shape

        preds_mean = torch.mean(ensemble_preds, axis=0).reshape(batch_size, seq_len)
        preds_std = torch.std(ensemble_preds, axis=0).reshape(batch_size, seq_len)

        if self.cusum_mode in ["old", "correct"]:
            change_masks, normal_to_change_stats = self.cusum_detector(
                preds_mean, preds_std
            )
        else:
            change_masks, normal_to_change_stats = self.new_scores_aggregator(
                preds_mean, preds_std
            )

        self.preds_mean = preds_mean
        self.preds_std = preds_std

        self.change_masks = change_masks
        self.normal_to_change_stats = normal_to_change_stats

        return change_masks

    def predict_cusum_trajectories(
        self, inputs: torch.Tensor, q: float = 0.5
    ) -> torch.Tensor:
        """Make a prediction.

        :param inputs: input batch of sequences

        :returns: torch.Tensor containing predictions of all the models
        """
        change_masks, _ = self.sample_cusum_trajectories(inputs)
        cp_idxs_batch = torch.argmax(change_masks, dim=2).float()
        cp_idxs_batch_aggr = torch.quantile(cp_idxs_batch, q, axis=0).round().int()
        _, bs, seq_len = change_masks.shape
        cusum_quantile_labels = torch.zeros(bs, seq_len).to(inputs.device)

        for b in range(bs):
            if cp_idxs_batch_aggr[b] > 0:
                cusum_quantile_labels[b, cp_idxs_batch_aggr[b] :] = 1

        return cusum_quantile_labels

    def fake_predict(self, series_batch: torch.Tensor, series_std_batch: torch.Tensor):
        """In case of pre-computed model outputs."""
        if self.cusum_mode in ["old", "correct"]:
            return self.cusum_detector(series_batch, series_std_batch)  # [0]
        elif self.cusum_mode == "new_criteria":
            return self.new_scores_aggregator(series_batch, series_std_batch)  # [0]


# class DistanceEnsembleCPDModel(EnsembleCPDModel):
#     def __init__(
#         self,
#         args: dict,
#         n_models: int,
#         window_size: int,
#         boot_sample_size: int = None,
#         train_anomaly_num: int = None,
#         anchor_window_type: str = "start",
#         threshold: float = 0.1,
#         distance: str = "mmd",
#         kernel: str = "rbf",
#     ) -> None:
#         super().__init__(args, n_models, boot_sample_size, train_anomaly_num)

#         assert anchor_window_type in [
#             "sliding",
#             "start",
#             "prev",
#             "combined",
#         ], "Unknown window type"
#         assert distance in [
#             "mmd",
#             "cosine",
#             "wasserstein_1d",
#             "wasserstein_nd",
#         ], "Unknown distance type"

#         if distance == "mmd":
#             assert kernel in [
#                 "rbf",
#                 "multiscale",
#             ], f"Wrong kernel type: {kernel}."

#         self.anchor_window_type = anchor_window_type
#         self.distance = distance
#         self.window_size = window_size
#         self.threshold = threshold
#         self.kernel = kernel

#     def distance_detector(self, ensemble_preds: torch.Tensor):
#         # anchor windows: 'start', 'prev', or 'combined'
#         scores = distances.anchor_window_detector_batch(
#             ensemble_preds,
#             window_size=self.window_size,
#             distance=self.distance,
#             kernel=self.kernel,
#             anchor_window_type=self.anchor_window_type,
#         )
#         labels = (scores > self.threshold).to(torch.int)

#         return labels, scores

#     def predict(
#         self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
#     ) -> torch.Tensor:
#         ensemble_preds = self.predict_all_models(inputs, scale, step, alpha)
#         preds, _ = self.distance_detector(ensemble_preds)
#         return preds

#     def fake_predict(self, ensemble_preds: torch.Tensor):
#         """In case of pre-computed model outputs."""
#         return self.distance_detector(ensemble_preds.transpose(0, 1))
#         # return preds, scores


class DistanceEnsembleCPDModel(ABC):
    def __init__(
        self,
        ens_model,
        # args: dict,
        # n_models: int,
        window_size: int,
        # boot_sample_size: int = None,
        # train_anomaly_num: int = None,
        anchor_window_type: str = "start",
        threshold: float = 0.1,
        distance: str = "mmd",
        kernel: str = "rbf",
    ) -> None:
        super().__init__()

        assert anchor_window_type in [
            "sliding",
            "start",
            "prev",
            "combined",
        ], "Unknown window type"
        assert distance in [
            "mmd",
            "cosine",
            "wasserstein_1d",
            "wasserstein_nd",
        ], "Unknown distance type"

        if distance == "mmd":
            assert kernel in [
                "rbf",
                "multiscale",
            ], f"Wrong kernel type: {kernel}."

        self.ens_model = ens_model

        self.anchor_window_type = anchor_window_type
        self.distance = distance
        self.window_size = window_size
        self.threshold = threshold
        self.kernel = kernel

    def eval(self):
        self.ens_model.eval()

    def to(self, device: str) -> None:
        self.ens_model.to(device)

    def distance_detector(self, ensemble_preds: torch.Tensor):
        # anchor windows: 'start', 'prev', or 'combined'
        scores = distances.anchor_window_detector_batch(
            ensemble_preds,
            window_size=self.window_size,
            distance=self.distance,
            kernel=self.kernel,
            anchor_window_type=self.anchor_window_type,
        )
        labels = (scores > self.threshold).to(torch.int)

        return labels, scores

    def predict(
        self, inputs: torch.Tensor, scale: int = None, step: int = 1, alpha: float = 1.0
    ) -> torch.Tensor:
        ensemble_preds = self.ens_model.predict_all_models(inputs, scale, step, alpha)
        preds, _ = self.distance_detector(ensemble_preds)
        return preds

    def fake_predict(self, ensemble_preds: torch.Tensor):
        """In case of pre-computed model outputs."""
        return self.distance_detector(ensemble_preds.transpose(0, 1))
        # return preds, scores
