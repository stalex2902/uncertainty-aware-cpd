# ------------------------------------------------------------------------------------------------------------#
#                             From https://github.com/gpleiss/temperature_scaling                             #
# ------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import numpy as np
import torch
from betacal import BetaCalibration
from sklearn.calibration import calibration_curve
from src.metrics.metrics_utils import collect_model_predictions_on_set
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, preprocessor=None, lr=1e-2, max_iter=50, verbose=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.preprocessor = preprocessor
        self.lr = lr
        self.max_iter = max_iter

        self.loss_history = []

    def get_logits(self, input):
        if self.preprocessor:
            input = self.preprocessor(input.float())
            input = input.transpose(1, 2).flatten(2)
        logits = self.model(input)
        return logits

    def forward(self, input):
        logits = self.get_logits(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def fit(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, disable=not self.verbose):
                input = input.cuda()
                _logits = self.get_logits(input)

                label = label.long().flatten()  # UPDATE
                _logits = _logits.flatten()  # UPDATE

                num_samples = len(label)
                logits = torch.empty(num_samples, 2)
                logits[:, 0] = 1 - _logits
                logits[:, 1] = _logits

                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        if self.verbose:
            print(
                "Before temperature - NLL: %.3f, ECE: %.3f"
                % (before_temperature_nll, before_temperature_ece)
            )

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS(
            [self.temperature], lr=self.lr, max_iter=self.max_iter
        )  # QQQ: parameters??

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()

            self.loss_history.append(round(loss.item(), 4))

            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        temp = self.temperature.item()

        if self.verbose:
            print("Optimal temperature: %.3f" % temp)
            print(
                "After temperature - NLL: %.3f, ECE: %.3f"
                % (after_temperature_nll, after_temperature_ece)
            )

        # update temperature after calibration
        self.model.temperature = temp

        return self

    def predict(self, dataloader, model_type="seq2seq", device="cpu", verbose=False):
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataloader,
            model_type=model_type,
            device=device,
            verbose=verbose,
        )
        preds_cal_flat = torch.vstack(test_out_bank).flatten()
        labels_flat = torch.vstack(test_labels_bank).flatten()

        return preds_cal_flat, labels_flat


class ModelBeta:
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, parameters="abm", preprocessor=None, verbose=True):
        super(ModelBeta, self).__init__()
        self.model = model
        self.preprocessor = preprocessor

        self.calibrator = BetaCalibration(parameters)

    # This function probably should live outside of this class, but whatever
    def fit(self, dataoader, model_type="seq2seq", device="cuda", verbose=True):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataoader,
            model_type=model_type,
            device=device,
            verbose=verbose,
        )

        test_out_flat = torch.vstack(test_out_bank).flatten().numpy()
        test_labels_flat = torch.vstack(test_labels_bank).flatten().numpy()

        self.calibrator.fit(test_out_flat.reshape(-1, 1), test_labels_flat)

        return self

    def predict(self, dataloader, model_type="seq2seq", device="cpu", verbose=False):
        test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
            self.model,
            dataloader,
            model_type=model_type,
            device=device,
            verbose=verbose,
        )

        preds_flat = torch.vstack(test_out_bank).flatten()
        labels_flat = torch.vstack(test_labels_bank).flatten()

        preds_cal_flat = self.calibrator.predict(preds_flat)

        return preds_cal_flat, labels_flat


# ------------------------------------------------------------------------------------------------------------#
#                                         Utils for calibration                                               #
# ------------------------------------------------------------------------------------------------------------#


def calibrate_single_model(
    core_model,
    val_dataloader,
    cal_type="beta",
    parameters_beta="abm",
    preprocessor=None,
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
    # return_loss=False,
):
    assert cal_type in ["temperature", "beta"], f"Unknown calibration type {cal_type}"

    core_model.to(device)
    if preprocessor:
        preprocessor.to(device)

    if cal_type == "temperature":
        core_model.return_logits = True
        scaled_model = ModelWithTemperature(
            core_model,
            preprocessor=preprocessor,
            lr=lr,
            max_iter=max_iter,
        )
        scaled_model.fit(val_dataloader, verbose=verbose)
        core_model.return_logits = False

    else:
        scaled_model = ModelBeta(
            core_model,
            parameters=parameters_beta,
            preprocessor=preprocessor,
        )
        scaled_model.fit(val_dataloader, verbose=verbose)

    return scaled_model

    # print("Loss calibration hystory:", scaled_model.loss_history)

    # if return_loss:
    #     return core_model, scaled_model.loss_history
    # else:
    #     return core_model, None

    # return scaled_model


def calibrate_all_models_in_ensemble(
    ensemble_model,
    val_dataloader,
    cal_type,
    preprocessor=None,
    lr=1e-2,
    max_iter=50,
    verbose=True,
    device="cpu",
):
    cal_models = []
    for cpd_model in ensemble_model.models_list:
        cal_model = calibrate_single_model(
            cpd_model.model,
            val_dataloader,
            cal_type,
            preprocessor=preprocessor,
            lr=lr,
            max_iter=max_iter,
            verbose=verbose,
            device=device,
        )
        cal_models.append(cal_model)
    return cal_models


def manually_temperature_calibrate_all_models_in_ensemble(
    ensemble_model, temperature_list
):
    for cpd_model, T in zip(ensemble_model.models_list, temperature_list):
        cpd_model.model.temperature = T


def temperature_uncalibrate_all_models_in_ensemble(ensemble_model):
    for cpd_model in ensemble_model.models_list:
        cpd_model.model.temperature = 1.0


def plot_calibration_curves(
    cal_models_list,
    test_dataloader,
    model_type="seq2seq",
    device="cpu",
    title=None,
    verbose=False,
):
    x_ideal = np.linspace(0, 1, 20)

    plt.figure(figsize=(10, 8))
    plt.plot(x_ideal, x_ideal, linestyle="--", label="Ideally calibrated", c="black")

    for i, cal_model in enumerate(cal_models_list):
        test_out_flat, test_labels_flat = cal_model.predict(
            test_dataloader, model_type=model_type, device=device, verbose=verbose
        )

        prob_true, prob_pred = calibration_curve(
            test_labels_flat, test_out_flat, n_bins=10
        )

        plt.plot(
            prob_pred,
            prob_true,
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=1,
            label=f"Model num {i}",
        )
    if title:
        plt.title(title, fontsize=14)
    plt.xlabel("Predicted probability", fontsize=12)
    plt.ylabel("Fraction of positives", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
