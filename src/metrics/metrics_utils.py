import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import klcpd, tscp
from utils.cpd_models import ClassicBaseline

# ------------------------------------------------------------------------------------------------------------#
#                                         Calculate CPD metrics                                               #
# ------------------------------------------------------------------------------------------------------------#


def find_first_change(mask: np.array) -> np.array:
    """Find first change in batch of predictions.

    :param mask:
    :return: mask with -1 on first change
    """
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind


def calculate_errors(
    real: torch.Tensor, pred: torch.Tensor, seq_len: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real true change points idxs for a batch
    :param pred: predicted change point idxs for a batch
    :param seq_len: length of sequence
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
    """
    FP_delay = torch.zeros_like(real, requires_grad=False)
    delay = torch.zeros_like(real, requires_grad=False)

    tn_mask = torch.logical_and(real == pred, real == -1)
    fn_mask = torch.logical_and(real != pred, pred == -1)
    tp_mask = torch.logical_and(real <= pred, real != -1)
    fp_mask = torch.logical_or(
        torch.logical_and(torch.logical_and(real > pred, real != -1), pred != -1),
        torch.logical_and(pred != -1, real == -1),
    )

    TN = tn_mask.sum().item()
    FN = fn_mask.sum().item()
    TP = tp_mask.sum().item()
    FP = fp_mask.sum().item()

    FP_delay[tn_mask] = seq_len
    FP_delay[fn_mask] = seq_len
    FP_delay[tp_mask] = real[tp_mask]
    FP_delay[fp_mask] = pred[fp_mask]

    delay[tn_mask] = 0
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    delay[fp_mask] = 0

    assert (TN + TP + FN + FP) == len(real)

    return TN, FP, FN, TP, FP_delay, delay


def calculate_conf_matrix_margin(
    real: torch.Tensor, pred: torch.Tensor, margin: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real labels of change points
    :param pred: predicted labels (0 or 1) of change points
    :param margin: if |true_cp_idx - pred_cp_idx| <= margin, report TP
    :return: tuple of (TN, FP, FN, TP)
    """
    tn_mask_margin = torch.logical_and(real == pred, real == -1)
    fn_mask_margin = torch.logical_and(real != pred, pred == -1)

    tp_mask_margin = torch.logical_and(
        torch.logical_and(torch.abs(real - pred) <= margin, real != -1), pred != -1
    )

    fp_mask_margin = torch.logical_or(
        torch.logical_and(
            torch.logical_and(torch.abs(real - pred) > margin, real != -1), pred != -1
        ),
        torch.logical_and(pred != -1, real == -1),
    )

    TN_margin = tn_mask_margin.sum().item()
    FN_margin = fn_mask_margin.sum().item()
    TP_margin = tp_mask_margin.sum().item()
    FP_margin = fp_mask_margin.sum().item()

    assert (TN_margin + TP_margin + FN_margin + FP_margin) == len(
        real
    ), "Check TP, TN, FP, FN cases."

    return TN_margin, FP_margin, FN_margin, TP_margin


def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


def overlap(A: set, B: set):
    """Return the overlap (i.e. Jaccard index) of two sets.

    :param A: set #1
    :param B: set #2
    return Jaccard index of the 2 sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations: List[int], n_obs: int) -> List[set]:
    """Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.

    :param locations: idxs of the change points
    :param n_obs: length of the sequence
    :return partition of the sequence (list of sets with idxs)
    """
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(true_partitions: List[set], pred_partitions: List[set]) -> float:
    """Compute the covering of a true segmentation by a predicted segmentation.

    :param true_partitions: partition made by true CPs
    :param true_partitions: partition made by predicted CPs
    """
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))

    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(
            overlap(t_part, p_part) for p_part in pred_partitions
        )
    cover /= seq_len
    return cover


def calculate_cover(
    real_change_ind: List[int], predicted_change_ind: List[int], seq_len: int
) -> List[float]:
    """Calculate covering for a given sequence.

    :param real_change_ind: indexes of true CPs
    :param predicted_change_ind: indexes of predicted CPs
    :param seq_len: length of the sequence
    :return cover
    """
    covers = []

    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))
    return covers


def F1_score(confusion_matrix: Tuple[int, int, int, int]) -> float:
    """Calculate F1-score.

    :param confusion_matrix: tuple with elements of the confusion matrix
    :return: f1_score
    """
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


def calculate_metrics(
    true_labels: torch.Tensor, predictions: torch.Tensor, margin_list: List[int] = None
) -> Tuple[int, int, int, int, np.array, np.array, int]:
    """Calculate confusion matrix, detection delay, time to false alarms, covering.

    :param true_labels: true labels (0 or 1) of change points
    :param predictions: predicted labels (0 or 1) of change points
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
        - covering
    """
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)

    TN, FP, FN, TP, FP_delay, delay = calculate_errors(
        real_change_ind, predicted_change_ind, seq_len
    )
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    # add margin metrics
    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = {}, {}, {}, {}
        for margin in margin_list:
            TN_margin, FP_margin, FN_margin, TP_margin = calculate_conf_matrix_margin(
                real_change_ind, predicted_change_ind, margin
            )
            TN_margin_dict[margin] = TN_margin
            FP_margin_dict[margin] = FP_margin
            FN_margin_dict[margin] = FN_margin
            TP_margin_dict[margin] = TP_margin

    return (TN, FP, FN, TP, FP_delay, delay, cover), (
        TN_margin_dict,
        FP_margin_dict,
        FN_margin_dict,
        TP_margin_dict,
    )


def evaluate_metrics_on_set(
    test_out_bank: List[torch.Tensor],
    test_uncertainties_bank: List[torch.Tensor],
    test_labels_bank: List[torch.Tensor],
    threshold: float = 0.5,
    verbose: bool = True,
    device: str = "cuda",
    uncert_th: float = None,
    margin_list: List[int] = None,
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD.

    :param model: trained CPD model for evaluation
    :param test_loader: dataloader with test data
    :param threshold: alarm threshold (if change prob > threshold, report about a CP)
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param subseq_len: length of a subsequence (for 'weak_labels' baseline)
    :param device: 'cuda' or 'cpu'
    :param scale: scale factor (for KL-CPD and TSCP models)
    :param uncert_th: std threshold for CPD-with-rejection, set to 'None' if not rejection is needed
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: tuple of
        - TN, FP, FN, TP
        - mean time to a false alarm
        - mean detection delay
        - mean covering
    """
    FP_delays = []
    delays = []
    covers = []
    TN, FP, FN, TP = (0, 0, 0, 0)

    TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
        None,
        None,
        None,
        None,
    )

    if margin_list is not None:
        TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict = (
            {},
            {},
            {},
            {},
        )
        for margin in margin_list:
            TN_margin_dict[margin] = 0
            FP_margin_dict[margin] = 0
            FN_margin_dict[margin] = 0
            TP_margin_dict[margin] = 0

    with torch.no_grad():
        for test_out, test_uncertainties, test_labels in zip(
            test_out_bank, test_uncertainties_bank, test_labels_bank
        ):
            if test_uncertainties is not None and uncert_th is not None:
                cropped_outs = (test_out > threshold) & (test_uncertainties < uncert_th)

            else:
                cropped_outs = test_out > threshold

            (
                (tn, fp, fn, tp, FP_delay, delay, cover),
                (tn_margin_dict, fp_margin_dict, fn_margin_dict, tp_margin_dict),
            ) = calculate_metrics(test_labels, cropped_outs, margin_list)

            TN += tn
            FP += fp
            FN += fn
            TP += tp

            if margin_list is not None:
                for margin in margin_list:
                    TN_margin_dict[margin] += tn_margin_dict[margin]
                    FP_margin_dict[margin] += fp_margin_dict[margin]
                    FN_margin_dict[margin] += fn_margin_dict[margin]
                    TP_margin_dict[margin] += tp_margin_dict[margin]

            FP_delays.append(FP_delay.detach().cpu())
            delays.append(delay.detach().cpu())
            covers.extend(cover)

    mean_FP_delay = torch.cat(FP_delays).float().mean().item()
    mean_delay = torch.cat(delays).float().mean().item()
    mean_cover = np.mean(covers)

    if verbose:
        print(
            "TN: {}, FP: {}, FN: {}, TP: {}, DELAY:{}, FP_DELAY:{}, COVER: {}".format(
                TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover
            )
        )

    del FP_delays, delays, covers
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return (
        (TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover),
        (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
    )


# ------------------------------------------------------------------------------------------------------------#
#                                         Collect predictions                                                 #
# ------------------------------------------------------------------------------------------------------------#


def get_models_predictions(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scale: int = None,
    q: float = None,
    step: int = 1,
    alpha: float = 1.0,
) -> List[torch.Tensor]:
    """Get model's prediction.

    :param inputs: input data
    :param labels: true labels
    :param model: CPD model
    :param model_type: default "seq2seq" for BCE model, "klcpd" for KLCPD model
    :param device: device name
    :param scales: scale parameter for KL-CPD predictions
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: model's predictions
    """
    try:
        inputs = inputs.to(device)
    except AttributeError:
        inputs = [t.to(device) for t in inputs]

    true_labels = labels.to(device)

    if model_type in ["simple", "weak_labels"]:
        outs = []
        true_labels = []
        for batch_n in range(inputs.shape[0]):
            inp = inputs[batch_n].to(device)
            lab = labels[batch_n].to(device)

            if model_type == "simple":
                # TODO FIX
                # out = [model(inp[i].flatten().unsqueeze(0).float()).squeeze() for i in range(0, len(inp))]
                out = [
                    model(inp[:, i].unsqueeze(0).float()).squeeze()
                    for i in range(0, len(inp))
                ]

            elif (model_type == "weak_labels") and (subseq_len is not None):
                out_end = [
                    model(inp[i : i + subseq_len].flatten(1).unsqueeze(0).float())
                    for i in range(0, len(inp) - subseq_len)
                ]
                out = [torch.zeros(len(lab) - len(out_end), 1, device=device)]
                out.extend(out_end)
                out = torch.cat(out)
            true_labels += [lab]
            # TODO: fix
            try:
                outs.append(torch.stack(out))
            except:  # noqa: E722
                outs.append(out)
        outs = torch.stack(outs)
        true_labels = torch.stack(true_labels)

        # no uncertainty for these models
        uncertainties = None

    elif model_type == "tscp":
        # outs = tscp.get_tscp_output_scaled(model, inputs, model.window_1, model.window_2, scale=scale, step=step)
        # outs = tscp.get_tscp_output_scaled_padded(
        #    model, inputs, model.window_1, model.window_2, scale=scale, step=step, alpha=alpha
        # )
        # outs = tscp.get_tscp_output_padded(model, inputs, model.window_1, model.window_2, step=step)

        outs = tscp.get_tscp_output(
            model, inputs, model.window_1, model.window_2, step=step
        )
        outs = tscp.post_process_tscp_output(outs, scale=scale, alpha=alpha)
        uncertainties = None

    elif model_type == "kl_cpd":
        outs = klcpd.get_klcpd_output_scaled(
            model, inputs, model.window_1, model.window_2, scale=scale
        )
        uncertainties = None

    elif model_type == "ensemble":
        # take mean values and std (as uncertainty measure)
        outs, uncertainties = model.predict(inputs, scale=scale, step=step, alpha=alpha)

    elif model_type == "ensemble_all_models":
        outs = model.predict_all_models(inputs, scale=scale, step=step, alpha=alpha)
        uncertainties = None

    elif model_type == "ensemble_quantile":
        outs, uncertainties = model.get_quantile_predictions(inputs, q, scale=scale)

    elif model_type == "ensemble_max":
        outs, uncertainties = model.get_min_max_predictions(
            inputs, mode="max", scale=scale
        )

    elif model_type == "ensemble_min":
        outs, uncertainties = model.get_min_max_predictions(
            inputs, mode="min", scale=scale
        )

    elif model_type == "cusum_aggr":
        outs = model.predict(inputs, scale=scale, step=step, alpha=alpha)
        uncertainties = None

    elif model_type == "mmd_aggr":
        outs = model.predict(inputs, scale=scale, step=step, alpha=alpha)
        uncertainties = None

    elif model_type == "cusum_traject":
        outs = model.predict_cusum_trajectories(inputs, q)
        uncertainties = None

    elif model_type == "seq2seq":
        outs = model(inputs)
        uncertainties = None

    elif model_type == "fake_ensemble":
        outs, uncertainties = inputs[0], inputs[1]

    elif model_type == "fake_cusum":
        outs = model.fake_predict(inputs[0], inputs[1])
        uncertainties = None

    elif model_type == "fake_mmd":
        outs = model.fake_predict(inputs)
        uncertainties = None

    else:
        raise ValueError(f"Wrong model type {model_type}.")

    return outs, uncertainties, true_labels


def collect_model_predictions_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    verbose: bool = True,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scale: int = None,
    q: float = None,
    step: int = 1,
    alpha: float = 1.0,
):
    if model is not None:
        model.eval()
        model.to(device)

    test_out_bank, test_uncertainties_bank, test_labels_bank = [], [], []

    with torch.no_grad():
        if verbose:
            print("Collectting model's outputs")

        # collect model's predictions once and reuse them
        # for test_inputs, test_labels in tqdm(test_loader):
        for test_inputs, test_labels in tqdm(test_loader):
            test_out, test_uncertainties, test_labels = get_models_predictions(
                test_inputs,
                test_labels,
                model,
                model_type=model_type,
                subseq_len=subseq_len,
                device=device,
                scale=scale,
                q=q,
                step=step,
                alpha=alpha,
            )

            try:
                test_out = test_out.squeeze(2)
                test_uncertainties = test_uncertainties.squeeze(2)
            except:  # noqa: E722
                try:
                    test_out = test_out.squeeze(1)
                    test_uncertainties = test_uncertainties.squeeze(1)
                except:  # noqa: E722
                    test_out = test_out
                    test_uncertainties = test_uncertainties

            # in case of different sizes, crop start of labels sequence (for TS-CP)
            crop_size = test_labels.shape[-1] - test_out.shape[-1]
            test_labels = test_labels[:, crop_size:]

            test_out_bank.append(test_out.cpu())
            test_uncertainties_bank.append(
                test_uncertainties.cpu()
                if test_uncertainties is not None
                else test_uncertainties
            )
            test_labels_bank.append(test_labels.cpu())

    del test_labels, test_out, test_uncertainties, test_inputs
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return test_out_bank, test_uncertainties_bank, test_labels_bank


# ------------------------------------------------------------------------------------------------------------#
#                                      Evaluate classic baselines                                            #
# ------------------------------------------------------------------------------------------------------------#


def get_classic_baseline_predictions(
    dataloader: DataLoader,
    baseline_model: ClassicBaseline,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get predictions of a classic baseline model.

    :param dataloader: validation dataloader
    :param baseline_model: core model of a classic baseline (from ruptures package)
    :return: tuple of
        - predicted labels
        - true pabels
    """
    all_predictions = []
    all_labels = []
    for inputs, labels in dataloader:
        all_labels.append(labels)
        baseline_pred = baseline_model(inputs)
        all_predictions.append(baseline_pred)

    all_labels = torch.from_numpy(np.vstack(all_labels))
    all_predictions = torch.from_numpy(np.vstack(all_predictions))
    return all_predictions, all_labels


def classic_baseline_metrics(
    all_labels: torch.Tensor, all_preds: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """Calculate metrics for a classic baseline model.

    :param all_labels: tensor of true labels
    :param all_preds: tensor of predictions
    :param threshold: alarm threshold (=0.5 for classic models)
    :return: turple of metrics
        - best threshold for F1-score (always 0.5)
        - mean Time to a False Alarm
        - mean Detection Delay
        - None (no AUC metric for classic baselines)
        - best confusion matrix (number of TN, FP, FN and TP predictions)
        - F1-score
        - covering metric
        - best thresold for covering metric (always 0.5)
        - covering metric
    Note that we return some unnecessary values for consistency with our general evaluation pipeline.
    """
    TN, FP, FN, TP = (0, 0, 0, 0)
    TN, FP, FN, TP, FP_delay, delay, cover = calculate_metrics(
        all_labels, all_preds > threshold
    )
    f1 = F1_score((TN, FP, FN, TP))
    FP_delay = torch.mean(FP_delay.float()).item()
    delay = torch.mean(delay.float()).item()
    cover = np.mean(cover)
    return 0.5, FP_delay, delay, None, (TN, FP, FN, TP), f1, cover, 0.5, cover


def calculate_baseline_metrics(
    model: ClassicBaseline, val_dataloader: DataLoader, verbose: bool = False
) -> Tuple[float, float, float, None, Tuple[int], float, float, float, float]:
    """Calculate metrics for a classic baseline model.

    :param model: core model of a classic baseline (from ruptures package)
    :param val_dataloader: validation dataloader
    :param verbose: if true, print the metrics to the console
    :return: tuple of metrics (see 'classic_baseline_metrics' function)
    """
    pred, labels = get_classic_baseline_predictions(val_dataloader, model)
    metrics = classic_baseline_metrics(labels, pred)

    _, mean_FP_delay, mean_delay, _, (TN, FP, FN, TP), f1, cover, _, _ = metrics

    if verbose:
        print(
            f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}, DELAY: {mean_delay}, FP_DELAY:{mean_FP_delay}, F1:{f1}, COVER: {cover}"
        )
    return metrics


# ------------------------------------------------------------------------------------------------------------#
#                                              Save results                                                  #
# ------------------------------------------------------------------------------------------------------------#
def write_metrics_to_file(
    filename: str,
    metrics: tuple,
    seed: int,
    timestamp: str,
    comment: str = None,
) -> None:
    """Write metrics to a .txt file.

    :param filename: path to the .txt file
    :param metrics: tuple of metrics (output of the 'evaluation_pipeline' function)
    :param seed: initialization seed for the model under evaluation
    :param timestamp: timestamp indicating which model was evaluated
    :param comment: any additional information about the experiment
    """
    (
        (
            best_th_f1,
            best_time_to_FA,
            best_delay,
            auc,
            best_conf_matrix,
            best_f1,
            best_cover,
            best_th_cover,
            max_cover,
        ),
        (best_th_f1_margin_dict, max_f1_margin_dict),
    ) = metrics

    with open(filename, "a") as f:
        f.writelines("Comment: {}\n".format(comment))
        f.writelines("SEED: {}\n".format(seed))
        f.writelines("Timestamp: {}\n".format(timestamp))
        f.writelines("AUC: {}\n".format(auc))
        f.writelines(
            "Time to FA {}, delay detection {} for best-F1 threshold: {}\n".format(
                round(best_time_to_FA, 4), round(best_delay, 4), round(best_th_f1, 4)
            )
        )
        f.writelines(
            "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}\n".format(
                best_conf_matrix[0],
                best_conf_matrix[1],
                best_conf_matrix[2],
                best_conf_matrix[3],
                round(best_th_f1, 4),
            )
        )
        f.writelines(
            "Max F1 {}: for best-F1 threshold {}\n".format(
                round(best_f1, 4), round(best_th_f1, 4)
            )
        )
        f.writelines(
            "COVER {}: for best-F1 threshold {}\n".format(
                round(best_cover, 4), round(best_th_f1, 4)
            )
        )

        f.writelines(
            "Max COVER {}: for threshold {}\n".format(max_cover, best_th_cover)
        )
        if max_f1_margin_dict is not None:
            for margin, max_f1_margin in max_f1_margin_dict.items():
                f.writelines(
                    "Max F1 with margin {}: {} for threshold {}\n".format(
                        margin, max_f1_margin, best_th_f1_margin_dict[margin]
                    )
                )
        f.writelines(
            "----------------------------------------------------------------------\n"
        )


def dump_results(metrics_local: tuple, pickle_name: str) -> None:
    """Save result metrics as a .pickle file."""
    (
        best_th_f1,
        best_time_to_FA,
        best_delay,
        auc,
        best_conf_matrix,
        best_f1,
        best_cover,
        best_th_cover,
        max_cover,
    ) = metrics_local
    results = dict(
        best_th_f1=best_th_f1,
        best_time_to_FA=best_time_to_FA,
        best_delay=best_delay,
        auc=auc,
        best_conf_matrix=best_conf_matrix,
        best_f1=best_f1,
        best_cover=best_cover,
        best_th_cover=best_th_cover,
        max_cover=max_cover,
    )

    with Path(pickle_name).open("wb") as f:
        pickle.dump(results, f)


# ------------------------------------------------------------------------------------------------------------#
#                                          Rejection metrics                                                 #
# ------------------------------------------------------------------------------------------------------------#


def compute_binary_confusion_matrix(test_labels_batch, predictions_batch):
    TP = (
        ((test_labels_batch == 1) & (predictions_batch == 1)).astype(int).sum()
    )  # .sum(axis=1)
    FN = (
        ((test_labels_batch == 1) & (predictions_batch == 0)).astype(int).sum()
    )  # .sum(axis=1)
    FP = (
        ((test_labels_batch == 0) & (predictions_batch == 1)).astype(int).sum()
    )  # .sum(axis=1)
    TN = (
        ((test_labels_batch == 0) & (predictions_batch == 0)).astype(int).sum()
    )  # .sum(axis=1)
    return TP, FN, FP, TN


def evaluate_rejection_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scale: int = None,
    reject_uncert_th: float = np.inf,
    q: float = None,
) -> Tuple[float, float]:
    model.eval()
    model.to(device)

    TP_global = 0
    FN_global = 0
    FP_global = 0
    TN_global = 0

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_out, test_uncertainties, test_labels = get_models_predictions(
                test_inputs,
                test_labels,
                model,
                model_type=model_type,
                subseq_len=subseq_len,
                device=device,
                scale=scale,
                q=q,
            )

            try:
                test_out = test_out.squeeze(2)
                test_uncertainties = test_uncertainties.squeeze(2)
            except:  # noqa: E722
                try:
                    test_out = test_out.squeeze(1)
                    test_uncertainties = test_uncertainties.squeeze(1)
                except:  # noqa: E722
                    test_out = test_out
                    test_uncertainties = test_uncertainties

            cropped_outs = (test_out > threshold).to(int)

            # rejection class label = 2
            cropped_outs[test_uncertainties > reject_uncert_th] = 2
            test_labels[test_uncertainties > reject_uncert_th] = 2

            TP, FN, FP, TN = compute_binary_confusion_matrix(
                test_labels.cpu().numpy(), cropped_outs.cpu().numpy()
            )

            TP_global += TP
            FN_global += FN
            FP_global += FP
            TN_global += TN

            del test_labels
            del test_out

    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    EPS = 1e-9
    f1_score_global = 2.0 * TP_global / (2 * TP_global + FN_global + FP_global + EPS)
    accuracy_global = (TP_global + TN_global) / (
        TP_global + FN_global + FP_global + TN_global + EPS
    )

    return f1_score_global, accuracy_global


def calculate_rejection_curves(
    model: nn.Module,
    test_loader: DataLoader,
    # reject_uncert_th_list: List[float],
    rates_num: int,
    threshold: float = 0.5,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scale: int = None,
    q: float = None,
) -> Tuple[dict, dict]:
    f1_scores_dict = dict()
    acc_scores_dict = dict()

    reject_uncert_th_dict = evaluate_rej_rates_for_dataloader(
        model, test_loader, rates_num, model_type, subseq_len, device, scale, q
    )

    for rate, reject_uncert_th in tqdm(reject_uncert_th_dict.items()):
        mean_f1_score, mean_acc_score = evaluate_rejection_metrics_on_set(
            model=model,
            test_loader=test_loader,
            threshold=threshold,
            model_type=model_type,
            subseq_len=subseq_len,
            device=device,
            scale=scale,
            reject_uncert_th=reject_uncert_th,
            q=q,
        )

        f1_scores_dict[rate] = mean_f1_score
        acc_scores_dict[rate] = mean_acc_score

    return f1_scores_dict, acc_scores_dict


def evaluate_rej_rates_for_dataloader(
    model,
    dataloader,
    rates_num: int,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scale: int = None,
    q: float = None,
):
    model.eval()
    model.to(device)

    all_uncerts = []

    with torch.no_grad():
        for test_inputs, test_labels in dataloader:
            test_out, test_uncertainties, test_labels = get_models_predictions(
                test_inputs,
                test_labels,
                model,
                model_type=model_type,
                subseq_len=subseq_len,
                device=device,
                scale=scale,
                q=q,
            )
            try:
                test_out = test_out.squeeze(2)
                test_uncertainties = test_uncertainties.squeeze(2)
            except:  # noqa: E722
                try:
                    test_out = test_out.squeeze(1)
                    test_uncertainties = test_uncertainties.squeeze(1)
                except:  # noqa: E722
                    test_out = test_out
                    test_uncertainties = test_uncertainties

            all_uncerts.append(test_uncertainties)

    all_uncerts = torch.cat(all_uncerts)
    rej_rates = torch.linspace(0, 1, rates_num).to(device)
    rej_thresholds_dict = {
        rate: torch.quantile(all_uncerts, 1 - rate) for rate in rej_rates
    }

    return rej_thresholds_dict


# ------------------------------------------------------------------------------------------------------------#
#                                Check std for Ensenble / Bayes models                                       #
# ------------------------------------------------------------------------------------------------------------#


def compute_stds(
    model,
    test_dataloader: DataLoader,
    windows_list: List[int],
    scale: int = None,
    step: int = 1,
    alpha: float = 1.0,
    verbose: bool = True,
    device: str = "cpu",
) -> Dict[int, Tuple[List[float], List[float]]]:
    model.to(device)

    batch_bank, labels_bank, preds_std_bank = [], [], []
    if verbose:
        print("Computing model's outputs...")
    for batch, labels in tqdm(test_dataloader):
        batch = batch.to(device)
        _, preds_std = model.predict(batch, scale=scale, step=step, alpha=alpha)
        crop_start = labels.shape[1] - preds_std.shape[1]
        labels = labels[:, crop_start:]
        batch_bank.append(batch.detach().cpu())
        labels_bank.append(labels.detach().cpu())
        preds_std_bank.append(preds_std.detach().cpu())

    res_dict = {}
    for window in windows_list:
        if verbose:
            print(f"Window: {window}")
        normal_stds_list = []
        cp_stds_list = []
        for _, labels, preds_std in zip(batch_bank, labels_bank, preds_std_bank):
            cp_idxs_batch = labels.cpu().argmax(axis=1)
            for cp_idx, std_series in zip(cp_idxs_batch, preds_std):
                if cp_idx == 0:
                    normal_stds_list.append(std_series.mean().item())
                elif cp_idx < window + 1:
                    cp_slice = std_series[: cp_idx + window]
                    cp_stds_list.append(cp_slice.mean().item())
                else:
                    cp_slice = std_series[cp_idx - window : cp_idx + window]
                    normal_slice = std_series[: cp_idx - window]
                    cp_stds_list.append(cp_slice.mean().item())
                    normal_stds_list.append(normal_slice.mean().item())
        if verbose:
            check_stds_equality(normal_stds_list, cp_stds_list)
            print("-" * 50)

        res_dict[window] = (normal_stds_list, cp_stds_list)
    return res_dict


def check_stds_equality(
    normal_stds_list: List[float], cp_stds_list: List[float], permutations: int = 10000
) -> None:
    print("CP stds list:")
    print(f"Mean = {np.mean(cp_stds_list)}, number is {len(cp_stds_list)}")

    print("Normal stds list:")
    print(f"Mean = {np.mean(normal_stds_list)}, number is {len(normal_stds_list)}")

    p_perm = ttest_ind(
        cp_stds_list, normal_stds_list, equal_var=False, permutations=permutations
    ).pvalue
    p_an = ttest_ind(cp_stds_list, normal_stds_list, equal_var=False).pvalue

    print(f"p_val analytical = {p_an}, p_val permutational = {p_perm}")

    if p_perm < 0.05 and p_an < 0.05:
        print("Stds are not statistically equal")
    else:
        print("No conclusion")
