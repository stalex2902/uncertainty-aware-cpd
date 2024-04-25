import itertools
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from src.datasets.datasets import AllModelsOutputDataset, OutputDataset
from src.ensembles.ensembles import (
    CusumEnsembleCPDModel,
    DistanceEnsembleCPDModel,
)
from src.metrics.metrics_utils import (
    F1_score,
    area_under_graph,
    collect_model_predictions_on_set,
    estimate_threshold_range,
    evaluate_metrics_on_set,
    write_metrics_to_file,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluation_pipeline(
    model: pl.LightningModule,
    test_dataloader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
    verbose: bool = False,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    scale: int = None,
    uncert_th: float = None,
    q: float = None,
    margin_list: List[int] = None,
    step: int = 1,
    alpha: float = 1.0,
) -> Tuple[Tuple[float], dict, dict]:
    """Evaluate trained CPD model.

    :param model: trained CPD model to be evaluated
    :param test_dataloader: test data for evaluation
    :param threshold_list: listh of alarm thresholds
    :param device: 'cuda' or 'cpu'
    :param verbose: if True, print the results
    :param model_type: type of the model ('seq2seq', 'kl_cpd', 'tscp', baselines)
    :param subseq_len: subsequence length (for 'weak_labels' baseline)
    :param scale: scale factor (for KL-CPD and TSCP models)
    :param uncert_th: std threshold for CPD-with-rejection, set to 'None' if not rejection is needed
    :param q: probability for quantile-based predictions of the EnsembleCPDModel, set to 'None' is no ensemble is used
    :return: tuple of
        - threshold th_1 corresponding to the maximum F1-score
        - mean time to a False Alarm corresponding to th_1
        - mean Detection Delay corresponding to th_1
        - Area under the Detection Curve
        - number of TN, FP, FN, TP corresponding to th_1
        - value of Covering corresponding to th_1
        - threshold th_2 corresponding to the maximum Covering metric
        - maximum value of Covering
    """
    try:
        model.to(device)
        model.eval()
    except AttributeError:
        print("Cannot move model to device")

    (
        test_out_bank,
        test_uncertainties_bank,
        test_labels_bank,
    ) = collect_model_predictions_on_set(
        model=model,
        test_loader=test_dataloader,
        verbose=verbose,
        model_type=model_type,
        subseq_len=subseq_len,
        device=device,
        scale=scale,
        q=q,
        step=step,
        alpha=alpha,
    )

    cover_dict = {}
    f1_dict = {}

    if margin_list is not None:
        final_f1_margin_dict = {}

    delay_dict = {}
    fp_delay_dict = {}
    confusion_matrix_dict = {}

    if model_type == "cusum_aggr":
        threshold_list = [0.5]
        if verbose and len(threshold_list) > 1:
            print("No need in threshold list for CUSUM. Take threshold = 0.5.")

    for threshold in threshold_list:
        if margin_list is not None:
            final_f1_margin_dict[threshold] = {}

        (
            (TN, FP, FN, TP, mean_delay, mean_fp_delay, cover),
            (TN_margin_dict, FP_margin_dict, FN_margin_dict, TP_margin_dict),
        ) = evaluate_metrics_on_set(
            test_out_bank=test_out_bank,
            test_uncertainties_bank=test_uncertainties_bank,
            test_labels_bank=test_labels_bank,
            threshold=threshold,
            verbose=verbose,
            device=device,
            uncert_th=uncert_th,
            margin_list=margin_list,
        )

        confusion_matrix_dict[threshold] = (TN, FP, FN, TP)
        delay_dict[threshold] = mean_delay
        fp_delay_dict[threshold] = mean_fp_delay

        cover_dict[threshold] = cover
        f1_dict[threshold] = F1_score((TN, FP, FN, TP))

        if margin_list is not None:
            f1_margin_dict = {}
            for margin in margin_list:
                (TN_margin, FP_margin, FN_margin, TP_margin) = (
                    TN_margin_dict[margin],
                    FP_margin_dict[margin],
                    FN_margin_dict[margin],
                    TP_margin_dict[margin],
                )
                f1_margin_dict[margin] = F1_score(
                    (TN_margin, FP_margin, FN_margin, TP_margin)
                )
            final_f1_margin_dict[threshold] = f1_margin_dict

    # fix dict structure
    if margin_list is not None:
        final_f1_margin_dict_fixed = {}
        for margin in margin_list:
            f1_scores_for_margin_dict = {}
            for threshold in threshold_list:
                f1_scores_for_margin_dict[threshold] = final_f1_margin_dict[threshold][
                    margin
                ]
            final_f1_margin_dict_fixed[margin] = f1_scores_for_margin_dict

    if model_type == "cusum_aggr":
        auc = None
    else:
        auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))

    # Conf matrix and F1
    best_th_f1 = max(f1_dict, key=f1_dict.get)

    best_conf_matrix = (
        confusion_matrix_dict[best_th_f1][0],
        confusion_matrix_dict[best_th_f1][1],
        confusion_matrix_dict[best_th_f1][2],
        confusion_matrix_dict[best_th_f1][3],
    )
    best_f1 = f1_dict[best_th_f1]

    # Cover
    best_cover = cover_dict[best_th_f1]

    best_th_cover = max(cover_dict, key=cover_dict.get)
    max_cover = cover_dict[best_th_cover]

    if margin_list is not None:
        max_f1_margins_dict = {}
        max_th_f1_margins_dict = {}
        for margin in margin_list:
            curr_max_th_f1_margin = max(
                final_f1_margin_dict_fixed[margin],
                key=final_f1_margin_dict_fixed[margin].get,
            )
            max_th_f1_margins_dict[margin] = curr_max_th_f1_margin
            max_f1_margins_dict[margin] = final_f1_margin_dict_fixed[margin][
                curr_max_th_f1_margin
            ]
    else:
        max_f1_margins_dict, max_th_f1_margins_dict = None, None

    # Time to FA, detection delay
    best_time_to_FA = fp_delay_dict[best_th_f1]
    best_delay = delay_dict[best_th_f1]

    if verbose:
        print("AUC:", round(auc, 4) if auc is not None else auc)
        print(
            "Time to FA {}, delay detection {} for best-F1 threshold: {}".format(
                round(best_time_to_FA, 4), round(best_delay, 4), round(best_th_f1, 4)
            )
        )
        print(
            "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}".format(
                best_conf_matrix[0],
                best_conf_matrix[1],
                best_conf_matrix[2],
                best_conf_matrix[3],
                round(best_th_f1, 4),
            )
        )
        print(
            "Max F1 {}: for best-F1 threshold {}".format(
                round(best_f1, 4), round(best_th_f1, 4)
            )
        )
        print(
            "COVER {}: for best-F1 threshold {}".format(
                round(best_cover, 4), round(best_th_f1, 4)
            )
        )

        print(
            "Max COVER {}: for threshold {}".format(
                round(cover_dict[max(cover_dict, key=cover_dict.get)], 4),
                round(max(cover_dict, key=cover_dict.get), 4),
            )
        )
        if margin_list is not None:
            for margin in margin_list:
                print(
                    "Max F1 with margin {}: {} for best threshold {}".format(
                        margin,
                        round(max_f1_margins_dict[margin], 4),
                        round(max_th_f1_margins_dict[margin], 4),
                    )
                )

    return (
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
        (max_th_f1_margins_dict, max_f1_margins_dict),
        delay_dict,
        fp_delay_dict,
    )


def evaluate_cusum_ensemble_model(
    ens_model,
    cusum_threshold_number: List[float],
    output_dataloader: DataLoader,
    margin_list: List[int],
    cusum_mode: str,
    conditional: bool = False,
    global_sigma: float = None,
    lambda_null: float = None,
    lambda_inf: float = None,
    half_wnd: int = None,
    var_coeff: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
    write_metrics_filename: str = None,
    min_th_quant: float = 0.1,
    max_th_quant: float = 0.9,
):
    res_dict = {}
    best_th = None
    best_f1_global = 0

    # # TODO: repeatition!!!
    test_cusum_model = CusumEnsembleCPDModel(
        ens_model=ens_model,
        global_sigma=global_sigma,
        cusum_threshold=0.0,
        cusum_mode=cusum_mode,
        conditional=conditional,
        lambda_null=lambda_null,
        lambda_inf=lambda_inf,
        half_wnd=half_wnd,
        var_coeff=var_coeff,
    )

    (out_series_batch, out_series_std_batch), _ = next(iter(output_dataloader))

    min_th, max_th = estimate_threshold_range(
        model=test_cusum_model,
        out_series_batch=out_series_batch,
        out_series_std_batch=out_series_std_batch,
        quant_min=min_th_quant,
        quant_max=max_th_quant,
    )

    print(f"Threshold range: ({min_th}, {max_th})")

    cusum_threshold_list = np.linspace(min_th, max_th, cusum_threshold_number)

    for cusum_th in tqdm(cusum_threshold_list):
        cusum_model = CusumEnsembleCPDModel(
            ens_model=ens_model,
            global_sigma=global_sigma,
            cusum_threshold=cusum_th,
            cusum_mode=cusum_mode,
            conditional=conditional,
            lambda_null=lambda_null,
            lambda_inf=lambda_inf,
            half_wnd=half_wnd,
            var_coeff=var_coeff,
        )

        metrics_local, (max_th_f1_margins_dict, max_f1_margins_dict), _, _ = (
            evaluation_pipeline(
                model=cusum_model,
                test_dataloader=output_dataloader,
                threshold_list=[0.5],
                device=device,
                model_type="fake_cusum",
                verbose=False,
                margin_list=margin_list,
            )
        )

        if write_metrics_filename is not None:
            write_metrics_to_file(
                filename=write_metrics_filename,
                metrics=(metrics_local, (max_th_f1_margins_dict, max_f1_margins_dict)),
                seed=None,
                timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
                comment=f"mode_{cusum_mode}_cond_{conditional}_cusum_th_{cusum_th}",
            )

        (
            _,
            best_time_to_FA,
            best_delay,
            audc,
            _,
            best_f1,
            best_cover,
            _,
            max_cover,
        ) = metrics_local

        res_dict[cusum_th] = (
            audc,
            best_time_to_FA,
            best_delay,
            best_f1,
            best_cover,
            max_cover,
            max_f1_margins_dict,
        )

        if best_f1 > best_f1_global:
            best_f1_global = best_f1
            best_th = cusum_th

    if verbose:
        (
            _audc,
            _best_time_to_FA,
            _best_delay,
            _best_f1,
            _best_cover,
            _max_cover,
            _max_f1_margins_dict,
        ) = res_dict[best_th]
        _audc = np.round(_audc, 4)
        _best_time_to_FA = np.round(_best_time_to_FA, 4)
        _best_delay = np.round(_best_delay, 4)
        _best_f1 = np.round(_best_f1, 4)
        _best_cover = np.round(_best_cover, 4)
        _max_cover = np.round(_max_cover, 4)

        print(f"Results for best threshold = {best_th}")
        print(
            f"AUDC: {_audc}, Time to FA: {_best_time_to_FA}, DD: {_best_delay}, F1: {_best_f1}, Cover: {_best_cover}, Max Cover: {_max_cover}"
        )
        for margin in margin_list:
            print(
                f"Max F1 with margin {margin}: {np.round(_max_f1_margins_dict[margin], 4)}"
            )
    return res_dict


"""
def evaluate_cusum_bayes_model(
    cusum_threshold_list: List[float],
    output_dataloader: DataLoader,
    margin_list: List[int],
    args: Dict,
    model: Any,
    train_dataset: Dataset,
    test_dataset: Dataset,
    kl_coeff: float,
    n_samples: int,
    save_path: str,
    cusum_mode: str,
    conditional: bool = False,
    global_sigma: float = None,
    lambda_null: float = None,
    lambda_inf: float = None,
    half_wnd: int = None,
    var_coeff: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
    write_metrics_filename: str = None,
):
    res_dict = {}
    best_th = None
    best_f1_global = 0

    for cusum_th in tqdm(cusum_threshold_list):
        cusum_model = CusumBayesCPDModel(
            args=args,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            kl_coeff=kl_coeff,
            n_samples=n_samples,
            global_sigma=global_sigma,
            cusum_threshold=cusum_th,
            cusum_mode=cusum_mode,
            conditional=conditional,
            lambda_null=lambda_null,
            lambda_inf=lambda_inf,
            half_wnd=half_wnd,
            var_coeff=var_coeff,
        )
        cusum_model.model.load_state_dict(torch.load(save_path))

        # if verbose:
        #    print("cusum_th:", cusum_th)

        metrics_local, (_, max_f1_margins_dict), _, _ = evaluation_pipeline(
            model=cusum_model,
            test_dataloader=output_dataloader,
            threshold_list=[0.5],
            device=device,
            model_type="fake_cusum",
            verbose=True,
            margin_list=margin_list,
        )

        if write_metrics_filename is not None:
            write_metrics_to_file(
                filename=write_metrics_filename,
                metrics=(metrics_local, (_, max_f1_margins_dict)),
                seed=None,
                timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
                comment=f"mode_{cusum_mode}_cond_{conditional}_cusum_th_{cusum_th}",
            )

        (
            _,
            best_time_to_FA,
            best_delay,
            audc,
            _,
            best_f1,
            best_cover,
            _,
            max_cover,
        ) = metrics_local
        res_dict[cusum_th] = (
            audc,
            best_time_to_FA,
            best_delay,
            best_f1,
            best_cover,
            max_cover,
            max_f1_margins_dict,
        )

        if best_f1 > best_f1_global:
            best_f1_global = best_f1
            best_th = cusum_th

    if verbose:
        (
            _audc,
            _best_time_to_FA,
            _best_delay,
            _best_f1,
            _best_cover,
            _max_cover,
            _max_f1_margins_dict,
        ) = res_dict[best_th]
        _audc = np.round(_audc, 4)
        _best_time_to_FA = np.round(_best_time_to_FA, 4)
        _best_delay = np.round(_best_delay, 4)
        _best_f1 = np.round(_best_f1, 4)
        _best_cover = np.round(_best_cover, 4)
        _max_cover = np.round(_max_cover, 4)

        print(f"Results for best threshold = {best_th}")
        print(
            f"AUDC: {_audc}, Time to FA: {_best_time_to_FA}, DD: {_best_delay}, F1: {_best_f1}, Cover: {_best_cover}, Max Cover: {_max_cover}"
        )
        for margin in margin_list:
            print(
                f"Max F1 with margin {margin}: {np.round(_max_f1_margins_dict[margin], 4)}"
            )
    return res_dict
"""


def all_cusums_evaluation_pipeline(
    ens_model,
    threshold_number: int,
    test_dataloader: DataLoader,
    margin_list: List[int],
    var_coeff: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
    write_metrics_filename: str = None,
    min_th_quant: float = 0.1,
    max_th_quant: float = 0.9,
):
    test_out_bank, test_uncertainties_bank, test_labels_bank = (
        collect_model_predictions_on_set(
            ens_model, test_dataloader, model_type="ensemble", device=device
        )
    )

    out_dataset = OutputDataset(
        test_out_bank, test_uncertainties_bank, test_labels_bank
    )
    out_dataloader = DataLoader(
        out_dataset, batch_size=128, shuffle=True
    )  # batch size does not matter, shuffle to get a diverse batch

    normal_sigma, cp_sigma, half_window = ens_model.args["cusum"].values()
    global_sigma = normal_sigma
    lambda_null = 1.0 / cp_sigma**2
    lambda_inf = 1.0 / normal_sigma**2

    all_results = {}

    for cusum_mode in ["old", "correct", "new_criteria"]:
        for conditional in [False, True]:
            if cusum_mode == "old" and not conditional:
                continue

            if verbose:
                print(
                    f"Evaluating CUSUM model with cusum_mode = {cusum_mode} and conditional = {conditional}"
                )

            res_dict = evaluate_cusum_ensemble_model(
                ens_model=ens_model,
                cusum_threshold_number=threshold_number,
                output_dataloader=out_dataloader,
                margin_list=margin_list,
                cusum_mode=cusum_mode,
                conditional=conditional,
                global_sigma=global_sigma,
                lambda_null=lambda_null,
                lambda_inf=lambda_inf,
                half_wnd=half_window,
                var_coeff=var_coeff,
                device="cpu",  # use 'fake cusum'
                verbose=verbose,
                write_metrics_filename=write_metrics_filename,
                min_th_quant=min_th_quant,
                max_th_quant=max_th_quant,
            )

            all_results[(cusum_mode, conditional)] = res_dict

    return all_results


def evaluate_distance_ensemble_model(
    ens_model,
    threshold_list: List[float],
    output_dataloader: DataLoader,
    margin_list: List[int],
    window_size: int,
    anchor_window_type: str = "start",
    distance: str = "wasserstein",
    kernel: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
    write_metrics_filename: str = None,
):
    res_dict = {}
    best_th = None
    best_f1_global = 0

    for th in tqdm(threshold_list):
        model = DistanceEnsembleCPDModel(
            ens_model=ens_model,
            threshold=th,
            kernel=kernel,
            window_size=window_size,
            anchor_window_type=anchor_window_type,
            distance=distance,
        )

        metrics_local, (_, max_f1_margins_dict), _, _ = evaluation_pipeline(
            model=model,
            test_dataloader=output_dataloader,
            threshold_list=[0.5],
            device=device,
            model_type="fake_mmd",
            verbose=False,
            margin_list=margin_list,
        )

        if write_metrics_filename is not None:
            write_metrics_to_file(
                filename=write_metrics_filename,
                metrics=(metrics_local, (_, max_f1_margins_dict)),
                seed=None,
                timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
                comment=f"kernel_{kernel}_window_size_{window_size}_th_{th}",
            )

        (
            _,
            best_time_to_FA,
            best_delay,
            audc,
            _,
            best_f1,
            best_cover,
            _,
            max_cover,
        ) = metrics_local
        res_dict[th] = (
            audc,
            best_time_to_FA,
            best_delay,
            best_f1,
            best_cover,
            max_cover,
            max_f1_margins_dict,
        )

        if best_f1 > best_f1_global:
            best_f1_global = best_f1
            best_th = th

    if verbose:
        (
            _audc,
            _best_time_to_FA,
            _best_delay,
            _best_f1,
            _best_cover,
            _max_cover,
            _max_f1_margins_dict,
        ) = res_dict[best_th]

        _audc = np.round(_audc, 4)
        _best_time_to_FA = np.round(_best_time_to_FA, 4)
        _best_delay = np.round(_best_delay, 4)
        _best_f1 = np.round(_best_f1, 4)
        _best_cover = np.round(_best_cover, 4)
        _max_cover = np.round(_max_cover, 4)

        print(f"Results for best threshold = {best_th}")
        print(
            f"AUDC: {_audc}, Time to FA: {_best_time_to_FA}, DD: {_best_delay}, F1: {_best_f1}, Cover: {_best_cover}, Max Cover: {_max_cover}"
        )
        for margin in margin_list:
            print(
                f"Max F1 with margin {margin}: {np.round(_max_f1_margins_dict[margin], 4)}"
            )
    return res_dict, best_th


def all_distances_evaluation_pipeline(
    ens_model,
    test_dataloader,
    distance="wasserstein",
    device="cpu",
    verbose=True,
    window_size_list=[1, 2, 3],
    anchor_window_type_list=["start", "prev"],
    threshold_list=np.linspace(0, 1, 50),
):
    test_out_bank, _, test_labels_bank = collect_model_predictions_on_set(
        ens_model,
        test_dataloader,
        model_type="ensemble_all_models",
        device=device,
        verbose=verbose,
    )

    out_dataset = AllModelsOutputDataset(test_out_bank, test_labels_bank)

    out_dataloader = DataLoader(
        out_dataset, batch_size=128, shuffle=False
    )  # batch size does not matter

    res_dict = {}

    for window_size, anchor_window_type in itertools.product(
        window_size_list, anchor_window_type_list
    ):
        if verbose:
            print(
                f"window_size = {window_size}, anchor_window_type = {anchor_window_type}"
            )

        res, best_th = evaluate_distance_ensemble_model(
            ens_model=ens_model,
            threshold_list=threshold_list,
            output_dataloader=out_dataloader,
            margin_list=[1, 2, 4],
            window_size=window_size,
            anchor_window_type=anchor_window_type,
            distance=distance,
            device="cpu",
            verbose=verbose,
        )

        res_dict[(window_size, anchor_window_type)] = res[best_th]

    return res_dict
