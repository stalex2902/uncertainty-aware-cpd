from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from src.datasets.datasets import CPDDatasets
from src.ensembles.bayes_models import BaseBayesRnn, BayesCPDModel
from src.ensembles.ensembles import EnsembleCPDModel
from src.metrics.evaluation_pipelines import (
    all_cusums_evaluation_pipeline,
    all_distances_evaluation_pipeline,
    evaluation_pipeline,
)
from src.metrics.metrics_utils import compute_stds
from src.utils.calibration import calibrate_all_models_in_ensemble
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import DataLoader


def evaluate_ensemble(
    experiments_name: str,
    model_type: str,
    loss_type: Optional[str] = None,
    is_ensemble: bool = True,
    n_models: int = 10,
    calibrate: bool = False,
    ensemble_num: int = 1,
    threshold_number: int = 300,
    evaluation_components: List[str] = [
        "mean",
        "min",
        "max",
        "median",
        "distances",
        "cusums",
    ],
    seed: int = 42,
    kl_coeff: float = 0.2,
    prior_mu: float = 0.0,
    prior_sigma: float = 0.1,
    verbose: bool = True,
    save_df: bool = False,
):
    if not is_ensemble:
        assert (
            model_type == "seq2seq" and loss_type == "bce"
        ), "We have only BCE Bayes models"
        assert not calibrate, " No calibration for Bayes models"
        assert experiments_name in [
            "synthetic_1D",
            "human_activity",
        ], f"We do have BCE Bayes model for {experiments_name}"

    if verbose:
        print("Loading datasets and models")

    if experiments_name not in ["explosion", "road_accidents"]:
        path_to_config = "configs/" + experiments_name + "_" + model_type + ".yaml"
        device = "cpu"
        min_th_quant = 0.1
        max_th_quant = 0.9
    else:
        path_to_config = "configs/" + "video" + "_" + model_type + ".yaml"
        device = "cuda:0"
        min_th_quant = 0.0
        max_th_quant = 1.0

    with open(path_to_config, "r") as f:
        args_config = yaml.safe_load(f.read())

    args_config["experiments_name"] = experiments_name
    args_config["model_type"] = model_type
    args_config["num_workers"] = 2

    if model_type == "seq2seq":
        if loss_type == "bce":
            args_config["loss_type"] = "bce"
        elif loss_type == "indid":
            args_config["loss_type"] = "indid"
        else:
            raise ValueError(f"Wrong loss type {loss_type}")

        scale, step, alpha = None, 1, 1.0

        if experiments_name == "synthetic_1D":
            if is_ensemble:
                if loss_type == "bce":
                    path_to_models_folder = (
                        f"saved_models/bce/synthetic_1D/full_sample/ens_{ensemble_num}"
                    )
                elif loss_type == "indid":
                    path_to_models_folder = (
                        f"saved_models/indid/synthetic_1D/ens_{ensemble_num}"
                    )
            else:
                path_to_models_folder = (
                    f"saved_models/total_bayes/synthetic_1D/ens_{ensemble_num}"
                )

        elif experiments_name == "human_activity":
            if is_ensemble:
                if loss_type == "bce":
                    path_to_models_folder = f"saved_models/bce/human_activity/full_sample/ens_{ensemble_num}"
                elif loss_type == "indid":
                    path_to_models_folder = (
                        f"saved_models/indid/human_activity/ens_{ensemble_num}"
                    )
            else:
                path_to_models_folder = (
                    f"saved_models/total_bayes/human_activity/ens_{ensemble_num}"
                )

        elif experiments_name == "explosion":
            if loss_type == "bce":
                path_to_models_folder = f"saved_models/bce/explosion/layer_norm/train_anomaly_num_155/ens_{ensemble_num}"
                args_config["model"]["ln_type"] = "after"
            elif loss_type == "indid":
                path_to_models_folder = f"saved_models/indid/explosion/layer_norm_before/ens_{ensemble_num}"  # !!!!!!!!!!!!1
                args_config["model"]["ln_type"] = "before"

        elif experiments_name == "road_accidents":
            if loss_type == "bce":
                path_to_models_folder = (
                    f"saved_models/bce/road_accidents/layer_norm/ens_{ensemble_num}"
                )
                args_config["model"]["ln_type"] = "after"

            elif loss_type == "indid":
                path_to_models_folder = f"saved_models/indid/road_accidents/layer_norm_before/ens_{ensemble_num}"
                args_config["model"]["ln_type"] = "before"
        else:
            raise ValueError(f"Wrong experiments name {experiments_name}")

    elif model_type == "tscp":
        scale, step, alpha = args_config["predictions"].values()

        if experiments_name == "synthetic_1D":
            path_to_models_folder = f"saved_models/tscp/synthetic_1D/window_{args_config['model']['window']}/ens_{ensemble_num}"

        elif experiments_name == "human_activity":
            path_to_models_folder = f"saved_models/tscp/human_activity/window_{args_config['model']['window']}/ens_{ensemble_num}"

        elif experiments_name == "yahoo":
            path_to_models_folder = f"saved_models/tscp/yahoo/window_{args_config['model']['window']}/ens_{ensemble_num}"

        else:
            raise ValueError(f"Wrong experiments name {experiments_name}")

    elif model_type == "ts2vec":
        if experiments_name == "yahoo":
            path_to_models_folder = f"saved_models/ts2vec/yahoo/ens_{ensemble_num}"
        elif experiments_name == "human_activity":
            path_to_models_folder = (
                f"saved_models/ts2vec/human_activity/ens_{ensemble_num}"
            )
        else:
            raise ValueError(f"Wrong experiments name {experiments_name}")

    else:
        raise ValueError(f"Wrong model type {model_type}")

    fix_seeds(seed)

    train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()

    test_dataloader = DataLoader(
        test_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
    )

    if is_ensemble:
        ens_model = EnsembleCPDModel(args_config, n_models=n_models)
        ens_model.load_models_list(path_to_models_folder)
    else:
        model = BaseBayesRnn(
            input_size=args_config["model"]["input_size"],
            hidden_dim=args_config["model"]["hidden_dim"],
            n_layers=args_config["model"]["n_layers"],
            drop_prob=args_config["model"]["drop_prob"],
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
        )

        ens_model = BayesCPDModel(
            args=args_config,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            kl_coeff=kl_coeff,
            n_samples=n_models,
            seed=ensemble_num,
        )

        try:
            ens_model.load_state_dict(
                torch.load(
                    path_to_models_folder
                    + "/"
                    + f"{experiments_name}_bayes_{model_type}_model_num_{ensemble_num}.pth"
                )
            )
        except RuntimeError:
            ens_model.model.load_state_dict(
                torch.load(
                    path_to_models_folder
                    + "/"
                    + f"{experiments_name}_bayes_{model_type}_model_num_{ensemble_num}.pth"
                )
            )

    if calibrate:
        if verbose:
            print("Calibrating the models using Beta calibration")

        _, val_dataset = train_test_split(
            train_dataset, test_size=0.3, random_state=seed
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
        )

        _ = calibrate_all_models_in_ensemble(
            ens_model,
            val_dataloader,
            cal_type="beta",
            verbose=verbose,
            device=device,
        )

    ens_model.eval()

    columns = [
        "Model name",
        "AUDC",
        "Time to FA",
        "DD",
        "Max F1",
        "Cover",
        "Max Cover",
        "Max F1, m1",
        "Max F1, m2",
        "Max F1, m3",
        "params",
    ]
    results_df = pd.DataFrame(columns=columns)

    threshold_list_ens = np.linspace(-5, 5, threshold_number)
    threshold_list_ens = 1 / (1 + np.exp(-threshold_list_ens))
    threshold_list_ens = [-0.001] + list(threshold_list_ens) + [1.001]

    if "single" in evaluation_components:
        pass  # TODO: add single model evaluation

    if "mean" in evaluation_components:
        if verbose:
            print("Evaluating mean ensemble model:")

        res_mean = evaluation_pipeline(
            ens_model,
            test_dataloader,
            threshold_list_ens,
            device=device,
            model_type="ensemble",
            verbose=verbose,
            margin_list=args_config["evaluation"]["margin_list"],
        )

        # extract metrics for mean ensemble
        metrics, (_, max_f1_margins_dic), _, _ = res_mean
        best_th_f1, time_to_FA, delay, auc, _, f1, cover, _, max_cover = metrics
        f1_m1, f1_m2, f1_m3 = max_f1_margins_dic.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Mean",
                "AUDC": auc,
                "Time to FA": time_to_FA,
                "DD": delay,
                "Max F1": f1,
                "Cover": cover,
                "Max Cover": max_cover,
                "Max F1, m1": f1_m1,
                "Max F1, m2": f1_m2,
                "Max F1, m3": f1_m3,
                "params": f"th = {best_th_f1}",
            },
            ignore_index=True,
        )

    if "min" in evaluation_components:
        if verbose:
            print("Evaluating min ensemble model:")

        res_mean = evaluation_pipeline(
            ens_model,
            test_dataloader,
            threshold_list_ens,
            device=device,
            model_type="ensemble_quantile",
            q=0.0,
            verbose=verbose,
            margin_list=args_config["evaluation"]["margin_list"],
        )

        # extract metrics for min ensemble
        metrics, (_, max_f1_margins_dic), _, _ = res_mean
        best_th_f1, time_to_FA, delay, auc, _, f1, cover, _, max_cover = metrics
        f1_m1, f1_m2, f1_m3 = max_f1_margins_dic.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Min",
                "AUDC": auc,
                "Time to FA": time_to_FA,
                "DD": delay,
                "Max F1": f1,
                "Cover": cover,
                "Max Cover": max_cover,
                "Max F1, m1": f1_m1,
                "Max F1, m2": f1_m2,
                "Max F1, m3": f1_m3,
                "params": f"th = {best_th_f1}",
            },
            ignore_index=True,
        )

    if "max" in evaluation_components:
        if verbose:
            print("Evaluating max ensemble model:")

        res_mean = evaluation_pipeline(
            ens_model,
            test_dataloader,
            threshold_list_ens,
            device=device,
            model_type="ensemble_quantile",
            q=1.0,
            verbose=verbose,
            margin_list=args_config["evaluation"]["margin_list"],
        )

        # extract metrics for min ensemble
        metrics, (_, max_f1_margins_dic), _, _ = res_mean
        best_th_f1, time_to_FA, delay, auc, _, f1, cover, _, max_cover = metrics
        f1_m1, f1_m2, f1_m3 = max_f1_margins_dic.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Max",
                "AUDC": auc,
                "Time to FA": time_to_FA,
                "DD": delay,
                "Max F1": f1,
                "Cover": cover,
                "Max Cover": max_cover,
                "Max F1, m1": f1_m1,
                "Max F1, m2": f1_m2,
                "Max F1, m3": f1_m3,
                "params": f"th = {best_th_f1}",
            },
            ignore_index=True,
        )

    if "median" in evaluation_components:
        if verbose:
            print("Evaluating median ensemble model:")

        res_mean = evaluation_pipeline(
            ens_model,
            test_dataloader,
            threshold_list_ens,
            device=device,
            model_type="ensemble_quantile",
            q=0.5,
            verbose=verbose,
            margin_list=args_config["evaluation"]["margin_list"],
        )

        # extract metrics for min ensemble
        metrics, (_, max_f1_margins_dic), _, _ = res_mean
        best_th_f1, time_to_FA, delay, auc, _, f1, cover, _, max_cover = metrics
        f1_m1, f1_m2, f1_m3 = max_f1_margins_dic.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Median",
                "AUDC": auc,
                "Time to FA": time_to_FA,
                "DD": delay,
                "Max F1": f1,
                "Cover": cover,
                "Max Cover": max_cover,
                "Max F1, m1": f1_m1,
                "Max F1, m2": f1_m2,
                "Max F1, m3": f1_m3,
                "params": f"th = {best_th_f1}",
            },
            ignore_index=True,
        )

    if "distances" in evaluation_components:
        if verbose:
            print("Evaluating dustance-based approach:")

        # wasserstein
        threshold_list_dist = np.linspace(0, 1, threshold_number)

        res_dist = all_distances_evaluation_pipeline(
            ens_model,
            test_dataloader,
            distance="wasserstein_1d",
            device=device,
            verbose=verbose,
            window_size_list=args_config["distance"]["window_size_list"],
            margin_list=args_config["evaluation"]["margin_list"],
            anchor_window_type_list=args_config["distance"]["anchor_window_type_list"],
            threshold_list=threshold_list_dist,
        )

        # extract metrics for distances
        best_f1_start = 0
        best_res_start = None
        best_ws_start = None
        best_th_start = None

        best_f1_prev = 0
        best_res_prev = None
        best_ws_prev = None
        best_th_prev = None

        for (ws, anchor_type), (res, best_th) in res_dist.items():
            f1 = res[3]

            if anchor_type == "start":
                if f1 > best_f1_start:
                    best_f1_start = f1
                    best_res_start = res
                    best_ws_start = ws
                    best_th_start = best_th

            if anchor_type == "prev":
                if f1 > best_f1_prev:
                    best_f1_prev = f1
                    best_res_prev = res
                    best_ws_prev = ws
                    best_th_prev = best_th

        (
            auc_start,
            time_to_FA_start,
            delay_start,
            f1_start,
            cover_start,
            max_cover_start,
            f1_margins_start,
        ) = best_res_start
        f1_m1_start, f1_m2_start, f1_m3_start = f1_margins_start.values()

        (
            auc_prev,
            time_to_FA_prev,
            delay_prev,
            f1_prev,
            cover_prev,
            max_cover_prev,
            f1_margins_prev,
        ) = best_res_prev
        f1_m1_prev, f1_m2_prev, f1_m3_prev = f1_margins_prev.values()

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Wasserstein start",
                "AUDC": auc_start,
                "Time to FA": time_to_FA_start,
                "DD": delay_start,
                "Max F1": f1_start,
                "Cover": cover_start,
                "Max Cover": max_cover_start,
                "Max F1, m1": f1_m1_start,
                "Max F1, m2": f1_m2_start,
                "Max F1, m3": f1_m3_start,
                "params": f"ws = {best_ws_start}, th = {best_th_start}",
            },
            ignore_index=True,
        )

        results_df = results_df.append(
            {
                "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, Wasserstein prev",
                "AUDC": auc_prev,
                "Time to FA": time_to_FA_prev,
                "DD": delay_prev,
                "Max F1": f1_prev,
                "Cover": cover_prev,
                "Max Cover": max_cover_prev,
                "Max F1, m1": f1_m1_prev,
                "Max F1, m2": f1_m2_prev,
                "Max F1, m3": f1_m3_prev,
                "params": f"ws = {best_ws_prev}, th = {best_th_prev}",
            },
            ignore_index=True,
        )

    if "cusums" in evaluation_components:
        if verbose:
            print("Evaluating CUSUM-based approaches:")

        normal_std_list, cp_std_list = compute_stds(
            ens_model,
            test_dataloader=test_dataloader,
            half_windows_list=[args_config["cusum"]["half_window"]],
            scale=scale,
            step=step,
            alpha=alpha,
            verbose=verbose,
            device=device,
        )[args_config["cusum"]["half_window"]]

        args_config["cusum"]["normal_sigma"] = np.mean(normal_std_list)
        args_config["cusum"]["cp_sigma"] = np.mean(cp_std_list)

        res_cusums = all_cusums_evaluation_pipeline(
            ens_model,
            threshold_number=threshold_number,
            test_dataloader=test_dataloader,
            margin_list=args_config["evaluation"]["margin_list"],
            device=device,
            verbose=verbose,
            min_th_quant=min_th_quant,
            max_th_quant=max_th_quant,
        )

        for cusum_type, (res, best_th) in res_cusums.items():
            auc, time_to_FA, delay, f1, cover, max_cover, f1_margins = res
            f1_m1, f1_m2, f1_m3 = f1_margins.values()

            results_df = results_df.append(
                {
                    "Model name": f"Ens num {ensemble_num}, cal = {calibrate}, CUSUM {cusum_type}",
                    "AUDC": auc,
                    "Time to FA": time_to_FA,
                    "DD": delay,
                    "Max F1": f1,
                    "Cover": cover,
                    "Max Cover": max_cover,
                    "Max F1, m1": f1_m1,
                    "Max F1, m2": f1_m2,
                    "Max F1, m3": f1_m3,
                    "params": f"th = {best_th}",
                },
                ignore_index=True,
            )

    if save_df:
        if verbose:
            print("Saving the results")

        if model_type == "seq2seq":
            model_name = model_type + "_" + loss_type
            if loss_type == "indid" and experiments_name in [
                "explosion",
                "road_accidents",
            ]:
                model_name += "_ln_before"
        else:
            model_name = model_type

        if calibrate:
            model_name += "_calibrated"

        if model_type == "seq2seq":
            save_path = f"results/final_results/{experiments_name}/{loss_type}/{model_name}_ens_num_{ensemble_num}_{experiments_name}.csv"
        else:
            save_path = f"results/final_results/{experiments_name}/{model_type}/{model_name}_ens_num_{ensemble_num}_{experiments_name}.csv"

        results_df.to_csv(save_path)

    return results_df


def evaluate_all_ensembles(
    experiments_name: str,
    model_type: str,
    loss_type: str = None,
    is_ensemble: bool = True,
    n_models: int = 10,
    calibrate: bool = False,
    ensemble_num_list: List[int] = [1, 2, 3],
    threshold_number: int = 300,
    evaluation_components: List[str] = [
        "mean",
        "min",
        "max",
        "median",
        "distances",
        "cusums",
    ],
    seed: int = 42,
    verbose: bool = True,
    save_df: bool = False,
):
    results = []
    for ensemble_num in ensemble_num_list:
        curr_res_df = evaluate_ensemble(
            experiments_name,
            model_type,
            loss_type,
            is_ensemble,
            n_models,
            calibrate,
            ensemble_num,
            threshold_number,
            evaluation_components,
            seed,
            verbose,
        )
        results.append(curr_res_df)
    results_df = pd.concat(results)

    if save_df:
        if verbose:
            print("Saving the results")

        if model_type == "seq2seq":
            model_name = model_type + "_" + loss_type
            if loss_type == "indid" and experiments_name in [
                "explosion",
                "road_accidents",
            ]:
                model_name += "_ln_before"
        else:
            model_name = model_type

        if calibrate:
            model_name += "_calibrated"

        if is_ensemble:
            if model_type == "seq2seq":
                save_path = f"results/final_results/{experiments_name}/{loss_type}/{model_name}_all_ensembles_{experiments_name}.csv"
            else:
                save_path = f"results/final_results/{experiments_name}/{model_type}/{model_name}_all_ensembles_{experiments_name}.csv"
        else:
            if model_type == "seq2seq":
                save_path = f"results/final_results/{experiments_name}/{loss_type}/{model_name}_all_bayes_{experiments_name}.csv"
            else:
                save_path = f"results/final_results/{experiments_name}/{model_type}/{model_name}_all_bayes_{experiments_name}.csv"

        results_df.to_csv(save_path)

    return results_df
