import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from src.datasets.datasets import CPDDatasets
from src.ensembles.ensembles import EnsembleCPDModel
from src.metrics.evaluation_pipelines import evaluate_all_models_in_ensemble
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import DataLoader

SEED = 42
fix_seeds(SEED)

DEVICE = "cpu"
VERBOSE = False

model_type = "tscp"

experiments_name = "yahoo"

path_to_config = "configs/" + experiments_name + "_" + model_type + ".yaml"

with open(path_to_config, "r") as f:
    args_config = yaml.safe_load(f.read())

args_config["experiments_name"] = experiments_name
args_config["model_type"] = model_type

# args_config["loss_type"] = "bce"
args_config["num_workers"] = 2

train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()
test_dataloader = DataLoader(
    test_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
)

_, val_dataset = train_test_split(train_dataset, test_size=0.3, random_state=SEED)
val_dataloader = DataLoader(
    val_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
)

# path_to_models_folder = "saved_models/bce/road_accidents/layer_norm/ens_1"
# path_to_models_folder = (
#     "saved_models/bce/explosion/layer_norm/train_anomaly_num_155/ens_1"
# )
# path_to_models_folder = "saved_models/bce/human_activity/full_sample/ens_1"

# path_to_models_folder = "saved_models/tscp/human_activity/window_4/ens_1"

# path_to_models_folder = "saved_models/tscp/synthetic_1D/window_16/ens_1"
# path_to_models_folder = "saved_models/bce/synthetic_1D/full_sample/ens_1"

path_to_models_folder = "saved_models/tscp/yahoo/window_75/ens_1"

ens_bce = EnsembleCPDModel(args_config, n_models=10, boot_sample_size=None)
ens_bce.load_models_list(path_to_models_folder)

# _ = calibrate_all_models_in_ensemble(
#     ens_bce,
#     val_dataloader,
#     cal_type="beta",
#     verbose=VERBOSE,
#     device=DEVICE,
# )

(
    time_fa_list_1,
    delay_list_1,
    audc_list_1,
    f1_list_1,
    cover_list_1,
    max_cover_list_1,
    f1_m1_list_1,
    f1_m2_list_1,
    f1_m3_list_1,
) = evaluate_all_models_in_ensemble(
    ens_bce,
    test_dataloader,
    threshold_number=300,
    device=DEVICE,
    model_type=model_type,
    margin_list=[1, 2, 4],
    # scale=args_config["predictions"]["scale"],
    verbose=VERBOSE,
)

print("Ensemble 1 is evaluated")

# path_to_models_folder = (
#     "saved_models/bce/explosion/layer_norm/train_anomaly_num_155/ens_2"
# )
# path_to_models_folder = "saved_models/bce/human_activity/full_sample/ens_2"

# path_to_models_folder = "saved_models/tscp/human_activity/window_4/ens_2"

# path_to_models_folder = "saved_models/tscp/synthetic_1D/window_16/ens_2"
# path_to_models_folder = "saved_models/bce/synthetic_1D/full_sample/ens_2"
path_to_models_folder = "saved_models/tscp/yahoo/window_75/ens_2"

ens_bce = EnsembleCPDModel(args_config, n_models=10, boot_sample_size=None)
ens_bce.load_models_list(path_to_models_folder)

# _ = calibrate_all_models_in_ensemble(
#     ens_bce,
#     val_dataloader,
#     cal_type="beta",
#     verbose=VERBOSE,
#     device=DEVICE,
# )

(
    time_fa_list_2,
    delay_list_2,
    audc_list_2,
    f1_list_2,
    cover_list_2,
    max_cover_list_2,
    f1_m1_list_2,
    f1_m2_list_2,
    f1_m3_list_2,
) = evaluate_all_models_in_ensemble(
    ens_bce,
    test_dataloader,
    threshold_number=300,
    device=DEVICE,
    model_type=model_type,
    margin_list=[1, 2, 4],
    # scale=args_config["predictions"]["scale"],
    verbose=VERBOSE,
)

print("Ensemble 2 is evaluated")

# path_to_models_folder = (
#     "saved_models/bce/explosion/layer_norm/train_anomaly_num_155/ens_3"
# )
# path_to_models_folder = "saved_models/bce/human_activity/full_sample/ens_3"

# path_to_models_folder = "saved_models/tscp/human_activity/window_4/ens_3"

# path_to_models_folder = "saved_models/tscp/synthetic_1D/window_16/ens_3"
# path_to_models_folder = "saved_models/bce/synthetic_1D/full_sample/ens_3"
path_to_models_folder = "saved_models/tscp/yahoo/window_75/ens_3"

ens_bce = EnsembleCPDModel(args_config, n_models=10, boot_sample_size=None)
ens_bce.load_models_list(path_to_models_folder)

# _ = calibrate_all_models_in_ensemble(
#     ens_bce,
#     val_dataloader,
#     cal_type="beta",
#     verbose=VERBOSE,
#     device=DEVICE,
# )

(
    time_fa_list_3,
    delay_list_3,
    audc_list_3,
    f1_list_3,
    cover_list_3,
    max_cover_list_3,
    f1_m1_list_3,
    f1_m2_list_3,
    f1_m3_list_3,
) = evaluate_all_models_in_ensemble(
    ens_bce,
    test_dataloader,
    threshold_number=300,
    device=DEVICE,
    model_type=model_type,
    margin_list=[1, 2, 4],
    # scale=args_config["predictions"]["scale"],
    verbose=VERBOSE,
)

print("Ensemble 3 is evaluated")

time_fa_list = time_fa_list_1 + time_fa_list_2 + time_fa_list_3
delay_list = delay_list_1 + delay_list_2 + delay_list_3
audc_list = audc_list_1 + audc_list_2 + audc_list_3
f1_list = f1_list_1 + f1_list_2 + f1_list_3
cover_list = cover_list_1 + cover_list_2 + cover_list_3
max_cover_list = max_cover_list_1 + max_cover_list_2 + max_cover_list_3
f1_m1_list = f1_m1_list_1 + f1_m1_list_2 + f1_m1_list_3
f1_m2_list = f1_m2_list_1 + f1_m2_list_2 + f1_m2_list_3
f1_m3_list = f1_m3_list_1 + f1_m3_list_2 + f1_m3_list_3

print(
    f"Time to FA: {np.round(np.mean(time_fa_list), 2)} \pm {np.round(np.std(time_fa_list), 2)}"
)
print(f"ADD: {np.round(np.mean(delay_list), 2)} \pm {np.round(np.std(delay_list), 2)}")
print(f"AUDC: {np.round(np.mean(audc_list), 2)} \pm {np.round(np.std(audc_list), 2)}")
print(f"F1: {np.round(np.mean(f1_list), 4)} \pm {np.round(np.std(f1_list), 4)}")
print(
    f"Cover: {np.round(np.mean(cover_list), 4)} \pm {np.round(np.std(cover_list), 4)}"
)
print(
    f"Max Cover: {np.round(np.mean(max_cover_list), 4)} \pm {np.round(np.std(max_cover_list), 4)}"
)
print(
    f"F1, m1: {np.round(np.mean(f1_m1_list), 4)} \pm {np.round(np.std(f1_m1_list), 4)}"
)
print(
    f"F1, m2: {np.round(np.mean(f1_m2_list), 4)} \pm {np.round(np.std(f1_m2_list), 4)}"
)
print(
    f"F1, m3: {np.round(np.mean(f1_m3_list), 4)} \pm {np.round(np.std(f1_m3_list), 4)}"
)
