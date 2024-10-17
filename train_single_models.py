import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from src.datasets.datasets import CPDDatasets
from src.metrics.evaluation_pipelines import evaluation_pipeline
from src.metrics.metrics_utils import write_metrics_to_file
from src.models.model_utils import get_models_list
from src.utils.fix_seeds import fix_seeds

warnings.filterwarnings("ignore")

model_type = "seq2seq"
experiments_name = "road_accidents"
loss_type = "bce"

# read config file
path_to_config = "configs/" + "video" + "_" + model_type + ".yaml"

with open(path_to_config, "r") as f:
    args_config = yaml.safe_load(f.read())

args_config["experiments_name"] = experiments_name
args_config["model_type"] = model_type

args_config["num_workers"] = 2
args_config["loss_type"] = loss_type

args_config["model"]["ln_type"] = "after"

# prepare datasets
train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()

for ens_num in [2, 3]:
    for s in range(10):
        seed = s + 10 * (ens_num - 1)

        fix_seeds(seed)
        model = get_models_list(args_config, train_dataset, test_dataset)[-1]

        model_name = (
            args_config["experiments_name"]
            + "_"
            + args_config["loss_type"]
            # + "_seed_"
            + "_model_num_"
            + str(seed)
        )

        # logger = CometLogger(
        #     save_dir=f"logs/{experiments_name}",
        #     api_key="agnHNC2vEt7tOxnnxT4LzYf7Y",
        #     project_name="indid",
        #     workspace="stalex2902",
        #     experiment_name=model_name,
        # )

        # trainer = Trainer(
        #     max_epochs=args_config["learning"]["epochs"],
        #     accelerator="cpu",
        #     benchmark=True,
        #     check_val_every_n_epoch=1,
        #     gradient_clip_val=args_config["learning"]["grad_clip"],
        #     logger=logger,
        #     callbacks=EarlyStopping(**args_config["early_stopping"]),
        # )

        # trainer.fit(model)

        # torch.save(
        #     model.state_dict(),
        #     f"saved_models/{args_config['model_type']}/{experiments_name}/ens_{ens_num}/{model_name}.pth",
        # )

        model.load_state_dict(
            torch.load(
                f"saved_models/{args_config['loss_type']}/{experiments_name}/layer_norm/ens_{ens_num}/{model_name}.pth"
            )
        )

        model.eval()

        threshold_number = 300
        threshold_list = np.linspace(-5, 5, threshold_number)
        threshold_list = 1 / (1 + np.exp(-threshold_list))
        threshold_list = [-0.001] + list(threshold_list) + [1.001]

        all_metrics = evaluation_pipeline(
            model,
            model.val_dataloader(),
            threshold_list,
            device="cuda:1",
            model_type=model_type,
            verbose=True,
            margin_list=args_config["evaluation"]["margin_list"],
        )

        write_metrics_filename = f"results/{args_config['loss_type']}/{experiments_name}/single_model_results.txt"

        write_metrics_to_file(
            filename=write_metrics_filename,
            metrics=all_metrics,
            seed=None,
            timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
            comment=f"{experiments_name}, {args_config['model_type']}, {args_config['loss_type']}, seed = {seed}",
        )
