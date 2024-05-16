import warnings

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CometLogger
from src.datasets.datasets import CPDDatasets
from src.metrics.evaluation_pipelines import evaluation_pipeline
from src.metrics.metrics_utils import write_metrics_to_file
from src.models.model_utils import get_models_list
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

model_type = "seq2seq"

experiments_name = "explosion"

# read config file
path_to_config = "configs/" + "video" + "_" + model_type + ".yaml"

with open(path_to_config, "r") as f:
    args_config = yaml.safe_load(f.read())

args_config["experiments_name"] = experiments_name
args_config["model_type"] = model_type

args_config["loss_type"] = "bce"
args_config["num_workers"] = 2

# prepare datasets
train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()
val_dataloader = DataLoader(
    test_dataset, batch_size=args_config["learning"]["batch_size"], shuffle=False
)

model = get_models_list(args_config, train_dataset, test_dataset)[-1]

for seed in range(20, 30):
    fix_seeds(seed)

    model_name = f'{experiments_name}_{args_config["loss_type"]}_model_num_{seed}'

    logger = CometLogger(
        save_dir=f"logs/{experiments_name}",
        api_key="agnHNC2vEt7tOxnnxT4LzYf7Y",
        project_name="cpd-with-uncertainty",
        workspace="stalex2902",
        experiment_name=model_name,
    )

    trainer = Trainer(
        max_epochs=args_config["learning"]["epochs"],
        # max_epochs=1,
        gpus=1,
        benchmark=True,
        check_val_every_n_epoch=1,
        gradient_clip_val=args_config["learning"]["grad_clip"],
        logger=logger,
        callbacks=EarlyStopping(**args_config["early_stopping"]),
    )
    trainer.fit(model)

    threshold_number = 300
    threshold_list = np.linspace(-5, 5, threshold_number)
    threshold_list = 1 / (1 + np.exp(-threshold_list))
    threshold_list = [-0.001] + list(threshold_list) + [1.001]

    # metrics, (max_th_f1_margins_dict, max_f1_margins_dic), _, _
    metrics = evaluation_pipeline(
        model,
        val_dataloader,
        threshold_list,
        device="cuda",
        model_type="seq2seq",
        verbose=False,
        margin_list=args_config["evaluation"]["margin_list"],
    )

    write_metrics_to_file(
        filename=f'results/{args_config["loss_type"]}/{experiments_name}/{experiments_name}_single_model.txt',
        metrics=metrics,
        seed=seed,
        timestamp=None,
        comment=f"Standard_bce_single_model_num_{seed}",
    )

    torch.save(
        model.state_dict(),
        # f'saved_models/{args_config["loss_type"]}/{experiments_name}/layer_norm/{model_name}.pth',
        f'saved_models/{args_config["loss_type"]}/{experiments_name}/layer_norm/train_anomaly_num_155/{model_name}.pth',
    )
