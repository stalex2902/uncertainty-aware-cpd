import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import numpy as np

from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.datasets import CPDDatasets
from utils.model_utils import fix_seeds
from utils.core_models import CombinedVideoRNN
from utils.sgld import CPDModelCustomNoisyAdam
from utils.metrics import evaluation_pipeline, write_metrics_to_file


model_type = "seq2seq"

experiments_name = "explosion"

path_to_config = "configs/" + "video" + "_" + model_type + ".yaml"

with open(path_to_config, 'r') as f:
    args_config = yaml.safe_load(f.read())

args_config["experiments_name"] = experiments_name
args_config["model_type"] = model_type

args_config["loss_type"] = "bce"
args_config["num_workers"] = 4
args_config["learning"]["gpus"] = 1

args_config["learning"]["epochs"] = 100

train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()

args_config["learning"]["lr"] = 1e-3
args_config["learning"]["temperature"] = 0.001

# for ABAnnealingLR
args_config["learning"]["final_lr"] = 1e-4
args_config["learning"]["gamma"] = 0.5
args_config["learning"]["T_max"] = 50000

threshold_number = 100
threshold_list = np.linspace(-5, 5, threshold_number)
threshold_list = 1 / (1 + np.exp(-threshold_list))
threshold_list = [-0.001] + list(threshold_list) + [1.001]


for SEED in range(5, 10):
    fix_seeds(SEED)
    
    timestamp = datetime.now().strftime("%y%m%dT%H%M%S")

    core_model = CombinedVideoRNN(
        input_dim=args_config["model"]["input_size"],
        rnn_hidden_dim=args_config["model"]["hidden_rnn"],
        num_layers=args_config["model"]["rnn_n_layers"],
        rnn_dropout=args_config["model"]["rnn_dropout"],
        dropout=args_config["model"]["dropout"],
        layer_norm=args_config["model"]["layer_norm"]
    )

    bce_sgld_model = CPDModelCustomNoisyAdam(
        loss_type="bce",
        args=args_config,
        model=core_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    logger = CometLogger(
        save_dir='logs/bce/sgld',
        api_key='agnHNC2vEt7tOxnnxT4LzYf7Y',
        project_name='cpd-with-uncertainty',
        workspace='stalex2902',
        experiment_name=f'BCE-SGLD_Explosion_seed_{SEED}',
        display_summary_level=0
    )

    trainer = Trainer(
        max_epochs=args_config["learning"]["epochs"],
        gpus=args_config["learning"]["gpus"],
        benchmark=True,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.,
        logger=logger,
        callbacks=EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
    )

    trainer.fit(bce_sgld_model)
    
    loss_type = args_config["loss_type"]
    path_to_folder = "saved_models/bce/explosion/sgld_adam/"

    path = (
        path_to_folder
        + "/"
        + args_config["experiments_name"]
        + "_loss_type_"
        + str(loss_type)
        + "sgld_adam_"
        + "_model_num_"
        + str(SEED)
        + ".pth"
    )
    
    torch.save(bce_sgld_model.state_dict(), path)
    
    metrics, (max_th_f1_margins_dict, max_f1_margins_dic), _, _ = evaluation_pipeline(
        bce_sgld_model,
        bce_sgld_model.val_dataloader(),
        threshold_list,
        device="cuda", # choose 'cpu' or 'cuda' if available
        model_type="seq2seq",
        verbose=True,
        margin_list=[1, 2, 4],
    )
    
    write_metrics_to_file(
        filename="results/bce/explosion/sgld_adam/BCE_single_sgld_adam.txt",
        metrics=(metrics, (max_th_f1_margins_dict, max_f1_margins_dic)),
        seed=SEED,
        timestamp=timestamp,
        comment="Single BCE-SGLD model, default params",
    )