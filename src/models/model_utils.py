"""Module with utility functions for models initialization."""

from typing import List

import pytorch_lightning as pl
import ruptures as rpt
from src.baselines import klcpd, ts2vec_sn, tscp
from src.models import core_models, cpd_models
from torch.utils.data import Dataset


def get_models_list(
    args: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> List[pl.LightningModule]:
    """Initialize CPD models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :return models_list: list with 2 models in case of a 'seq2seq' model with the combined loss,
                         list with 1 model, otherwise
    """
    if args["model_type"] == "seq2seq":
        models_list = get_seq2seq_models_list(args, train_dataset, test_dataset)

    elif args["model_type"] == "kl_cpd":
        models_list = get_kl_cpd_models_list(args, train_dataset, test_dataset)

    elif args["model_type"] == "tscp":
        models_list = get_tscp_models_list(args, train_dataset, test_dataset)

    elif args["model_type"] == "ts2vec":
        models_list = get_ts2vec_models_list(args, train_dataset, test_dataset)

    elif args["model_type"].startswith("classic"):
        models_list = get_classic_models_list(args)

    else:
        raise ValueError(f'Unknown model {args["model_type"]}.')

    return models_list


def get_seq2seq_models_list(
    args: dict, train_dataset: Dataset, test_dataset: Dataset
) -> List[cpd_models.CPDModel]:
    """Initialize seq2seq models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 2 CPD models in case of 'combined loss',
              list with 1 CPD model in case of 'indid' or 'bce' loss
    """
    # Initialize core models for synthetic_1D, _100D and human_activity experiments
    if args["experiments_name"] in [
        "synthetic_1D",
        "synthetic_100D",
        "human_activity",
        "yahoo",
    ]:
        # initialize default base model for Synthetic Normal 1D experiment
        if args["model"]["layer_norm"]:
            assert "ln_type" in args["model"].keys()
            core_model = core_models.BaseRnn(
                input_size=args["model"]["input_size"],
                hidden_dim=args["model"]["hidden_dim"],
                n_layers=args["model"]["n_layers"],
                drop_prob=args["model"]["drop_prob"],
                layer_norm=args["model"]["layer_norm"],
                ln_type=args["model"]["ln_type"],
            )
        else:
            core_model = core_models.BaseRnn(
                input_size=args["model"]["input_size"],
                hidden_dim=args["model"]["hidden_dim"],
                n_layers=args["model"]["n_layers"],
                drop_prob=args["model"]["drop_prob"],
                layer_norm=args["model"]["layer_norm"],
                ln_type=None,
            )

    elif args["experiments_name"] == "mnist":
        # initialize default base model for MNIST experiment
        core_model = core_models.MnistRNN(
            input_size=args["model"]["input_size"],
            hidden_rnn=args["model"]["hidden_rnn"],
            rnn_n_layers=args["model"]["rnn_n_layers"],
            linear_dims=args["model"]["linear_dims"],
            rnn_dropout=args["model"]["rnn_dropout"],
            dropout=args["model"]["dropout"],
            rnn_type=args["model"]["rnn_type"],
        )

    elif args["experiments_name"] in ["explosion", "road_accidents"]:
        if args["model"]["layer_norm"]:
            assert "ln_type" in args["model"].keys()
            core_model = core_models.CombinedVideoRNN(
                input_dim=args["model"]["input_size"],
                rnn_hidden_dim=args["model"]["hidden_rnn"],
                num_layers=args["model"]["rnn_n_layers"],
                rnn_dropout=args["model"]["rnn_dropout"],
                dropout=args["model"]["dropout"],
                layer_norm=args["model"]["layer_norm"],
                ln_type=args["model"]["ln_type"],
            )
        else:
            core_model = core_models.CombinedVideoRNN(
                input_dim=args["model"]["input_size"],
                rnn_hidden_dim=args["model"]["hidden_rnn"],
                num_layers=args["model"]["rnn_n_layers"],
                rnn_dropout=args["model"]["rnn_dropout"],
                dropout=args["model"]["dropout"],
                layer_norm=args["model"]["layer_norm"],
                ln_type=None,
            )
    else:
        raise ValueError("Wrong experiment name.")

    # Initialize CPD models
    if args["loss_type"] in ["indid", "bce"]:
        model = cpd_models.CPDModel(
            loss_type=args["loss_type"],
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        models_list = [model]

    elif args["loss_type"] == "combined":
        model_1 = cpd_models.CPDModel(
            loss_type="bce",
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        model_2 = cpd_models.CPDModel(
            loss_type="indid",
            args=args,
            model=core_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        models_list = [model_1, model_2]
    else:
        raise ValueError("Wrong loss type. Choose 'indid', 'bce' or 'combined'.'")
    return models_list


def get_kl_cpd_models_list(
    args: dict, train_dataset: Dataset, test_dataset: Dataset
) -> List[klcpd.KLCPD]:
    """Initialize KL-CPD models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 1 KL-CPD model
    """
    # the same model structures for all the experiments, args are read from different configs
    if args["experiments_name"] in ["explosion", "road_accidents"]:
        netD = klcpd.VideoNetD(args)
        netG = klcpd.VideoNetG(args)

    elif args["experiments_name"] in [
        "synthetic_1D",
        "synthetic_100D",
        "mnist",
        "human_activity",
    ]:
        netD = klcpd.NetD(args)
        netG = klcpd.NetG(args)
    else:
        raise ValueError("Wrong experiments name.")

    model = klcpd.KLCPD(
        args=args,
        net_generator=netG,
        net_discriminator=netD,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    models_list = [model]
    return models_list


def get_tscp_models_list(
    args: dict, train_dataset: Dataset, test_dataset: Dataset
) -> List[tscp.TSCP_model]:
    """Initialize TS-CP2 models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 1 TS-CP2 model
    """
    # universal encoder for all the experiments
    encoder = tscp.BaseTSCPEncoder(args)

    model = tscp.TSCP_model(
        args=args, model=encoder, train_dataset=train_dataset, test_dataset=test_dataset
    )

    models_list = [model]
    return models_list


def get_ts2vec_models_list(
    args: dict, train_dataset: Dataset, test_dataset: Dataset
) -> List[tscp.TSCP_model]:
    """Initialize TS-CP2 models for a particular experiment.

    :param args: dict with all the parameters
    :param train_dataset: training data
    :param test_dataset: testing data
    :returns: list with 1 TS-CP2 model
    """
    # universal encoder for all the experiments
    encoder = ts2vec_sn.TSEncoder(
        input_dim=args["model"]["input_dim"],
        output_dim=args["model"]["output_dim"],
        hidden_dim=args["model"]["hidden_dim"],
        depth=args["model"]["depth"],
        spec_norm=args["model"]["spec_norm"],
    )

    model = ts2vec_sn.TS2Vec(
        args=args, model=encoder, train_dataset=train_dataset, test_dataset=test_dataset
    )

    models_list = [model]
    return models_list


def get_classic_models_list(args: dict) -> List[cpd_models.ClassicBaseline]:
    """Initialize classic baseline models.

    :param args: dict with all the parameters
    :return: ClassicBaseline model
    """

    if (args["n_pred"] is None and args["pen"] is None) or (
        args["n_pred"] is not None and args["pen"] is not None
    ):
        raise ValueError(
            "You should specify either 'n_pred' or 'pen' parameter for a ClassicBaseline model."
        )

    if args["model_type"] == "classic_binseg":
        model = cpd_models.ClassicBaseline(
            rpt.Binseg(model=args["core_model"]), n_pred=args["n_pred"], pen=args["pen"]
        )

    elif args["model_type"] == "classic_pelt":
        model = cpd_models.ClassicBaseline(
            rpt.Pelt(model=args["core_model"]), n_pred=args["n_pred"], pen=args["pen"]
        )

    elif args["model_type"] == "classic_kernel":
        model = cpd_models.ClassicBaseline(
            rpt.KernelCPD(kernel=args["kernel"]), n_pred=args["n_pred"], pen=args["pen"]
        )

    else:
        raise ValueError(f'Wrong classic baseline type: {args["model_type"]}.')

    # for consistency with the general interface
    model = None
    models_list = [model]

    return models_list
