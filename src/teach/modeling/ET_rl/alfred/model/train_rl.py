import logging
import os
import random
import shutil

import numpy as np
import torch
from alfred import constants
from alfred.config import exp_ingredient, train_ingredient
from alfred.model.learned import LearnedModel
from alfred.utils import data_util, helper_util, model_util
from alfred.model.et_rl_model import ETRLModel
from sacred import Experiment

from teach.logger import create_logger

#RL Modules
import glob
import json
import multiprocessing as mp
from argparse import ArgumentParser
from datetime import datetime

from teach.eval.compute_metrics import aggregate_metrics
from alfred.utils.rl_agent import RLAgent, RLAgentConfig
from teach.logger import create_logger
from teach.utils import dynamically_load_class


ex = Experiment("train", ingredients=[train_ingredient, exp_ingredient])

logger = create_logger(__name__, level=logging.INFO)


def prepare(train, exp):
    """
    create logdirs, check dataset, seed pseudo-random generators
    """

    # Data loading args
    args = helper_util.AttrDict(**train, **exp)

    # make output dir
    # logger.info("Train args: %s" % str(args))
    # if not os.path.isdir(args.dout):
    #     os.makedirs(args.dout)

    # args and init
    args.data["train"] = args.data["train"].split(",")
    args.data["valid"] = args.data["valid"].split(",") if args.data["valid"] else []
    num_datas = len(args.data["train"]) + len(args.data["valid"])
    for key in ("ann_type",):
        args.data[key] = args.data[key].split(",")
        if len(args.data[key]) == 1:
            args.data[key] = args.data[key] * num_datas
        if len(args.data[key]) != num_datas:
            raise ValueError("Provide either 1 {} or {} separated by commas".format(key, num_datas))
    # set seeds
    torch.manual_seed(args.seed)
    random.seed(a=args.seed)
    np.random.seed(args.seed)

    return args


def load_only_matching_layers(model, pretrained_model, train_lmdb_name):
    pretrained_dict = {}
    model_dict = model.state_dict()

    logger.debug("Pretrained Model keys: %s" % str(pretrained_model["model"].keys()))
    logger.debug("Model state dict keys: %s" % str(model_dict.keys()))

    for name, param in pretrained_model["model"].items():
        model_name = name
        if name not in model_dict.keys():
            model_name = name.replace("lmdb_human", train_lmdb_name)
            if model_name not in model_dict.keys():
                logger.debug("No matching key ignoring %s" % model_name)
                continue

        if param.size() == model_dict[model_name].size():
            logger.debug(
                "Matched name and size: %s %s %s" % (name, str(param.size()), str(model_dict[model_name].size()))
            )
            pretrained_dict[model_name] = param
        else:
            logger.debug("Mismatched size: %s %s %s" % (name, str(param.size()), str(model_dict[model_name].size())))
    logger.debug("Matched keys: %s" % str(pretrained_dict.keys()))
    return pretrained_dict


def wrap_datasets(datasets, args):
    """
    wrap datasets with torch loaders
    """
    batch_size = args.batch // len(args.data["train"])
    loader_args = {
        "num_workers": args.num_workers,
        "drop_last": (torch.cuda.device_count() > 1),
        "collate_fn": helper_util.identity,
    }
    if args.num_workers > 0:
        # do not prefetch samples, this may speed up data loading
        loader_args["prefetch_factor"] = 1

    loaders = {}
    for dataset in datasets:
        if dataset.partition == "train":
            if args.data["train_load_type"] == "sample":
                weights = [1 / len(dataset)] * len(dataset)
                num_samples = 16 if args.fast_epoch else (args.data["length"] or len(dataset))
                num_samples = num_samples // len(args.data["train"])
                sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
                loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, **loader_args)
            else:
                loader = torch.utils.data.DataLoader(dataset, args.batch, shuffle=True, **loader_args)
        else:
            loader = torch.utils.data.DataLoader(dataset, args.batch, shuffle=(not args.fast_epoch), **loader_args)
        loaders[dataset.id] = loader
    return loaders


@ex.automain
def main(train, exp):
    """
    train a network using the lmdb dataset
    """
    start_time = datetime.now()

    mp.set_start_method("spawn", force=True)
    # parse args
    args = prepare(train, exp)

    # load dataset(s) and process vocabs
    # datasets = []
    # ann_types = iter(args.data["ann_type"])
    # for name, ann_type in zip(args.data["train"], ann_types):
    #     datasets.extend(load_data(name, args, ann_type))
    # for name, ann_type in zip(args.data["valid"], ann_types):
    #     datasets.extend(load_data(name, args, ann_type, valid_only=True))
    # # assign vocabs to datasets and check their sizes for nn.Embeding inits
    # embs_ann, vocab_out = process_vocabs(datasets, args)
    # args.embs_ann = embs_ann
    # args.vocab_out = vocab_out

    # # Model args for the simulator
    # model_args = [args.model_module, args.model_class]

    # logger.debug("In train.main, vocab_out = %s" % str(vocab_out))
    # # wrap datasets with loaders
    # loaders = wrap_datasets(datasets, args)
    # # create the model
    # model, optimizer, prev_train_info = create_model(args, embs_ann, vocab_out)
    # start train loop
    # model.run_train(loaders, prev_train_info, optimizer=optimizer)


    if args.edh_instance_file:
        edh_instance_files = [args.edh_instance_file]
    else:
        train_output_files = glob.glob(os.path.join(args.output_dir, "inference__*.json"))
        finished_edh_instance_files = [os.path.join(fn.split("__")[1]) for fn in train_output_files]
        edh_instance_files = [
            os.path.join(args.data_dir, "edh_instances", args.split, f)
            for f in os.listdir(os.path.join(args.data_dir, "edh_instances", args.split))
            if f not in finished_edh_instance_files
        ]
        if not edh_instance_files:
            print(
                f"all the edh instances have been ran for input_dir={os.path.join(args.data_dir, 'edh_instances', args.split)}"
            )
            exit(1)

    agent_config = RLAgentConfig(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        metrics_file=args.metrics_file,
        num_processes=args.num_processes,
        max_init_tries=args.max_init_tries,
        max_traj_steps=args.max_traj_steps,
        max_api_fails=args.max_api_fails,
        replay_timeout=args.replay_timeout,
        model_args=args,
        use_img_file=args.use_img_file,
    )

    model = ETRLModel(agent_config.model_args)

    agent = RLAgent(edh_instance_files, agent_config)
    metrics = agent.run(model)

    train_end_time = datetime.now()
    logger.info("Time for RL training: %s" % str(train_end_time - start_time))


