import logging
import os
import random
import shutil

import numpy as np
import torch
from alfred import constants
from alfred.config import exp_ingredient, train_ingredient
from alfred.data import GuidesEdhDataset, GuidesSpeakerDataset
from alfred.model.learned import LearnedModel
from alfred.utils import data_util, helper_util, model_util
from sacred import Experiment

from teach.logger import create_logger

#RL Modules
import glob
import json
import multiprocessing as mp
from argparse import ArgumentParser
from datetime import datetime

from teach.eval.compute_metrics import aggregate_metrics
from teach.rl_utils.rl_train_runner import RLTrainRunner, RLTrainRunnerConfig
from teach.logger import create_logger
from teach.utils import dynamically_load_class

ex = Experiment("train", ingredients=[train_ingredient, exp_ingredient])

logger = create_logger(__name__, level=logging.INFO)


def prepare(arg_parser, train, exp):
    """
    create logdirs, check dataset, seed pseudo-random generators
    """
    # # args and init
    # args = helper_util.AttrDict(**train, **exp)
    # args.dout = os.path.join(constants.ET_LOGS, args.name)
    # args.data["train"] = args.data["train"].split(",")
    # args.data["valid"] = args.data["valid"].split(",") if args.data["valid"] else []
    # num_datas = len(args.data["train"]) + len(args.data["valid"])
    # for key in ("ann_type",):
    #     args.data[key] = args.data[key].split(",")
    #     if len(args.data[key]) == 1:
    #         args.data[key] = args.data[key] * num_datas
    #     if len(args.data[key]) != num_datas:
    #         raise ValueError("Provide either 1 {} or {} separated by commas".format(key, num_datas))
    # # set seeds
    # torch.manual_seed(args.seed)
    # random.seed(a=args.seed)
    # np.random.seed(args.seed)
    
    # RL Runner args
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/teach-dataset",
        help='Base data directory containing subfolders "games" and "edh_instances',
    )
    arg_parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/ubuntu/simbot/teach/exp/images",
        help="Images directory for episode replay output",
    )
    arg_parser.add_argument(
        "--use_img_file",
        dest="use_img_file",
        action="store_true",
        help="synchronous save images with model api use the image file instead of streaming image",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/simbot/teach/exp",
        help="Directory to store output files from playing EDH instances",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )
    arg_parser.add_argument(
        "--edh_instance_file",
        type=str,
        help="Run only on this EDH instance. Split must be set appropriately to find corresponding game file.",
    )
    arg_parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
    arg_parser.add_argument(
        "--max_init_tries",
        type=int,
        default=5,
        help="Max attempts to correctly initialize an instance before declaring it as a failure",
    )
    arg_parser.add_argument(
        "--max_traj_steps",
        type=int,
        default=1000,
        help="Max predicted trajectory steps",
    )
    arg_parser.add_argument("--max_api_fails", type=int, default=30, help="Max allowed API failures")
    arg_parser.add_argument(
        "--metrics_file",
        type=str,
        default="/home/ubuntu/simbot/teach/exp/metrics_file",
        help="File used to store metrics",
    )
    arg_parser.add_argument(
        "--model_module",
        type=str,
        default="teach.inference.et_rl_model",
        help="Path of the python module to load the model class from.",
    )
    arg_parser.add_argument(
        "--model_class", type=str, default="ETRLModel", help="Name of the TeachModel class to use during inference."
    )
    arg_parser.add_argument(
        "--replay_timeout", type=int, default=500, help="The timeout for playing back the interactions in an episode."
    )

    # make output dir
    # logger.info("Train args: %s" % str(args))
    # if not os.path.isdir(args.dout):
    #     os.makedirs(args.dout)
    args, model_args = arg_parser.parse_known_args()

    return args, model_args


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


def create_model(args, embs_ann, vocab_out):
    """
    load a model and its optimizer
    """
    prev_train_info = model_util.load_log(args.dout, stage="train")
    if args.resume and os.path.exists(os.path.join(args.dout, "latest.pth")):
        # load a saved model
        loadpath = os.path.join(args.dout, "latest.pth")
        model, optimizer = model_util.load_model(loadpath, args.device, prev_train_info["progress"] - 1)
        assert model.vocab_out.contains_same_content(vocab_out)
        model.args = args
    else:
        # create a new model
        if not args.resume and os.path.isdir(args.dout):
            shutil.rmtree(args.dout)
        model = LearnedModel(args, embs_ann, vocab_out)
        model = model.to(torch.device(args.device))
        optimizer = None
        if args.pretrained_path:
            if "/" not in args.pretrained_path:
                # a relative path at the logdir was specified
                args.pretrained_path = model_util.last_model_path(args.pretrained_path)
            logger.info("Loading pretrained model from {}".format(args.pretrained_path))
            pretrained_model = torch.load(args.pretrained_path, map_location=torch.device(args.device))
            if args.use_alfred_weights:
                pretrained_dict = load_only_matching_layers(model, pretrained_model, args.data["train"][0])
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                loaded_keys = pretrained_dict.keys()
            else:
                model.load_state_dict(pretrained_model["model"], strict=False)
                loaded_keys = set(model.state_dict().keys()).intersection(set(pretrained_model["model"].keys()))
            assert len(loaded_keys)
            logger.debug("Loaded keys: %s", str(loaded_keys))
    # put encoder on several GPUs if asked
    if torch.cuda.device_count() > 1:
        logger.info("Parallelizing the model")
        model.model = helper_util.DataParallel(model.model)
    return model, optimizer, prev_train_info


def load_data(name, args, ann_type, valid_only=False):
    """
    load dataset and wrap them into torch loaders
    """
    partitions = ([] if valid_only else ["train"]) + ["valid_seen", "valid_unseen"]
    datasets = []
    for partition in partitions:
        if args.model == "speaker":
            dataset = GuidesSpeakerDataset(name, partition, args, ann_type)
        elif args.model == "transformer":
            dataset = GuidesEdhDataset(name, partition, args, ann_type)
        else:
            raise ValueError("Unknown model: {}".format(args.model))
        datasets.append(dataset)
    return datasets


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


def process_vocabs(datasets, args):
    """
    assign the largest output vocab to all datasets, compute embedding sizes
    """
    # find the longest vocabulary for outputs among all datasets
    for dataset in datasets:
        logger.debug("dataset.id = %s, vocab_out = %s" % (dataset.id, str(dataset.vocab_out)))
    vocab_out = sorted(datasets, key=lambda x: len(x.vocab_out))[-1].vocab_out
    # make all datasets to use this vocabulary for outputs translation
    for dataset in datasets:
        dataset.vocab_translate = vocab_out
    # prepare a dictionary for embeddings initialization: vocab names and their sizes
    embs_ann = {}
    for dataset in datasets:
        embs_ann[dataset.name] = len(dataset.vocab_in)
    return embs_ann, vocab_out


@ex.automain
def main(train, exp):
    """
    train a network using an lmdb dataset
    """
    mp.set_start_method("spawn", force=True)
    # parse args
    arg_parser = ArgumentParser()
    args, model_args = prepare(arg_parser, train, exp)

    # # load dataset(s) and process vocabs
    # datasets = []
    # ann_types = iter(args.data["ann_type"])
    # for name, ann_type in zip(args.data["train"], ann_types):
    #     datasets.extend(load_data(name, args, ann_type))
    # for name, ann_type in zip(args.data["valid"], ann_types):
    #     datasets.extend(load_data(name, args, ann_type, valid_only=True))
    # # assign vocabs to datasets and check their sizes for nn.Embeding inits
    # embs_ann, vocab_out = process_vocabs(datasets, args)
    # logger.debug("In train.main, vocab_out = %s" % str(vocab_out))
    # # wrap datasets with loaders
    # loaders = wrap_datasets(datasets, args)
    # # create the model
    # model, optimizer, prev_train_info = create_model(args, embs_ann, vocab_out)
    # start train loop
    # model.run_train(loaders, prev_train_info, optimizer=optimizer)

    start_time = datetime.now()

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

    runner_config = RLTrainRunnerConfig(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        metrics_file=args.metrics_file,
        num_processes=args.num_processes,
        max_init_tries=args.max_init_tries,
        max_traj_steps=args.max_traj_steps,
        max_api_fails=args.max_api_fails,
        model_class=dynamically_load_class(args.model_module, args.model_class),
        replay_timeout=args.replay_timeout,
        model_args=model_args,
        use_img_file=args.use_img_file,
    )

    runner = RLTrainRunner(edh_instance_files, runner_config)
    metrics = runner.run()

    train_end_time = datetime.now()
    logger.info("Time for RL training: %s" % str(train_end_time - start_time))


