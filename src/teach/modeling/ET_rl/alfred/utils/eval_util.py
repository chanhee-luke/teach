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
from alfred.nn.enc_visual import FeatureExtractor

from teach.logger import create_logger


logger = create_logger(__name__, level=logging.INFO)


def load_agent(model_path, dataset_info, args, for_inference=False):
    """
    load a pretrained agent and its feature extractor
    """
    logger.info("In load_agent, model_path = %s, dataset_info = %s" % (str(model_path), str(dataset_info)))
    learned_model, _ = model_util.load_model(model_path, args.device, for_inference=for_inference)
    model = learned_model.model
    model.eval()
    model.args.device = args.device
    extractor = FeatureExtractor(
        archi=dataset_info["visual_archi"],
        device=args.device,
        checkpoint=args.visual_checkpoint,
        compress_type=dataset_info["compress_type"],
    )
    return model, extractor


def load_object_predictor(args):
    if args.object_predictor is None:
        return None
    return FeatureExtractor(
        archi="maskrcnn",
        device=args.device,
        checkpoint=args.object_predictor,
        load_heads=True,
    )

def create_agent(args, dataset_info, embs_ann, vocab_out):
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
    
    extractor = FeatureExtractor(
        archi=dataset_info["visual_archi"],
        device=args.device,
        checkpoint=args.visual_checkpoint,
        compress_type=dataset_info["compress_type"],
    )

    return model, extractor, optimizer, prev_train_info