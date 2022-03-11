# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os
from pathlib import Path
from typing import List
from importlib import import_module

import numpy as np
import torch
from alfred import constants
from alfred.data import GuidesEdhDataset
from alfred.data.preprocessor import Preprocessor
from alfred.utils import data_util, eval_util, model_util
from alfred.model import train
from alfred.nn.enc_visual import FeatureExtractor

from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

logger = create_logger(__name__)


class ETRLModel(torch.nn.Module):
    """
    Wrapper around ET Model for inference
    """

    def __init__(self, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """

        torch.nn.Module.__init__(self)
        args = model_args
        args.dout = args.model_dir
        self.args = args
        # sentinel tokens
        self.pad, self.seg = 0, 1
        # summary self.writer
        self.summary_writer = None

        logger.info("ETRLModel using args %s" % str(args))
        np.random.seed(args.seed)

        self.object_predictor = None
        self.extractor = None
        self.vocab = None
        self.preprocessor = None

        self.input_dict = None
        self.cur_edh_instance = None
        vocab_obj_file = os.path.join(constants.ET_ROOT, constants.OBJ_CLS_VOCAB)
        logger.info("Loading object vocab from %s" % vocab_obj_file)
        self.vocab_obj = torch.load(vocab_obj_file)

        self.create_model(args)


    
    def create_model(self, args):
        os.makedirs(self.args.dout, exist_ok=True)
        model_path = os.path.join(self.args.model_dir, self.args.checkpoint)
        logger.info("Loading model from %s" % model_path)

        #self.et_model_args = model_util.load_model_args(model_path)
        dataset_info = data_util.read_dataset_info_for_inference(self.args.model_dir)
        train_data_name = self.args.data["train"][0]
        train_vocab = data_util.load_vocab_for_inference(self.args.model_dir, train_data_name)

        datasets = []
        ann_types = iter(args.data["ann_type"])
        for name, ann_type in zip(args.data["train"], ann_types):
            datasets.extend(train.load_data(name, args, ann_type))
        for name, ann_type in zip(args.data["valid"], ann_types):
            datasets.extend(train.load_data(name, args, ann_type, valid_only=True))
        # assign vocabs to datasets and check their sizes for nn.Embeding inits
        embs_ann, vocab_out = train.process_vocabs(datasets, args)

        # create the model
        ModelClass = import_module("alfred.model.{}".format(args.model)).Model
        self.model = ModelClass(self.args, embs_ann, vocab_out, self.pad, self.seg, for_inference=False)
        self.model = self.model.to(torch.device(args.device))

        self.optimizer = None
        self.prev_train_info = model_util.load_log(args.dout, stage="train")
        self.extractor = FeatureExtractor(
            archi=dataset_info["visual_archi"],
            device=args.device,
            checkpoint=args.visual_checkpoint,
            compress_type=dataset_info["compress_type"],
        )

        self.object_predictor = eval_util.load_object_predictor(self.args)
        self.vocab = {"word": train_vocab["word"], "action_low": self.model.vocab_out}
        self.preprocessor = Preprocessor(vocab=self.vocab)
    
    def load_data(self, edh_instance):

        feat = dict()
        feat["lang"] = GuidesEdhDataset.load_lang(edh_instance)
        feat["action"] = GuidesEdhDataset.load_action(edh_instance, self.vocab["action_low"])
        feat["obj_interaction_action"] = [
            a["obj_interaction_action"] for a in edh_instance["num"]["driver_actions_low"]
        ]
        feat["driver_actions_pred_mask"] = edh_instance["num"]["driver_actions_pred_mask"]
        feat["object"] = self.load_object_classes(edh_instance, self.vocab_obj)
    
        return feat
    
    def load_object_classes(self, task_json, vocab=None):
        """
        load object classes for interactive actions, helper function for load_data()
        from GuidesEDHDataset
        """
        object_classes = []
        for idx, action in enumerate(task_json["num"]["driver_actions_low"]):
            if self.args.compute_train_loss_over_history or task_json["num"]["driver_actions_pred_mask"][idx] == 1:
                if action["oid"] is not None:
                    object_class = action["oid"].split("|")[0]
                    object_classes.append(object_class if vocab is None else vocab.word2index(object_class))
        return object_classes

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):

        self.cur_edh_instance = data_util.process_traj(
            edh_instance, Path(os.path.join("test", edh_instance["instance_id"])), 0, self.preprocessor
        )
        feat_numpy = self.load_data(self.cur_edh_instance)
        _, self.input_dict, self.gt_dict = data_util.tensorize_and_pad(
            [(self.cur_edh_instance, feat_numpy)], self.args.device, constants.PAD
        )

        if not self.args.skip_edh_history and edh_history_images is not None and len(edh_history_images) > 0:
            img_features = self.extractor.featurize(edh_history_images, batch=32)
            self.model.frames_traj = img_features
            self.model.frames_traj = torch.unsqueeze(self.model.frames_traj, dim=0)
            self.model.action_traj = torch.tensor(
                [
                    self.vocab["action_low"].word2index(action["action_name"])
                    for action in edh_instance["driver_action_history"]
                ],
                device=self.args.device,
            )
            self.model.action_traj = torch.unsqueeze(self.model.action_traj, 0)

        return True


    def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
        """
        Sample function producing random actions at every time step. When running model inference, a model should be
        called in this function instead.
        :param img: PIL Image containing agent's egocentric image
        :param edh_instance: EDH instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param edh_name: EDH instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        img_feat = self.extractor.featurize([img], batch=1)
        self.input_dict["frames"] = img_feat

        prev_api_action = None
        if prev_action is not None and "action" in prev_action:
            prev_api_action = prev_action["action"]
        m_out = self.model.step(self.input_dict, self.vocab, prev_action=prev_api_action, is_train=True)

        step_out = {}
        for key, value in m_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]

        m_pred = model_util.extract_action_preds(
            step_out, self.model.pad, self.vocab["action_low"], clean_special_tokens=False
        )[0]
        action = m_pred["action"]

        obj = None
        if action in obj_interaction_actions and len(m_pred["object"]) > 0 and len(m_pred["object"][0]) > 0:
            obj = m_pred["object"][0][0]

        predicted_click = None
        if obj is not None:
            predicted_click = self.get_obj_click(obj, img)
        logger.debug("Predicted action: %s, obj = %s, click = %s" % (str(action), str(obj), str(predicted_click)))

        # Assume previous action succeeded if no better info available
        prev_success = True
        if prev_action is not None and "success" in prev_action:
            prev_success = prev_action["success"]

        # remove blocking actions
        action = self.obstruction_detection(action, prev_success, step_out, self.model.vocab_out)
        return m_out, action, predicted_click

    def get_obj_click(self, obj_class_idx, img):
        rcnn_pred = self.object_predictor.predict_objects(img)
        obj_class_name = self.object_predictor.vocab_obj.index2word(obj_class_idx)
        candidates = list(filter(lambda p: p.label == obj_class_name, rcnn_pred))
        if len(candidates) == 0:
            return [np.random.uniform(), np.random.uniform()]
        index = np.argmax([p.score for p in candidates])
        mask = candidates[index].mask[0]
        predicted_click = list(np.array(mask.nonzero()).mean(axis=1))
        predicted_click = [
            predicted_click[0] / mask.shape[1],
            predicted_click[1] / mask.shape[0],
        ]
        return predicted_click

    def obstruction_detection(self, action, prev_action_success, m_out, vocab_out):
        """
        change 'MoveAhead' action to a turn in case if it has failed previously
        """
        if action != "Forward" or prev_action_success:
            return action
        dist_action = m_out["action"][0][0].detach().cpu()
        idx_rotateR = vocab_out.word2index("Turn Right")
        idx_rotateL = vocab_out.word2index("Turn Left")
        action = "Turn Left" if dist_action[idx_rotateL] > dist_action[idx_rotateR] else "Turn Right"
        logger.debug("Blocking action is changed to: %s" % str(action))
        return action
