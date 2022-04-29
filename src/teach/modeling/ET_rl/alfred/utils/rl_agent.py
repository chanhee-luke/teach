# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import isdir
from pathlib import Path
from typing import List, Type

from PIL import Image

from teach.dataset.definitions import Definitions
from teach.dataset.interaction import Interaction
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from teach.inference.actions import obj_interaction_actions
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
)

import torch
from tensorboardX import SummaryWriter
from alfred.utils import data_util, model_util

definitions = Definitions(version="2.0")
action_id_to_info = definitions.map_actions_id2info
logger = create_logger(__name__)


@dataclass
class RLAgentConfig:
    data_dir: str
    split: str
    output_dir: str
    images_dir: str
    model_args: List[str]
    metrics_file: str = "metrics.json"
    num_processes: int = 1
    max_init_tries: int = 3
    max_traj_steps: int = 1000
    max_api_fails: int = 30
    use_img_file: bool = False
    replay_timeout: int = 500


class RLAgent:
    def __init__(self, edh_instance_files, config: RLAgentConfig):
        self._edh_instance_files = edh_instance_files
        self._config = config

    def run(self, model):
        self._launch_processes(self._edh_instance_files, self._config, model)
        return self._load_metrics()

    def _load_metrics(self):
        metrics = dict()
        for metrics_file in RLAgent._get_metrics_files(self._config):
            if os.path.isfile(metrics_file):
                with open(metrics_file) as h:
                    thread_replay_status = json.load(h)
                metrics.update(thread_replay_status)
        return metrics

    @staticmethod
    def _get_metrics_files(config):
        return [
            RLAgent._get_metrics_file_name_for_process(x, config.metrics_file)
            for x in range(config.num_processes)
        ]

    @staticmethod
    def _launch_processes(edh_instance_files, config: RLAgentConfig, model):
        processes = []
        ers = []
        try:
            for process_index in range(config.num_processes):
                er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
                ers.append(er)
                process = RLAgent._launch_process(process_index, edh_instance_files, config, er, model)
                processes.append(process)
        finally:
            RLAgent._join_processes(processes)
            for er in ers:
                er.simulator.shutdown_simulator()

    @staticmethod
    def _launch_process(process_index, edh_instance_files, config: RLAgentConfig, er: EpisodeReplay, model):
        num_files = len(edh_instance_files)
        num_files_per_process = RLAgent._get_num_files_per_process(
            num_files=num_files, num_processes=config.num_processes
        )
        start_index, end_index = RLAgent._get_range_to_process(
            process_index=process_index,
            num_files_per_process=num_files_per_process,
            num_files=num_files,
        )

        files_to_process = edh_instance_files[start_index:end_index]

        #process = mp.Process(target=RLAgent._run, args=(process_index, files_to_process, config, er))

        #process.start()
        time.sleep(0.1)

        #return process

        return RLAgent._run(model, process_index, files_to_process, config, er)

        

    @staticmethod
    def _run(model, process_index, files_to_process, config: RLAgentConfig, er: EpisodeReplay):
        metrics_file = RLAgent._get_metrics_file_name_for_process(process_index, config.metrics_file)
        metrics = dict()

        # Setup model
        #model = config.model_class(process_index, config.num_processes, model_args=config.model_args)
        
        # initialize summary writer for tensorboardX
        model.summary_writer = SummaryWriter(log_dir=model.args.dout)

        # optimizer
        optimizer, schedulers = model_util.create_optimizer_and_schedulers(
            model.prev_train_info["progress"], model.args, model.model.parameters(), model.optimizer
        )

        # Main training loop
        for epoch in range(model.prev_train_info["progress"], model.args.epochs):
            model.model.train()
            for file_index, instance_file in enumerate(files_to_process):
                try:
                    instance_id, instance_metrics = RLAgent._run_edh_instance(instance_file, config, model, er, optimizer)
                    metrics[instance_id] = instance_metrics
                    save_dict_as_json(metrics, metrics_file)

                    logger.info(f"Instance {instance_id}, metrics: {instance_metrics}")
                    logger.info(f"Process {process_index} completed {file_index + 1} / {len(files_to_process)} instances")
                except Exception:
                    err_msg = f"exception happened for instance={instance_file}, continue with the rest"
                    logger.error(err_msg, exc_info=True)
                    continue

            # compute metrics for train
            logger.info("Computing train metrics...")
            metrics = {data: {k: sum(v) / len(v) for k, v in metr.items()} for data, metr in metrics.items()}
            stats = {
                "epoch": epoch,
                "general": {"learning_rate": optimizer.param_groups[0]["lr"]},
                **metrics,
            }

            # save the checkpoint
            logger.info("Saving models...")
            model_util.save_model(model.model, "model_{:02d}.pth".format(epoch), stats, optimizer=optimizer)
            model_util.save_model(model.model, "latest.pth", stats, symlink=True)

            # write averaged stats
            for loader_id in stats.keys():
                if isinstance(stats[loader_id], dict):
                    for stat_key, stat_value in stats[loader_id].items():
                        # for comparison with old epxs, maybe remove later
                        summary_key = "{}/{}".format(
                            loader_id.replace(":", "/").replace("lmdb/", "").replace(";lang", "").replace(";", "_"),
                            stat_key.replace(":", "/").replace("lmdb/", ""),
                        )
                        model.summary_writer.add_scalar(summary_key, stat_value, info["iters"]["train"])
            # dump the training info
            model_util.save_log(
                model.args.dout,
                progress=epoch + 1,
                total=model.args.epochs,
                stage="train",
                best_loss=info["best_loss"],
                iters=info["iters"],
            )
            model_util.adjust_lr(model.args, epoch, schedulers)
            logger.info(
                "{} epochs are completed, all the models were saved to: {}".format(model.args.epochs, model.args.dout)
            )

    @staticmethod
    def _load_edh_history_images(edh_instance, config: RLAgentConfig):
        image_file_names = edh_instance["driver_image_history"]
        image_dir = os.path.join(config.data_dir, "images", config.split, edh_instance["game_id"])
        return load_images(image_dir, image_file_names)

    @staticmethod
    def _clip_gtdict(gt_dict, length):
        tmp = {}
        print("here")
        print(length)
        print(gt_dict.keys())
        print("raw gt dict object", gt_dict["object"])
        for key, value in gt_dict.items():
            if key == "object":
                tmp[key] = gt_dict[key]
            else:
                tmp[key] = gt_dict[key][:,:length]
        
        print("there")
        print(tmp)
        exit()
        return tmp

    @staticmethod
    def _run_edh_instance(instance_file, config: RLAgentConfig, model, er: EpisodeReplay, optimizer):
        
        # Load edh instance from json
        edh_instance = RLAgent._load_edh_instance(instance_file)
        edh_check_task = create_task_thor_from_state_diff(edh_instance["state_changes"])
        game_file = RLAgent._get_game_file(edh_instance, config)

        metrics = create_new_traj_metrics(edh_instance)
        instance_id = edh_instance["instance_id"]
        logger.debug(f"Processing instance {instance_id}")
        
        # Set up simualator with the task
        try:
            init_success, er = with_retry(
                fn=lambda: RLAgent._initialize_episode_replay(
                    edh_instance, game_file, edh_check_task, config.replay_timeout, er
                ),
                retries=config.max_init_tries - 1,
                check_first_return_value=True,
            )
        except Exception:
            init_success = False
            logger.error(f"Failed to initialize episode replay for instance={instance_id}", exc_info=True)

        # Set up model with EDH instance
        edh_history_images = None
        try:
            if not config.use_img_file:
                edh_history_images = RLAgent._load_edh_history_images(edh_instance, config)
        except Exception:
            init_success = False
            logger.error(f"Failed to load_edh_history_images for {instance_id}", exc_info=True)

        metrics["init_success"] = init_success
        if not init_success:
            return edh_instance["instance_id"], metrics

        model_started_success = False
        try:
            model_started_success = model.start_new_edh_instance(edh_instance, edh_history_images, instance_file)
        except Exception:
            model_started_success = False
            metrics["error"] = 1
            logger.error(f"Failed to start_new_edh_instance for {instance_id}", exc_info=True)
            exit()

        # Training loop
        if model_started_success:
            prev_action = None
            er.simulator.is_record_mode = True
            pred_actions = list()

            traj_steps_taken = 0
            model_outs, losses_train = {}, {}
            batch_idx = 0
            
            # Init the logs
            rewards = []
            hidden_states = []
            policy_log_probs = []
            masks = []
            entropys = []

            for action_idx in range(config.max_traj_steps):
                traj_steps_taken += 1
                try:
                    img = RLAgent._get_latest_ego_image(er)
                    image_name = RLAgent._save_image(config, edh_instance, img, traj_steps_taken)
                    model_out, action, obj_relative_coord = model.get_next_action(
                        img, edh_instance, prev_action, image_name, instance_file
                    )
                    # print("model out")
                    # print(model_out)
                    # print(model_out.keys())
                    # print("model outout")
                    model_outs[batch_idx] = model_out

                    # Don't execute action, execute GT action
                    # print("action:", action)
                    # print("gtdict", model.gt_dict)
                    # exit()
                    # print("after model out")

                    # print(model_out["action"].size())
                    # print(model_out["action"].view(-1, model_out["action"].shape[-1]).size())
                    # print(model.gt_dict["action"].size())
                    # print(model.gt_dict["action"][:,:model_outs[batch_idx]["action"].size(1)].size())
                    # print(model_out["action"])
                    # print(model.gt_dict["action"][:,:model_outs[batch_idx]["action"].size(1)])
                    #
                    # print("length of action:", model_outs[batch_idx]["action"].size(1))
                    # print(model_outs[batch_idx]["object"].size())
                    action_length = model_outs[batch_idx]["action"].size(1)
                    gt_dict = RLAgent._clip_gtdict(model.gt_dict, action_length)
                    # for key, value in gt_dict.items():
                    #     if key == "object":
                    #         print(key, torch.cat(value, dim=0).size())
                    #     else:
                    #         print(key, value.size())
                    
                    # compute losses
                    losses_train = model.model.compute_loss(
                        model_outs,
                        {batch_idx: gt_dict},
                    )

                    #TODO WIP
                    if train_rl:
                        dist = np.zeros(batch_size, np.float32)
                        reward = np.zeros(batch_size, np.float32)
                        mask = np.ones(batch_size, np.float32)
                        for i, ob in enumerate(perm_obs):
                            dist[i] = ob['distance']
                            path_act = [vp[0] for vp in traj[i]['path']]

                            if ended[i]:
                                reward[i] = 0.0
                                mask[i] = 0.0
                            else:
                                action_idx = cpu_a_t[i]
                                # Target reward
                                if action_idx == -1:                              # If the action now is end
                                    if dist[i] < 3.0:                             # Correct
                                        reward[i] = 2.0 + ndtw_score[i] * 2.0
                                    else:                                         # Incorrect
                                        reward[i] = -2.0
                                else:                                             # The action is not end
                                    # Path fidelity rewards (distance reward)
                                    reward[i] = - (dist[i] - last_dist[i])
                                    if reward[i] > 0.0:                           # Quantification
                                        reward[i] = 1.0
                                    elif reward[i] < 0.0:
                                        reward[i] = -1.0
                                    else:
                                        raise NameError("The action doesn't change the move")
                                    # Miss the target penalty
                                    if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                        reward[i] -= (1.0 - last_dist[i]) * 2.0
                                        
                                    # Interaction reward:
                                    is_correct_inter = check_inter_action(model_outs[batch_idx]["action"], gt_dict)
                                    if is_correct_inter:                           
                                        reward[i] += 2.0
                                    else:
                                        reward[i] -= 2.0
                                    
                            rewards.append(reward)
                            masks.append(mask)
                            last_dist[:] = dist


                    # compute metrics
                    for dataset_name in losses_train.keys():
                        self.model.compute_metrics(
                            model_outs[dataset_name],
                            batches[dataset_name][2],
                            metrics["train:" + dataset_name],
                            self.args.compute_train_loss_over_history,
                        )
                        for key, value in losses_train[dataset_name].items():
                            metrics["train:" + dataset_name]["loss/" + key].append(value.item())
                        metrics["train:" + dataset_name]["loss/total"].append(sum_loss.detach().cpu().item())

                    # Determine next model inputs
                    if self.feedback == 'teacher':
                        a_t = target                 # teacher forcing
                    elif self.feedback == 'argmax':
                        _, a_t = logit.max(1)        # student forcing - argmax
                        a_t = a_t.detach()
                        log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                        policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
                    elif self.feedback == 'sample':
                        probs = F.softmax(logit, 1)  # sampling an action from model
                        c = torch.distributions.Categorical(probs)
                        self.logs['entropy'].append(c.entropy().sum().item())            # For log
                        entropys.append(c.entropy())                                     # For optimization
                        a_t = c.sample().detach()
                        policy_log_probs.append(c.log_prob(a_t))
                    else:
                        print(self.feedback)
                        sys.exit('Invalid feedback option')

                    # Calculate imitation learning loss
                    optimizer.zero_grad()
                    sum_loss = sum([sum(loss.values()) for name, loss in losses_train.items()])
                    sum_loss.backward()
                    optimizer.step()

                    # Execute action in the simulator
                    step_success = RLAgent._execute_action(er.simulator, action, obj_relative_coord)
                    RLAgent._update_metrics(metrics, action, obj_relative_coord, step_success)
                    prev_action = {"action": action, "obj_relative_coord": obj_relative_coord}
                    pred_actions.append(prev_action)
                except Exception as e:
                    logger.error(
                        f"_run_edh_instance Exception: {str(e)} for instance_id={instance_id}, "
                        f"traj_steps_taken={traj_steps_taken}",
                        exc_info=True,
                    )
                    metrics["error"] = 1
                    #break
                    exit()
                if RLAgent._should_end_inference(action, metrics, config.max_api_fails):
                    break

        # # RL after last action in A2C
        if train_rl:
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.model.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.model.critic(hidden_states[t])
                a_ = (r_ - v_).detach()
        # a = r + gamma * V(s') - V(s)
        #
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            #self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

            # Do the backprop for RL loss
            rl_loss.backward()

        # Calculate training metrics
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = RLAgent._check_episode_progress(er, edh_check_task)

        metrics_diff = evaluate_traj(
            success,
            edh_instance,
            traj_steps_taken,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        )
        metrics.update(metrics_diff)

        # Uncomment this to get the outputs for individual trajectory as jsons
        # os.makedirs(config.output_dir, exist_ok=True)
        # pred_actions_file = os.path.join(config.output_dir, "pred_actions__" + instance_id + ".json")
        # with open(pred_actions_file, "w") as handle:
        #     json.dump(pred_actions, handle)

        # er.simulator.dir_out = config.output_dir
        # # output_file = os.path.join(config.output_dir, "inference__" + instance_id + ".json")
        # # er.simulator.save(file_name=output_file)

        return instance_id, metrics


    @staticmethod
    def _check_episode_progress(er, task):
        (
            _,
            success,
            _,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = er.simulator.check_episode_progress(task)
        return success, final_goal_conditions_total, final_goal_conditions_satisfied

    @staticmethod
    # Episode reply == simulator
    def _initialize_episode_replay(edh_instance, game_file, task, replay_timeout, er: EpisodeReplay):
        start_time = time.perf_counter()
        er.set_episode_by_fn_and_idx(game_file, 0, 0)
        edh_interactions = list()
        for interaction in edh_instance["interactions"][: edh_instance["pred_start_idx"]]:
            action = action_id_to_info[interaction["action_id"]]
            edh_interactions.append(Interaction.from_dict(interaction, action["action_type"]))
        er.episode.interactions = edh_interactions

        init_success = False
        with ThreadPoolExecutor() as tp:
            future = tp.submit(er._set_up_new_episode, task=task, obs_dir=None, turn_on_lights=False)
            logger.info(f"Started episode replay with timeout: {replay_timeout} sec")
            init_success, _ = future.result(timeout=replay_timeout)
        init_success = True

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Elapsed time for episode replay: {elapsed_time}")

        return init_success, er if init_success else None

    @staticmethod
    def _get_latest_ego_image(er):
        return Image.fromarray(er.simulator.get_latest_images()["ego"])

    @staticmethod
    def _execute_action(simulator, action, obj_relative_coord):
        if action == "Stop":
            return True

        if action in obj_interaction_actions:
            y = obj_relative_coord[0]
            x = obj_relative_coord[1]
            step_success, _, _ = simulator.apply_object_interaction(action, 1, x, y)
            return step_success

        step_success, _, _ = simulator.apply_motion(action, 1)
        return step_success

    @staticmethod
    def _get_game_file(edh_instance, config: RLAgentConfig):
        return os.path.join(
            config.data_dir,
            "games",
            config.split,
            f"{edh_instance['game_id']}.game.json",
        )

    @staticmethod
    def _get_metrics_file_name_for_process(process_index, metrics_file):
        return f"{metrics_file}.json.{process_index}"

    @staticmethod
    def _update_metrics(metrics, action, obj_relative_coord, step_success):
        metrics["pred_actions"].append((action, obj_relative_coord))

        if action == "Stop":
            metrics["predicted_stop"] = 1

        if not step_success:
            metrics["num_api_fails"] += 1

    @staticmethod
    def _should_end_inference(action, metrics, max_api_fails):
        return action == "Stop" or metrics["num_api_fails"] >= max_api_fails

    @staticmethod
    def _load_edh_instance(instance_file):
        with open(instance_file) as handle:
            edh_instance = json.load(handle)
        return edh_instance

    @staticmethod
    def _get_range_to_process(process_index, num_files_per_process, num_files):
        start_index = process_index * num_files_per_process
        end_index = min(start_index + num_files_per_process, num_files)
        return start_index, end_index

    @staticmethod
    def _get_num_files_per_process(num_files, num_processes):
        return int(num_files / num_processes) + 1

    @staticmethod
    def _join_processes(processes):
        for process in processes:
            process.join()

    @staticmethod
    def _save_image(config, edh_instance, img, traj_steps_taken):
        image_name = f"img__{edh_instance['instance_id']}_{traj_steps_taken}.jpeg"
        if config.use_img_file:
            RLAgent._save_image_sync(img, image_name, config)
        else:
            RLAgent._save_image_async(img, image_name, config)
        return image_name

    @staticmethod
    def _save_image_async(img, image_name, config: RLAgentConfig):
        process = mp.Process(target=RLAgent._save_image_sync, args=(img, image_name, config))
        process.start()
        return image_name

    @staticmethod
    def _save_image_sync(img, image_name, config: RLAgentConfig):
        if not isdir(config.images_dir):
            Path(config.images_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(config.images_dir, image_name)
        img.save(image_path)
        return image_name
