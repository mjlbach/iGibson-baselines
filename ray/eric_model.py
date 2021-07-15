"""
Example showing how to wrap the iGibson class using ray for rllib.
Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
are installed *without* MKL support.

This example requires ray to be installed with rllib support, and pytorch to be installed:
    `pip install torch "ray[rllib]"`

Note: rllib only supports a single observation modality:
"""
import argparse

from igibson.envs.igibson_env import iGibsonEnv

import os

from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.tune.registry import register_env

import ray
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import igibson


# Eric's structure
# MultiInputActorCriticPolicy(
#   (features_extractor): CombinedExtractor(
#     (extractors): ModuleDict(
#       (depth): NatureCNN(
#         (cnn): Sequential(
#           (0): Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
#           (1): ReLU()
#           (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
#           (3): ReLU()
#           (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#           (5): ReLU()
#           (6): Flatten()
#         )
#         (linear): Sequential(
#           (0): Linear(in_features=11264, out_features=256, bias=True)
#           (1): ReLU()
#         )
#       )
#       (rgb): NatureCNN(
#         (cnn): Sequential(
#           (0): Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4))
#           (1): ReLU()
#           (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
#           (3): ReLU()
#           (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
#           (5): ReLU()
#           (6): Flatten()
#         )
#         (linear): Sequential(
#           (0): Linear(in_features=11264, out_features=256, bias=True)
#           (1): ReLU()
#         )
#       )
#       (scan): Flatten()
#       (task_obs): Flatten()
#     )
#   )
#   (mlp_extractor): MlpExtractor(
#     (shared_net): Sequential()
#     (policy_net): Sequential(
#       (0): Linear(in_features=744, out_features=64, bias=True)
#       (1): Tanh()
#       (2): Linear(in_features=64, out_features=64, bias=True)
#       (3): Tanh()
#     )
#     (value_net): Sequential(
#       (0): Linear(in_features=744, out_features=64, bias=True)
#       (1): Tanh()
#       (2): Linear(in_features=64, out_features=64, bias=True)
#       (3): Tanh()
#     )
#   )
#   (action_net): Linear(in_features=64, out_features=2, bias=True)
#   (value_net): Linear(in_features=64, out_features=1, bias=True)
# )
# ray.init(local_mode=True)
ray.init()


class FC(nn.Module):

    def __init__(self, in_shape=232, out_shape=2):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_shape, 64, bias=True)
        self.fc2 = nn.Linear(64, out_shape, bias=True)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class iGibsonPPOModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.value_head = FC(in_shape=232, out_shape=1)
        self.action_head = FC(in_shape=232, out_shape=2)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        obs["task_obs"] = obs["task_obs"].float().flatten(start_dim=1)
        obs["scan"] = obs["scan"].float().flatten(start_dim=1)

        policy_input = torch.cat([
            obs["task_obs"],
            obs["scan"],
            ],
            dim=1
        )

        self._value_out = torch.flatten(self.value_head(policy_input))
        action_out = self.action_head(policy_input)
        print(action_out.shape, self._value_out.shape)

        return action_out, []

    def value_function(self):
        return self._value_out



class iGibsonRayEnv(iGibsonEnv):
    def __init__(self, env_config):
        super().__init__(
                config_file=env_config['config_file'],
                mode=env_config['mode'],
                action_timestep=env_config['action_timestep'],
                physics_timestep=env_config['physics_timestep'],
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default=os.path.join(igibson.root_path, "examples", "configs", "turtlebot_point_nav.yaml"),
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument(
        '--ray_mode',
        default="train",
        help='Whether to run ray in train or test mode')
    parser.add_argument(
        '--local_dir',
        default=None,
        help='Directory where to save model logs and default checkpoints')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        choices=[None, "PROMPT"],
        help='Whether to resume the experiment. Note this does *not* restore the checkpoint, just re-uses the same config/log.')
    parser.add_argument(
        '--restore_checkpoint',
        default=None,
        help='Checkpoint to force restore')
    parser.add_argument('--exp_name',
                        default='my_igibson_run',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    
    args = parser.parse_args()

    ModelCatalog.register_custom_model("iGibsonPPOModel", iGibsonPPOModel)
    register_env("iGibsonEnv", lambda c: iGibsonRayEnv(c))
    # See here for stable-baselines3 defaults
    #https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/ppo/ppo.py#L69-L91
    # Things currently missing/not confirmed to be equal
    # num_epochs
    # clip_range
    # clip_range_vf
    # not sure GAE/lambda matches
    # not sure if observation filtering matches
    config = {
        "env" : "iGibsonEnv",
        "env_config" : {
            "config_file": args.config,
            "mode": args.mode,
            # matches eric
            "action_timestep": 1.0 / 10.0,
            # matches eric
            "physics_timestep": 1.0 / 120.0
        },
        #ray specific
        "num_gpus": 1,
        # ray specific
        "num_cpus_for_driver": 5,
        # "remote_worker_envs": True,
        # number of workers == number of environments, confirmed match
        "num_workers": 8,
        # ray specific
        "num_envs_per_worker": 1,
        # ray specific
        "num_cpus_per_worker": 2,
        # normalization, none for now
        # "observation_filter": "MeanStdFilter",
        "horizon": 3000,
        # equivalent to buffer size (num_envs * n_steps)
        "train_batch_size": 16384,
        # equivalent to learning rate, confirmed match
        "lr": 3e-4,
        # equivalent to n_steps, confirmed match
        "rollout_fragment_length": 2048,
        # equivalent to batch_size, confirmed match
        "sgd_minibatch_size": 64,
        # equivalent to num_epochs... maybe??
        "num_sgd_iter": 10,
        # kl_target (none in stable-basdelines3)
        #https://github.com/ray-project/ray/blob/06f6f4e0ecb4aa549af274aebc5e6028b9d866e3/rllib/agents/ppo/ppo_torch_policy.py#L185
        # kl_coeff (None in stable-baselinse3
        #https://github.com/ray-project/ray/blob/06f6f4e0ecb4aa549af274aebc5e6028b9d866e3/rllib/agents/ppo/ppo_torch_policy.py#L183
        #https://github.com/DLR-RM/stable-baselines3/blob/2fa06ae8d24662a40a7c247dd96625bebf182dce/stable_baselines3/ppo/ppo.py#L76
        "gamma": 0.99,
        "use_gae": True,
        # Equivalent to GAE lambda?
        "lambda": 0.95,
        # MISSING CLIP_RANGE
        # MISSING CLIP_RANGE_VF
        #https://github.com/ray-project/ray/blob/06f6f4e0ecb4aa549af274aebc5e6028b9d866e3/rllib/agents/ppo/ppo_torch_policy.py#L245
        "entropy_coeff": 0.0,
        "vf_loss_coeff": 0.5,
        # MISSING MAX_GRAD_NORM
        # Everything else from this point on set to false or none in constructor
        "model": {
            "custom_model": "iGibsonPPOModel",
        },
        "framework": "torch"
    }
    stop={"training_iteration": 100000}
    if args.resume is not None:
        assert args.restore_checkpoint is not None, "Error: When resuming must provide explicit path to checkpoint"

    results = tune.run("PPO",
        config=config,
        verbose=2,
        restore=args.restore_checkpoint,
        name=args.exp_name,
        local_dir=args.local_dir,
        checkpoint_freq=100, 
        resume=args.resume
)
