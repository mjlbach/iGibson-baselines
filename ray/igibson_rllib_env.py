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

# ray.init(local_mode=True)
ray.init()

class ConvNet1D(nn.Module):

    def __init__(self):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool1d(F.relu(self.conv1(x)), 5)
        # If the size is a square, you can specify with a single number
        x = F.max_pool1d(F.relu(self.conv2(x)), 5)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet2D(nn.Module):

    def __init__(self, channels=1):
        super(ConvNet2D, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(15984, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FC(nn.Module):

    def __init__(self, input_size, output_size=10):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class iGibsonPPOModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.rgb_encoder = ConvNet2D(channels = 3)
        self.depth_encoder = ConvNet2D(channels = 1)
        self.scan_encoder = ConvNet1D()
        self.task_encoder = FC(4)

        self.value_head = FC(40, 1)
        self.action_head = FC(40, 2)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        obs["task_obs"] = obs["task_obs"].float()
        obs["rgb"] = obs["rgb"].float().permute(0, 3, 1, 2)
        obs["depth"] = obs["depth"].float().permute(0, 3, 1, 2)
        obs["scan"] = obs["scan"].float().permute(0, 2, 1)

        rgb_obs_encoding = self.rgb_encoder(obs['rgb'])
        depth_obs_encoding = self.depth_encoder(obs['depth'])
        scan_obs_encoding = self.scan_encoder(obs['scan'])
        task_obs_encoding = self.task_encoder(obs['task_obs'])

        policy_input = torch.cat([
            rgb_obs_encoding,
            depth_obs_encoding,
            scan_obs_encoding,
            task_obs_encoding
            ],
            dim=1
        )

        self._value_out = torch.flatten(self.value_head(policy_input))
        action_out = self.action_head(policy_input)

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
        "episode_horizon": 3000,
        # equivalent to buffer size (num_envs * n_steps)
        "train_batch_size": 16384,
        # equivalent to n_steps, confirmed match
        "rollout_fragment_length": 2048,
        # equivalent to batch, confirmed match
        "sgd_minibatch_size": 64,
        # equivalent to learning rate, confirmed match
        "lr": 3e-4,
        "model": {
            "custom_model": "iGibsonPPOModel",
        },
        "framework": "torch"
    }
    stop={"training_iteration": 1000000}
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
