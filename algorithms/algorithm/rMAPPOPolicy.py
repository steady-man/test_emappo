import json
import numpy as np
import torch
from torch import nn
from algorithms.utils.util import init, check
from algorithms.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule
from algorithms.utils.Unet import UNet
from algorithms.utils.mlp import MLPLayer
from algorithms.utils.DIFFPool import *
from algorithms.utils.graph_create import *


class RMAPPOPolicy:

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.episode_length = args.episode_length

        self.use_pretrain_model = args.use_pretrain_model
        self.agent_num = 27
        self.row = 100
        self.column = 100
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.diffpool = DiffPoolNet(6, self.row * self.column).to(device)
        self.diffpool_graph = check(create_adjacency_diffpool()).to(**self.tpdv)
        self.Unet = UNet(6, 64).to(device)

        if self.use_pretrain_model == True:
            self.actor.load_state_dict(
                torch.load("results/MyEnv/MyEnv/mappo/check/run1/models/actor.pt"))
            self.critic.load_state_dict(
                torch.load("results/MyEnv/MyEnv/mappo/check/run1/models/critic.pt"))
            self.Unet.load_state_dict(
                torch.load("results/MyEnv/MyEnv/mappo/check/run1/models/Unet.pt"))
            print('导入模型成功')

        with open('envs/10000_bs_info2.json', 'r') as file:
            js = file.read()
            bs = json.loads(js)
        self.bs = bs

        self.actor_optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), },
             {'params': self.Unet.parameters()},
             {'params': self.diffpool.parameters()}],
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)

        self.critic_optimizer = torch.optim.Adam([{'params': self.critic.parameters(), },
                                                  {'params': self.Unet.parameters()},
                                                  {'params': self.diffpool.parameters()}],
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):

        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):

        obs = check(obs).to(**self.tpdv)
        unet_obs = obs[0][7:].reshape(1, 6, self.row, self.column)
        unet_result = self.Unet(unet_obs)
        unet_result = unet_result.cpu().numpy()

        diff_obs = obs[0][7:].reshape(6, -1).T
        diff_result = self.diffpool(diff_obs, self.diffpool_graph)
        diff_result = np.tile(torch.squeeze(diff_result, dim=0).cpu().numpy(), (27, 1))

        bs_info = np.zeros(64)
        for i, info in enumerate(self.bs):
            row = int(self.bs[info]["poi_grid"] / self.column)
            col = self.bs[info]["poi_grid"] % self.column
            bs_info = np.vstack((bs_info, unet_result[0, :, row, col]))

        final_obs = np.concatenate((obs[:, :7].cpu(), bs_info[1:, :], diff_result), axis=1)
        final_obs = check(final_obs).to(**self.tpdv)

        actions, action_log_probs, rnn_states_actor = self.actor(final_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(final_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):

        obs = check(cent_obs).to(**self.tpdv)
        unet_obs = obs[0][7:].reshape(1, 6, self.row, self.column)
        unet_result = self.Unet(unet_obs)
        unet_result = unet_result.cpu().numpy()

        diff_obs = obs[0][7:].reshape(6, -1).T
        diff_result = self.diffpool(diff_obs, self.diffpool_graph)
        diff_result = np.tile(torch.squeeze(diff_result, dim=0).cpu().numpy(), (27, 1))

        bs_info = np.zeros(64)
        for i, info in enumerate(self.bs):
            row = int(self.bs[info]["poi_grid"] / self.column)
            col = self.bs[info]["poi_grid"] % self.column
            bs_info = np.vstack((bs_info, unet_result[0, :, row, col]))

        final_obs = np.concatenate((obs[:, :7].cpu(), bs_info[1:, :], diff_result), axis=1)
        final_obs = check(final_obs).to(**self.tpdv)

        values, _ = self.critic(final_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):

        obs = check(cent_obs).to(**self.tpdv)
        unet_obs = obs[[i * self.agent_num for i in range(self.episode_length)]] \
            [:, 7:].reshape(self.episode_length, 6, self.row, self.column)
        unet_result = self.Unet(unet_obs)

        diff_obs = obs[[i * self.agent_num for i in range(self.episode_length)]] \
            [:, 7:].reshape(self.episode_length, 6, -1).transpose(-1, -2)

        diff_result = process_batch_diffpool(self.diffpool, diff_obs, self.diffpool_graph)

        bs_info = torch.zeros(64).to(**self.tpdv).unsqueeze(0)
        for i in range(self.episode_length):
            for j, info in enumerate(self.bs):
                row = int(self.bs[info]["poi_grid"] / self.column)
                col = self.bs[info]["poi_grid"] % self.column
                bs_info = torch.cat((bs_info, unet_result[i, :, row, col].unsqueeze(0)), 0)

        final_obs = torch.cat((obs[:, :7], bs_info[1:, :], torch.squeeze(diff_result, dim=1)), 1)
        final_obs = check(final_obs).to(**self.tpdv)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(final_obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(final_obs, rnn_states_critic, masks)

        return values, action_log_probs, dist_entropy, final_obs

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        unet_obs = obs[0][7:].reshape(1, 6, self.row, self.column)
        unet_result = self.Unet(unet_obs)
        unet_result = unet_result.cpu().numpy()

        diff_obs = obs[0][5:].reshape(6, -1).T
        diff_result = self.diffpool(diff_obs, self.diffpool_graph)
        diff_result = np.tile(torch.squeeze(diff_result, dim=0).cpu().numpy(), (27, 1))

        bs_info = np.zeros(64)
        for i, info in enumerate(self.bs):
            row = int(self.bs[info]["poi_grid"] / self.column)
            col = self.bs[info]["poi_grid"] % self.column
            bs_info = np.vstack((bs_info, unet_result[0, :, row, col]))

        final_obs = np.concatenate((obs[:, :7].cpu(), bs_info[1:, :], diff_result), axis=1)
        final_obs = check(final_obs).to(**self.tpdv)
        actions, _, rnn_states_actor = self.actor(final_obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
