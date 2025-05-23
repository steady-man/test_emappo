import math
import random
import numpy as np
import json
from TR38901_scalar import calculate_path_loss


class EnvCore(object):

    def __init__(self):
        self.aiz = 36
        self.tilt = 19
        self.agent_num = 27
        self.grid_num = 100 * 100

        self.obs_dim = 71 + 64
        self.buffer_obs_dim = (3 + 4 + self.grid_num * 6)
        self.action_dim = self.aiz * self.tilt

        self.dense_people_num = 100
        self.dense_range = range(2500, 7000)
        self.sparse_people_num = 100
        self.sparse_range = list(range(2500)) + list(range(7000, 10000))
        self.cover_his = []
        self.reward_his = []

        with open('envs/10000_bs_info2.json', 'r') as file:
            js = file.read()
            bs = json.loads(js)
        self.bs = bs

        with open('envs/10000_grid_info2.json', 'r') as file:
            js = file.read()
            grid = json.loads(js)
        self.grid = grid

        with open('envs/10000_is_grid_bs2.json', 'r') as file:
            js = file.read()
            is_grid_bs = json.loads(js)
        self.is_grid_bs = np.array(is_grid_bs["10000_is_bs"])

        with open('envs/10000_is_grid_indoor2.json', 'r') as file:
            js = file.read()
            is_grid_indoor = json.loads(js)
        self.is_indoor = np.array(is_grid_indoor["10000_is_indoor"])

        self.people_1 = np.zeros(self.grid_num)
        self.people_2 = np.zeros(self.grid_num)
        self.people_3 = np.zeros(self.grid_num)

        self.n = math.pow(10, (-174 / 10 + math.log10(1.8e5)))

    def reset(self):

        sub_agent_obs = []
        s2 = np.concatenate((np.array([0, 0]), np.array([63, 6.5])))
        s0 = np.zeros(self.grid_num)
        s3 = np.concatenate((s0, self.is_indoor, self.is_grid_bs, s0, s0, s0))
        for i in self.bs:
            s1 = np.append(self.bs[i]["longlat_position"], self.bs[i]["transmit_power"])
            s = np.concatenate((s1, s2, s3))
            sub_agent_obs.append(s)
        return sub_agent_obs

    def step(self, actions):

        actions = np.array(actions)
        sig = []
        rate = []

        self.people_3 = self.people_2
        self.people_2 = self.people_1
        self.people_1 = np.zeros(self.grid_num)

        dense_people = np.array(random.sample(self.dense_range, self.dense_people_num))
        sparse_people = np.array(random.sample(self.sparse_range, self.sparse_people_num))
        self.people_1[dense_people] = 3
        self.people_1[sparse_people] = 1

        for i in self.grid:
            usr_lon = self.grid[i]['ll'][0]
            usr_lat = self.grid[i]['ll'][1]
            grid_is_indoor = self.grid[i]['in_door']
            sig_unit = -1000
            num_bs = 0

            for j in self.bs:
                act = np.argmax(actions[num_bs])
                bs_azi = int(act / self.tilt) * 360 / self.aiz
                bs_tilt = (act % self.tilt) * 90 / (self.tilt - 1)
                self.bs[j]['azi_tile'] = [bs_azi, bs_tilt]
                bs_lon = self.bs[j]['longlat_position'][0]
                bs_lat = self.bs[j]['longlat_position'][1]
                loss = calculate_path_loss(bs_lon, bs_lat, usr_lon, usr_lat, bs_azi, bs_tilt, is_indoor=grid_is_indoor)
                sig_bs = self.bs[j]["transmit_power"] - math.log10(45) - loss

                num_bs = num_bs + 1
                if sig_bs > sig_unit:
                    sig_unit = sig_bs
            sig.append(sig_unit)
            sig_power = math.pow(10, (sig_unit / 10))
            sg_rate = self.people_1[int(i)] * 1.8e5 * (math.log2(1 + sig_power / self.n))
            rate.append(sg_rate)

        json.dump({'sig': sig}, fp=open('./sig' + '.json', 'w'))
        sig = np.array(sig)

        cover = (np.sum(sig > -100)) / self.grid_num
        sig = sig > -100

        rate_average = sum(rate) / len(rate)

        reward = cover * 0.995 + rate_average * 0.005 * 0.0001
        print("覆盖率:{},吞吐量：{},奖励：{}".format(cover, rate_average, reward))

        self.cover_his.append(cover)
        self.reward_his.append(reward)
        json.dump({'cover_his': self.cover_his}, fp=open('./cover_his' + '.json', 'w'))
        json.dump({'reward_his': self.reward_his}, fp=open('./reward_his' + '.json', 'w'))

        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []

        s3 = np.concatenate((sig, self.is_indoor, self.is_grid_bs, self.people_1, self.people_2, self.people_3))
        for i in self.bs:
            s1 = np.append(self.bs[i]["longlat_position"], self.bs[i]["transmit_power"])
            s2 = np.concatenate((self.bs[i]['azi_tile'], np.array([63, 6.5])))

            s = np.concatenate((s1, s2, s3))
            sub_agent_obs.append(s)
            sub_agent_reward.append(reward)
            sub_agent_done.append(False)
            sub_agent_info.append({})
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
