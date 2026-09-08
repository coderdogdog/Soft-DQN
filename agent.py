# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from net import Soft_Qnet


class ReplayBuffer:
    def __init__(self, max_len: int, state_dim: int):
        self.max_len = max_len

        self.next_idx = 0           # 下一个要写入的位置
        self.count = 0              # 当前已有数据量

        # 为每个数据字段预分配 NumPy 数组
        self.states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_len, 1), dtype=np.float32)
        self.rewards = np.zeros((max_len, 1), dtype=np.float32)
        self.next_states = np.zeros((max_len, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_len, 1), dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        idx = self.next_idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        self.count = min(self.count + 1, self.max_len)
        self.next_idx = (self.next_idx + 1) % self.max_len

    def sample(self, batch_size):
        """随机采样一个 batch"""
        # 从 [0, self.count) 范围内随机选取 batch_size 个索引
        # replace=False 的意思是：不放回抽样 不抽一样的
        indices = np.random.choice(self.count, batch_size, replace=False)
        # 直接从数组中根据索引取值
        return (self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices])


# --------------------------------------- 智能体设置 ---------------------------------------------
class Soft_DQN_Agent:
    def __init__(self, state_dim: int, action_dim: int, hid_dim: int,
                 gamma=0.99,
                 lr=5e-4,
                 lr_alpha=5e-4,
                 update_tau=0.01,
                 epsilon=0.02,          # 作为“安全网”，防止因Q值估计不准导致策略过早陷入局部最优
                 clip_norm=1.0,
                 alpha_min=1e-3,        # 温度系数下限，防止 α→0 时 q/α 数值溢出
                 alpha_max=1e2,         # 温度系数上限，防止 α 过大导致软目标膨胀
                 device="cpu"):

        self.Q_net1 = Soft_Qnet(state_dim, action_dim, hid_dim).to(device)
        self.targetQ_net1 = Soft_Qnet(state_dim, action_dim, hid_dim).to(device)
        self.targetQ_net1.load_state_dict(self.Q_net1.state_dict())

        self.Q_net2 = Soft_Qnet(state_dim, action_dim, hid_dim).to(device)
        self.targetQ_net2 = Soft_Qnet(state_dim, action_dim, hid_dim).to(device)
        self.targetQ_net2.load_state_dict(self.Q_net2.state_dict())

        self.optimizer = optim.Adam(
            list(self.Q_net1.parameters()) + list(self.Q_net2.parameters()),
            lr=lr
        )

        # ----------------- 温度参数 alpha 训练 --------------------------
        # 目标熵，离散动作空间一般用 目标熵 = 0.98 × log(|A|)
        # 比较好的方法是 目标熵设置的比平均熵小，让 alpha 逐渐降低
        self.target_entropy = 0.98 * math.log(action_dim)
        # self.target_entropy = 0.5 * math.log(action_dim)

        # 温度系数初始化
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self.alpha = self.log_alpha.exp()

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # 温度系数上下限
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # -----------------------------------------------------------------

        # 超参数
        self.epsilon = epsilon              # epsilon 作为“安全网”，防止因Q值估计不准导致策略过早陷入局部最优
        self.gamma = gamma                  # 折扣因子
        self.update_tau = update_tau        # 目标网络软更新系数
        # 梯度裁剪
        self.clip_norm = clip_norm

        self.action_dim = action_dim
        self.dvc = device

        self.train_num = 0

    # ----------------------------------- 选择动作 -----------------------------------
    def select_action(self, state, explore=True):

        if explore and np.random.rand() < self.epsilon:   # 生成[0, 1) 区间的均匀分布随机浮点数
            action = np.random.randint(0, self.action_dim)
            return action

        # 无论是否 explore，只要没触发随机，就走 Softmax
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.dvc)
            # 如果环境噪声大，想要动作价值估计更平滑，可以使用 (Q1+Q2)/2.0
            # 这里直接使用第一个 Q 网络的值
            q_values = self.Q_net1(state)        # shape: [1, action_dim]

            if explore:
                # 训练模式：按Softmax概率采样
                # 注意：温度系数 alpha 控制分布的"柔软度"
                # alpha > 0
                # 当 alpha 很大时，分布趋近于均匀分布（强探索）
                # 当 alpha 很小时，分布趋近于 one-hot（强利用）
                logits = q_values / self.alpha
                probs = F.softmax(logits, dim=1)
                dist = Categorical(probs)
                action = dist.sample().item()
                return action
            else:
                return q_values.argmax(dim=1).item()

    # ---------- 计算 target -----------
    def compute_soft_target(self, next_states, dones):
        """
        计算软目标值 (Soft Target)
        核心：使用 Log-Sum-Exp 实现软最大化
        V_soft(s') = alpha * log( sum_a' exp( Q_target(s', a') / alpha ) )
        """
        with torch.no_grad():
            # 获取目标网络对所有动作的Q值: [batch_size, action_dim]
            next_q1 = self.targetQ_net1(next_states)
            next_q2 = self.targetQ_net2(next_states)

            # 核心：log-sum-exp 计算软价值
            # 使用 torch.logsumexp 保证数值稳定性
            # 软价值计算（核心：先取两个Q的最小值，再做LogSumExp）
            # 注意：最小值要作用在原始Q上，再除以alpha
            min_next_q = torch.min(next_q1, next_q2)
            soft_values = self.alpha * torch.logsumexp(min_next_q / self.alpha, dim=1, keepdim=True)

            # 对于终止状态，软价值为0
            soft_values = soft_values * (1 - dones)
        return soft_values

    # ---------------------------------- 更新 -----------------------------------------
    def update(self, replay_buffer, batch_size, writer):
        self.train_num += 1

        # ---------- 从经验池中采样 ----------
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        # 转换为 PyTorch Tensor
        states = torch.tensor(state, dtype=torch.float32).to(self.dvc)
        actions = torch.tensor(action, dtype=torch.long).to(self.dvc)
        rewards = torch.tensor(reward, dtype=torch.float32).to(self.dvc)
        next_states = torch.tensor(next_state, dtype=torch.float32).to(self.dvc)
        dones = torch.tensor(done, dtype=torch.float32).to(self.dvc)

        # 使用了 Clipped Double-Q Learning
        # 计算当前Q值
        current_q1 = self.Q_net1(states).gather(1, actions)
        current_q2 = self.Q_net2(states).gather(1, actions)

        # 计算目标Q值 (软贝尔曼方程)
        soft_target = self.compute_soft_target(next_states, dones)
        target_q = rewards + self.gamma * soft_target

        # 计算损失 (Huber Loss 更鲁棒)
        loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        grad_norm1 = torch.nn.utils.clip_grad_norm_(self.Q_net1.parameters(), max_norm=self.clip_norm)
        grad_norm2 = torch.nn.utils.clip_grad_norm_(self.Q_net2.parameters(), max_norm=self.clip_norm)
        self.optimizer.step()

        # ------------- 温度系数alpha更新 -------------------
        with torch.no_grad():
            q1_all = self.Q_net1(states)        # [B, A]
            logits = q1_all / self.alpha
            log_probs = F.log_softmax(logits, dim=1)        # log π(a|s)
            entropy = -(log_probs.exp() * log_probs).sum(dim=1)         # 每个状态的精确熵 [B]
            entropy_mean = entropy.mean()  # E_{s~B}[ H(π(·|s)) ]

        # 最小化 J(α) = α·(E[H] − H̄)：熵低于目标 → 梯度为负 → α 上升；  高于目标 → α 下降
        alpha_loss = self.alpha * (entropy_mean - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 温度系数上下限保护：α→0 会使 q/α 溢出，α 过大会使软目标按 α·log|A| 膨胀
        # with torch.no_grad():
        #     self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))
        self.alpha = self.log_alpha.exp()
        # --------------------------------------------------

        # 软更新目标网络
        for param, target_param in zip(self.Q_net1.parameters(), self.targetQ_net1.parameters()):
            target_param.data.copy_(self.update_tau * param.data + (1 - self.update_tau) * target_param.data)

        for param, target_param in zip(self.Q_net2.parameters(), self.targetQ_net2.parameters()):
            target_param.data.copy_(self.update_tau * param.data + (1 - self.update_tau) * target_param.data)

        # -------------------------------- 记录训练日志 ------------------------------------
        if self.train_num % 50 == 0:        # 控制记录频率

            with torch.no_grad():
                # 计算当前状态的平均 Q 值
                q_mean = current_q1.mean().item()

            # TensorBoard 记录
            writer.add_scalar("Q_Net/Q_mean", q_mean, self.train_num)
            writer.add_scalar("Q_Net/Q_Loss", loss.item(), self.train_num)
            writer.add_scalar("Grad_norm/Q1_grad_norm", grad_norm1.item(), self.train_num)
            writer.add_scalar("Grad_norm/Q2_grad_norm", grad_norm2.item(), self.train_num)
            writer.add_scalar("Entropy", entropy_mean.item(), self.train_num)
            writer.add_scalar("alpha", self.alpha.item(), self.train_num)
            writer.add_scalar("alpha_loss", alpha_loss.item(), self.train_num)

    def save(self, path1, path2):
        """保存模型权重"""
        torch.save(self.Q_net1.state_dict(), path1)
        torch.save(self.Q_net2.state_dict(), path2)

    def load(self, path1, path2):
        """加载模型权重"""
        # weights_only=True 表示：只加载张量权重（tensor），不反序列化、不还原 Python 对象、类、函数，阻止恶意文件执行任意代码
        # 默认：weights_only=False
        self.Q_net1.load_state_dict(torch.load(path1, map_location=self.dvc, weights_only=True))
        self.Q_net2.load_state_dict(torch.load(path2, map_location=self.dvc, weights_only=True))
