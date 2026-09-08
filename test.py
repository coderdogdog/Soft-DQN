# -*- coding: utf-8 -*-
import gymnasium as gym
from pathlib import Path
from agent import Soft_DQN_Agent
from utils import evaluate_agent, set_seed


def test_agent(args):
    # 创建环境
    env_name = args.env_name
    if args.is_human_render:
        test_env = gym.make(env_name, render_mode="human")
    else:
        test_env = gym.make(env_name, render_mode=None)

    # 测试环境随机种子
    set_seed(args.test_seed)
    test_env.reset(seed=args.test_seed)

    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n.item()

    # 创建智能体
    agent = Soft_DQN_Agent(state_dim, action_dim, args.hidden_dim,
                           alpha=args.alpha,
                           gamma=args.gamma,
                           lr=args.lr,
                           update_tau=args.update_tau,
                           epsilon_start=args.epsilon_start,
                           clip_norm=args.clip_norm,
                           device=args.device)

    model_path1 = f"./model/{env_name}/" + args.load_name1
    model_path2 = f"./model/{env_name}/" + args.load_name2

    Path(model_path1).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path2).parent.mkdir(parents=True, exist_ok=True)

    agent.load(model_path1, model_path2)

    avg_scores, avg_steps = evaluate_agent(test_env, agent, args.is_human_render, args.test_num, print_infos=True)
    test_env.close()

    return avg_scores, avg_steps
