# -*- coding: utf-8 -*-

"""
基于 Pytorch 在 Gymnasium 环境下，使用深度强化学习算法 Soft-DQN 处理单维度离散动作空间问题
使用了 Clipped Double-Q Learning

python 3.11.15
gymnasium 1.3.0
torch 2.13.0.dev20260611+cu132

显卡：5060 pytorch 应该安装最新 Nightly 版本
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

训练：
python main.py --is_test_mode 0 --env_name "LunarLander-v3"

测试指定模型：
python main.py --is_test_mode 1 --env_name "LunarLander-v3" --load_name1 trained_10000_Q1.pth --load_name2 trained_10000_Q2.pth

"""


import argparse
from test import test_agent
from train import train_agent
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Soft-DQN On Gymnasium")

    parser.add_argument("--device", type=str, default="cuda:0", help="训练设备: --device cuda:0")
    parser.add_argument("--train_seed", type=int, default=76, help="训练环境种子: --train_seed 22")
    parser.add_argument("--is_test_mode", type=str2bool, default=False, help="是否设置为测试模式")

    # 测试
    parser.add_argument("--test_seed", type=int, default=99, help="测试环境种子: --test_seed 66")
    parser.add_argument("--test_num", type=int, default=5, help="测试环境次数: --test_num 5")
    parser.add_argument("--is_human_render", type=str2bool, default=1, help="测试时环境渲染 1: human; 0: none")
    parser.add_argument("--load_name1", type=str, default="trained_10000_Q1.pth", help="导入模型文件名")
    parser.add_argument("--load_name2", type=str, default="trained_10000_Q2.pth", help="导入模型文件名")

    # 环境
    parser.add_argument("--env_name", type=str, default="LunarLander-v3",
                        help="LunarLander-v3, CartPole-v1, MountainCar-v0")

    # epsilon
    parser.add_argument("--epsilon", type=float, default=0.02, help="epsilon 极小值 设置安全网，避免坠入局部最优")
    # 训练
    parser.add_argument("--warmup_steps", type=int, default=5000, help="先预热 随机选取动作积累经验 先走几步")
    parser.add_argument("--max_env_steps", type=int, default=1_000_000, help="环境最大运行总步数")
    parser.add_argument("--train_frequency", type=int, default=4, help="每隔几步训练1次")
    parser.add_argument("--eval_interval", type=int, default=1000, help="评估间隔（智能体更新 eval_steps 次后评估模型）")
    parser.add_argument("--save_interval", type=int, default=5000, help="保存模型间隔（智能体更新 save_steps 步后保存模型）")

    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="网络的隐藏层大小，例如：--hidden_dim 256")

    parser.add_argument("--buffer_max_len", type=int, default=int(1e6), help="经验回放池长度")
    parser.add_argument("--batch_size", type=int, default=256, help="训练时batch_size")
    parser.add_argument("--lr", type=float, default=5e-4, help="神经网络学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")

    parser.add_argument("--update_tau", type=float, default=0.01, help="滑动更新")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="网络梯度裁剪阈值")

    return parser.parse_args()


def main():

    args = parse_args()

    if args.is_test_mode:
        print(f"【测试环境】>> {args.env_name}")
        print(f"【测试种子】>> {args.test_seed}")
        print("开始测试...")
        avg_scores, avg_steps = test_agent(args)

    else:
        train_agent(args)

if __name__ == "__main__":
    main()
