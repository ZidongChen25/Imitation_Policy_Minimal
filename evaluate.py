# evaluate.py

import gymnasium as gym
import pickle
import numpy as np
import time

def evaluate_state_aligned(expert_pkl_path):
    # 1. 加载专家数据
    with open(expert_pkl_path, "rb") as f:
        episodes = pickle.load(f)  # list of dicts

    # 2. 创建环境（human 模式可选，这里关闭渲染）
    env = gym.make("Walker2d-v5", render_mode=None)
    unwrapped = env.unwrapped

    nq = unwrapped.model.nq       # 9
    nv = unwrapped.model.nv       # 9
    dt = unwrapped.frame_skip * unwrapped.model.opt.timestep

    all_returns = []

    for ep in episodes:
        # —— 一定要在 env 上 reset，让 OrderEnforcer 知道你重置过了 —— 
        obs, _ = env.reset()

        # 随后再读底层初始状态，用它的 qpos[0] 作为 root 基准
        base_qpos = unwrapped.data.qpos.copy()
        prev_x    = base_qpos[0]

        ep_ret = 0.0
        # 3. 对每一步：先同步状态，再用那个状态执行动作
        for ob, a in zip(ep["observations"], ep["actions"]):
            # 拼回完整的 qpos/qvel
            qpos = np.zeros(nq, dtype=np.float64)
            qpos[0]      = prev_x
            qpos[1:]     = ob[: (nq-1)]
            qvel         = ob[(nq-1):]

            # 同步到底层模拟器
            unwrapped.set_state(qpos, qvel)

            # 用同步后的状态执行 expert 动作，得到 v5 的 reward
            _, r, term, trunc, _ = env.step(a)
            ep_ret += r
            # 不在这里 break，想跑完整示范就算 done 也继续
            # if term or trunc:
            #     break

            # 累计 root 位置
            prev_x += qvel[0] * dt

        all_returns.append(ep_ret)

    env.close()

    avg_ret = np.mean(all_returns)
    print(f"Evaluated {len(all_returns)} episodes (state‑aligned)")
    print(f"Average return per episode: {avg_ret:.2f}")

if __name__ == "__main__":
    expert_path = r"C:\Users\Administrator\Desktop\Imitation_Policy_Minimal\walker2d-expert-v2.pkl"
    evaluate_state_aligned(expert_path)

