import gymnasium as gym
import pickle
import numpy as np
import time

# 1. 本地 expert data 路径
file_path = r"C:\Users\Administrator\Desktop\Imitation_Policy_Minimal\walker2d-expert-v2.pkl"
with open(file_path, "rb") as f:
    episodes = pickle.load(f)  # list of dicts

# 2. 创建 human 渲染环境
env = gym.make("Walker2d-v5", render_mode="human")
unwrapped = env.unwrapped

# 3. MuJoCo 状态维度 & 时间步长
nq = unwrapped.model.nq                     # 9
nv = unwrapped.model.nv                     # 9
dt = unwrapped.frame_skip * unwrapped.model.opt.timestep  # 时间增量

try:
    # 4. 遍历每条轨迹
    for ep_idx, ep in enumerate(episodes):
        print(f"Replaying episode {ep_idx+1}/{len(episodes)}")
        
        # 4.1 reset 得到一个有效的起始状态，并读取完整 qpos
        obs, _     = unwrapped.reset()
        base_qpos  = unwrapped.data.qpos.copy()
        prev_x     = base_qpos[0]   # 根位置 x

        # 4.2 逐帧回放
        for ob in ep["observations"]:
            # ob: 17 维 = [qpos[1:], qvel]
            qpos = np.zeros(nq, dtype=np.float64)
            qpos[0]      = prev_x             # 根 x
            qpos[1:]     = ob[: (nq-1)]       # 其余关节位置
            qvel         = ob[(nq-1):]        # 所有关节速度

            # 同步到模拟器
            unwrapped.set_state(qpos, qvel)
            # 绕过 OrderEnforcer，直接调用底层 render
            unwrapped.render()
            
            # 更新根 x，用上一帧速度累积
            prev_x += qvel[0] * dt

            # 控制回放帧率（可根据需求调整）
            time.sleep(0.02)

finally:
    env.close()



