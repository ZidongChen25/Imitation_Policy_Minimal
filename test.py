import pickle
import numpy as np

file_path = r"C:\Users\Administrator\Desktop\Imitation_Policy_Minimal\walker2d-expert-v2.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 1. 看看 data 是什么类型，有多少条轨迹
print("Type of data:", type(data))        # list
print("Number of episodes:", len(data))   # e.g. 25 条轨迹

# 2. 看看第一条轨迹里有什么字段
first = data[0]
print("Fields in each trajectory:", first.keys())
# 通常会看到： dict_keys(['observations', 'actions', 'rewards', 'terminals', 'timeouts'])

# 3. 每条轨迹的形状举例
print("Episode 0 lengths:")
print("  observations:",  np.array(first['observations']).shape)
print("  actions:",       np.array(first['actions']).shape)

# 4. 如果你想把所有轨迹拼接为单个大数组，方便做行为克隆：
all_obs = np.concatenate([np.array(ep['observations']) for ep in data], axis=0)
all_acts = np.concatenate([np.array(ep['actions'])      for ep in data], axis=0)
print("All episodes combined shapes:")
print("  all_obs:", all_obs.shape)   # (总步数, obs_dim)
print("  all_acts:", all_acts.shape) # (总步数, act_dim)

