import torch
import pickle
import gymnasium as gym
from gymnasium import spaces
from envs.CraftingEnv import Recipe, CraftingEnv, ACTION_LIST, STATUSES, ILLEGAL_PENALTY
from DQN import DQN
import numpy as np
from itertools import count

with open("model.pkl", "rb") as f:
    policy_net = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

recipe = Recipe(60, 7920/226, 17000/262, 723, False, "MM")
env = gym.wrappers.FlattenObservation(CraftingEnv(recipe))

c = 0

for t in range(1000):
    if t % 100 == 0: print(f"{c}/{t}")
    env.reset()
    # shape = {
    #     "prog_remaining": spaces.Box(0, 50, dtype=np.float16),
    #     "qual_remaining": spaces.Box(0, 50, dtype=np.float16),
    #     "durability": spaces.Discrete(50+1),
    #     "cp": spaces.Discrete(50+1),
    #     "innovation": spaces.Discrete(6+1),
    #     "great_strides": spaces.Discrete(5+1),
    #     "waste_not": spaces.Discrete(10+1),
    #     "manipulation": spaces.Discrete(10+1),
    #     "heart_and_soul": spaces.Discrete(1+1),
    #     "careful_observation": spaces.Discrete(3+1),
    #     "muscle_memory": spaces.Discrete(5+1),
    #     "veneration": spaces.Discrete(7+1),
    #     "inner_quiet": spaces.Discrete(10+1),
    #     "last_action": spaces.Discrete(len(ACTION_LIST)+1),
    #     "two_actions_ago_basic": spaces.Discrete(1+1),
    #     "status": spaces.Discrete(len(STATUSES))
    # }
    term = False
    rew = 0
    while not term and rew > -ILLEGAL_PENALTY:
        # state = {}
        # for query in shape:
        #     if isinstance(shape[query], spaces.Discrete):
        #         state[query] = int(input(f"Input {query}: "))
        #     else:
        #         state[query] = float(input(f"Input {query}: "))
        #
        # env.reset(options={"initial_state": state})
        #status = int(input("Status? "))
        #env.set_status(status)
        state = torch.tensor(env.observation(env.get_obs()), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state).max(1)[1].view(1, 1)
            #print(action, ACTION_LIST[action])
        #print(env.get_status())
        #success = int(input("Success (1 or -1)? "))
        #env.mod_success(success)
        obs, rew, term, _, _1 = env.step(action)
        #print(env.render())
    if env.score() == 0:
        c += 1
print(f"{c}/{t+1}")