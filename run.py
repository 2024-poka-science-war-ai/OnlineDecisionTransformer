from melee import enums
from melee_env.agents.util import ObservationSpace
from melee_env.env import MeleeEnv
from melee_env.agents.basic import *
import argparse
from agent import OnlineDecisionTransformerAgent
import torch
from torch.distributions import Categorical
import time
import os

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default='ssbm.iso', type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
parser.add_argument("--restore", default=False, type=bool)
parser.add_argument("--num_ep", default=10)
args = parser.parse_args()

time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
MODEL_PATH = 'melee_PG/model/'
SAVE_PATH = os.path.join(MODEL_PATH, time_str)

obs_space = ObservationSpace()
AGENT = OnlineDecisionTransformerAgent(obs_space)
if args.restore:
    AGENT.load_state_dict(torch.load(os.listdir(MODEL_PATH)[-1]))
players = [AGENT, Rest()]

env = MeleeEnv(args.iso, players, fast_forward=False)
env.start()

for episode in range(args.num_ep):
    R = 0
    score = 0
    gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    while not done: 
        players[0].act(gamestate)
        players[1].act(gamestate)

        gamestate, done = env.step()
    
    players[0].end_ep()
    
    save_path = SAVE_PATH + ".pth"
    print('Save model state_dict to', save_path)
    torch.save(players[0].state_dict(), save_path)

    print("# of episode :{}, score : {}".format(episode + 1, score))

env.close()