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
parser.add_argument("--iso", default='../ssbm.iso', type=str, help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
parser.add_argument("--restore", default=False, type=bool)
parser.add_argument("--num_ep", default=64)
parser.add_argument("--num_rounds", default=4)
args = parser.parse_args()

time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
MODEL_PATH = './models/'
SAVE_PATH = os.path.join(MODEL_PATH, time_str)

obs_space = ObservationSpace()
AGENT = OnlineDecisionTransformerAgent(obs_space)
if args.restore:
    AGENT.load_state_dict(torch.load(os.listdir(MODEL_PATH)[-1]))
players = [AGENT, Random(character=enums.Character.FOX)]

cnt = 0

for episode in range(args.num_ep):
    env = MeleeEnv(args.iso, players, fast_forward=False)
    env.start()
    gamestate, done = env.reset(enums.Stage.BATTLEFIELD)
    while not done: 
        players[0].act(gamestate)
        players[1].act(gamestate)
        gamestate, reward, done, info = env.step()
    players[0].end_ep()
    print(f"# of episode :{episode + 1}")
    env.close()
    cnt += 1

    if cnt % args.num_rounds == 0 and cnt > 0:
        players[0].train()
        save_path = SAVE_PATH + ".pth"
        print('Save model state_dict to', save_path)
        torch.save(players[0].model.state_dict(), save_path)