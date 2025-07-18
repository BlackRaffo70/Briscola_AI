from briscola_env import BriscolaEnv
from dqn_agent import DQNAgent
import numpy as np
from tqdm import trange
import torch
import os

EPISODI = 10000
STATO_DIM = 45  # 40 viste + 3 carte in mano + briscola + carta avversario (o -1)
AZIONI_DIM = 3  # 3 carte tra cui scegliere
SALVA_EVERY = 1000

env = BriscolaEnv()
agent = DQNAgent(state_dim=STATO_DIM, action_dim=AZIONI_DIM)

reward_history = []

for episodio in trange(EPISODI):
    stato = env.reset()
    done = False
    totale_reward = 0

    while not done:
        azione = agent.act(stato)
        nuovo_stato, reward, done, _ = env.step(azione)
        agent.remember(stato, azione, reward, nuovo_stato, done)
        agent.replay()
        stato = nuovo_stato
        totale_reward += reward

    reward_history.append(totale_reward)

    if episodio % 100 == 0:
        media = np.mean(reward_history[-100:])
        print(f"Episodio {episodio}, media reward ultime 100: {media:.2f}, epsilon: {agent.epsilon:.2f}")

    if episodio % SALVA_EVERY == 0 and episodio > 0:
        torch.save(agent.model.state_dict(), f"modello_dqn_ep{episodio}.pt")

# Salva il modello finale
torch.save(agent.model.state_dict(), "modello_dqn_finale.pt")
print("Training completato e modello salvato.")
