
import tkinter as tk
from PIL import Image, ImageTk
import torch
import numpy as np
from briscola_env import BriscolaEnv
from dqn_agent import DQNAgent

# Configurazione
MODEL_PATH = "modello_dqn_finale.pt"
ASSETS_DIR = "assets/"
CARD_WIDTH = 100
CARD_HEIGHT = 150

# Carica modello
state_dim = 45
agent = DQNAgent(state_dim, action_dim=3)
agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
agent.model.eval()

env = BriscolaEnv()
state = env.reset()

# Tkinter setup
root = tk.Tk()
root.title("Briscola vs AI")

canvas = tk.Canvas(root, width=600, height=600)
canvas.pack()

# Carica briscola
briscola_img = None
def load_image(name):
    img = Image.open(f"{ASSETS_DIR}/{name}").resize((CARD_WIDTH, CARD_HEIGHT))
    return ImageTk.PhotoImage(img)

assets = {}
for seme in env.SEMI:
    for valore in env.VALORI:
        key = f"{seme}_{valore}"
        assets[key] = load_image(f"{key}.png")
assets["retro"] = load_image("retro.png")

# Disegna interfaccia
player_cards = []

def render():
    canvas.delete("all")
    # Briscola
    key = f"{env.briscola.seme}_{env.briscola.valore}"
    canvas.create_image(500, 100, image=assets[key])

    # Carte giocatore
    for i, c in enumerate(env.mano_ai):
        key = f"{c.seme}_{c.valore}"
        img = assets[key]
        btn = tk.Button(root, image=img, command=lambda idx=i: play_card(idx))
        btn.image = img
        canvas.create_window(150 + i*120, 450, window=btn)
        player_cards.append(btn)

    # AI o avversario (se giocato)
    if env.carta_avv:
        key = f"{env.carta_avv.seme}_{env.carta_avv.valore}"
        canvas.create_image(300, 200, image=assets[key])
    else:
        canvas.create_image(300, 200, image=assets["retro"])

    canvas.update()

def play_card(idx):
    global state
    action = idx
    next_state, reward, done, _ = env.step(action)
    state = next_state
    render()
    if not done:
        # turno AI
        ai_act = agent.act(state)
        state, r2, done, _ = env.step(ai_act)
        render()
    if done:
        tk.messagebox.showinfo("Fine partita", f"Punti AI: {env.punti_ai}, Tu: {env.punti_avv}")
        root.quit()

# Avvia
render()
root.mainloop()
