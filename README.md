# 🃏 Briscola_AI – Reinforcement Learning Agent per Briscola

Questo progetto implementa un agente intelligente in grado di **giocare a Briscola 1v1** utilizzando tecniche di **Deep Q-Learning (DQN)**. Include sia un **ambiente simulato** per il training, sia una **interfaccia grafica in `tkinter`** con immagini di carte napoletane per giocare contro l’AI.

---

## 🚀 Funzionalità

- 🌱 Addestramento dell’agente con DQN (PyTorch)
- 🧠 Stato avanzato: tiene conto delle carte giocate (conteggio carte)
- 🕹️ Interfaccia grafica con carte napoletane (usando `tkinter` + `Pillow`)
- 🎮 Modalità interattiva: gioca tu contro l’AI

---

## 📦 Requisiti

Assicurati di avere Python 3.8 o superiore.  
Installa i pacchetti richiesti con:

```bash
pip install torch numpy tqdm pillow
bash```

Nota: su macOS è necessario avere tkinter funzionante (già incluso con Python).
Se hai problemi con Pillow, installa il pacchetto corretto con pip install Pillow.


---

## 📂 Struttura del progetto
Briscola_AI/
├── assets/                     # Immagini delle carte napoletane
│   ├── bastoni_A.png
│   ├── coppe_3.png
│   └── retro.png
├── briscola_env.py            # Ambiente di gioco Briscola 1v1
├── dqn_agent.py               # Agente DQN con replay buffer e rete neurale
├── train.py                   # Addestramento dell'agente AI
├── play_vs_ai.py              # Interfaccia grafica per giocare contro l'AI
├── play_vs_ai_terminal.py     # (Opzionale) versione testuale del gioco
├── modello_dqn_finale.pt      # Modello addestrato (generato dopo training)
└── README.md                  # Documentazione del progetto

