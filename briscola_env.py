import random
import numpy as np

SEMI = ["coppe", "denari", "spade", "bastoni"]
VALORI = ["1", "3", "10", "9", "8", "7", "6", "5", "4", "2"]
PUNTI = {"1": 11, "3": 10, "10": 4, "9": 3, "8": 2}

class Carta:
    def __init__(self, seme, valore):
        self.seme = seme
        self.valore = valore

    def punti(self):
        return PUNTI.get(self.valore, 0)

    def __repr__(self):
        return f"{self.valore} di {self.seme}"

def crea_mazzo():
    return [Carta(seme, valore) for seme in SEMI for valore in VALORI]

class BriscolaEnv:
    def __init__(self):
        self.mazzo = []
        self.mano_ai = []
        self.mano_avv = []
        self.briscola = None
        self.carta_avv = None
        self.done = False
        self.punti_ai = 0
        self.punti_avv = 0

    def reset(self):
        self.mazzo = crea_mazzo()
        random.shuffle(self.mazzo)
        self.briscola = self.mazzo.pop()
        self.mano_ai = [self.mazzo.pop() for _ in range(3)]
        self.mano_avv = [self.mazzo.pop() for _ in range(3)]
        self.carta_avv = None
        self.punti_ai = 0
        self.punti_avv = 0
        self.done = False
        return self._get_state()

    def step(self, azione):
        carta_ai = self.mano_ai.pop(azione)
        carta_avv = self.mano_avv.pop(random.randint(0, len(self.mano_avv)-1))
        self.carta_avv = carta_avv
        vincitore = self._chi_vince(carta_ai, carta_avv)
        punti_presa = carta_ai.punti() + carta_avv.punti()
        if vincitore == "ai":
            self.punti_ai += punti_presa
        else:
            self.punti_avv += punti_presa
        self._rimpiazza_carte()
        self.done = len(self.mano_ai) == 0 and len(self.mazzo) == 0
        reward = punti_presa if vincitore == "ai" else 0
        return self._get_state(), reward, self.done, {}

    def _rimpiazza_carte(self):
        if self.mazzo:
            self.mano_ai.append(self.mazzo.pop())
        if self.mazzo:
            self.mano_avv.append(self.mazzo.pop())

    def _chi_vince(self, c1, c2):
        if c1.seme == c2.seme:
            return "ai" if VALORI.index(c1.valore) < VALORI.index(c2.valore) else "avv"
        if c1.seme == self.briscola.seme:
            return "ai"
        if c2.seme == self.briscola.seme:
            return "avv"
        return "ai"

    def _get_state(self):
        stato = [VALORI.index(c.valore) + 10*SEMI.index(c.seme) for c in self.mano_ai]
        stato += [VALORI.index(self.briscola.valore) + 10*SEMI.index(self.briscola.seme)]
        if self.carta_avv:
            stato += [VALORI.index(self.carta_avv.valore) + 10*SEMI.index(self.carta_avv.seme)]
        else:
            stato += [-1]  # nessuna carta sul tavolo
        return np.array(stato, dtype=np.int32)

    def render(self):
        print(f"AI: {self.mano_ai}")
        print(f"Avversario: {len(self.mano_avv)} carte")
        print(f"Briscola: {self.briscola}")
        print(f"Carta avversario: {self.carta_avv}")
        print(f"Punti AI: {self.punti_ai}, Avversario: {self.punti_avv}")
