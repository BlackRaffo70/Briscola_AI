# briscola_ai/briscola_env.py
import random
import numpy as np

SEMI = ["coppe", "denari", "spade", "bastoni"]
VALORI = ["A", "3", "R", "C", "F", "7", "6", "5", "4", "2"]
PUNTI = {"A": 11, "3": 10, "R": 4, "C": 3, "F": 2}

class Carta:
    def __init__(self, seme, valore):
        self.seme = seme
        self.valore = valore

    def punti(self):
        return PUNTI.get(self.valore, 0)

    def __repr__(self):
        return f"{self.valore} di {self.seme}"

def codifica_carta(carta):
    return VALORI.index(carta.valore) + 10 * SEMI.index(carta.seme)

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
        self.carte_viste = np.zeros(40, dtype=np.int32)

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
        self.carte_viste = np.zeros(40, dtype=np.int32)

        for c in self.mano_ai + self.mano_avv + [self.briscola]:
            self.carte_viste[codifica_carta(c)] = 1

        return self._get_state()

    def step(self, azione):
        if len(self.mano_ai) == 0 or len(self.mano_avv) == 0:
            self.done = True
            return self._get_state(), 0, self.done, {}

        carta_ai = self.mano_ai.pop(azione)
        carta_avv = self.mano_avv.pop(random.randint(0, len(self.mano_avv)-1))
        self.carta_avv = carta_avv

        self.carte_viste[codifica_carta(carta_ai)] = 1
        self.carte_viste[codifica_carta(carta_avv)] = 1

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
            nuova_ai = self.mazzo.pop()
            self.mano_ai.append(nuova_ai)
            self.carte_viste[codifica_carta(nuova_ai)] = 1
        if self.mazzo:
            nuova_avv = self.mazzo.pop()
            self.mano_avv.append(nuova_avv)
            self.carte_viste[codifica_carta(nuova_avv)] = 1

    def _chi_vince(self, c1, c2):
        if c1.seme == c2.seme:
            return "ai" if VALORI.index(c1.valore) < VALORI.index(c2.valore) else "avv"
        if c1.seme == self.briscola.seme:
            return "ai"
        if c2.seme == self.briscola.seme:
            return "avv"
        return "ai"

    def _get_state(self):
        stato = []
        for i in range(3):
            if i < len(self.mano_ai):
                stato.append(codifica_carta(self.mano_ai[i]))
            else:
                stato.append(-1)
        stato += [codifica_carta(self.briscola)]
        stato += [-1 if not self.carta_avv else codifica_carta(self.carta_avv)]
        stato += list(self.carte_viste)
        return np.array(stato, dtype=np.int32)

    def render(self):
        print(f"AI: {self.mano_ai}")
        print(f"Avversario: {len(self.mano_avv)} carte")
        print(f"Briscola: {self.briscola}")
        print(f"Carta avversario: {self.carta_avv}")
        print(f"Punti AI: {self.punti_ai}, Avversario: {self.punti_avv}")