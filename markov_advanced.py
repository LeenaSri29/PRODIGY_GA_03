"""
markov_advanced.py
Advanced Markov chain text generator with smoothing, backoff, pruning, perplexity
"""

import random, math
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Iterable

class MarkovChain:
    def __init__(self, order: int = 1, mode: str = "word", smoothing: bool = False):
        assert order >= 1
        assert mode in ("word","char")
        self.order = order
        self.mode = mode
        self.smoothing = smoothing
        self.transitions: Dict[Tuple[str,...], Dict[str,int]] = defaultdict(lambda: defaultdict(int))
        self._trained = False

    def _tokenize(self, text: str) -> List[str]:
        return text.split() if self.mode=="word" else list(text)

    def train(self, text: str):
        tokens = self._tokenize(text)
        if len(tokens) <= self.order:
            return
        window = deque(maxlen=self.order)
        for i in range(self.order):
            window.append(tokens[i])
        for i in range(self.order, len(tokens)):
            state = tuple(window)
            nxt = tokens[i]
            self.transitions[state][nxt] += 1
            window.append(nxt)
        self._trained = True

    def train_files(self, filepaths: Iterable[str], encoding="utf-8"):
        for fp in filepaths:
            with open(fp,"r",encoding=encoding) as f:
                self.train(f.read())

    def _choose_next(self, state: Tuple[str,...]) -> Optional[str]:
        if state not in self.transitions:
            # backoff: shorten context until found
            for i in range(1, len(state)):
                substate = state[i:]
                if substate in self.transitions:
                    state = substate
                    break
            else:
                return None
        choices = self.transitions[state]
        if not choices:
            return None
        tokens = list(choices.keys())
        counts = list(choices.values())
        if self.smoothing:
            counts = [c+1 for c in counts]
        total = sum(counts)
        r = random.uniform(0,total)
        upto = 0
        for token, c in zip(tokens,counts):
            upto += c
            if r <= upto:
                return token
        return tokens[-1]

    def generate(self, length=100, seed=None, start_state=None) -> str:
        if not self._trained:
            raise RuntimeError("Model not trained")
        if seed is not None:
            random.seed(seed)
        if start_state is None:
            start_state = random.choice(list(self.transitions.keys()))
        state = tuple(start_state)
        out = list(state)
        for _ in range(length):
            nxt = self._choose_next(state)
            if not nxt:
                break
            out.append(nxt)
            state = tuple(list(state[1:])+[nxt]) if self.order>1 else (nxt,)
        return " ".join(out) if self.mode=="word" else "".join(out)

    def prune(self, min_count: int = 2):
        """Remove transitions with count < min_count"""
        for state in list(self.transitions.keys()):
            newchoices = {tok:c for tok,c in self.transitions[state].items() if c>=min_count}
            if newchoices:
                self.transitions[state] = newchoices
            else:
                del self.transitions[state]

    def perplexity(self, text: str) -> float:
        tokens = self._tokenize(text)
        if len(tokens)<=self.order:
            return float("inf")
        logprob = 0.0
        N = 0
        window = deque(maxlen=self.order)
        for i in range(self.order):
            window.append(tokens[i])
        for i in range(self.order,len(tokens)):
            state = tuple(window)
            nxt = tokens[i]
            if state not in self.transitions or nxt not in self.transitions[state]:
                prob = 1e-6
            else:
                choices = self.transitions[state]
                total = sum(choices.values())
                count = choices[nxt]
                prob = (count+1)/(total+len(choices)) if self.smoothing else count/total
            logprob += -math.log(prob)
            N += 1
            window.append(nxt)
        return math.exp(logprob/N) if N>0 else float("inf")
