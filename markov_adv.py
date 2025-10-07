"""
markov_adv.py
Advanced Markov chain text generator with:
- add-k (Laplace) smoothing for probability estimation
- backoff to lower-order models when no transitions available
- pruning of low-count transitions
- perplexity calculation over evaluation text
- word- and char-level tokenization
- save/load model
"""

from collections import defaultdict, deque, Counter
import random, math, json
from typing import List, Tuple, Dict, Optional, Iterable

class AdvancedMarkov:
    def __init__(self, order: int = 2, mode: str = "word"):
        assert order >= 1
        assert mode in ("word", "char")
        self.order = order
        self.mode = mode
        # transitions: order -> mapping state(tuple) -> Counter(next_token -> count)
        self.transitions: Dict[int, Dict[Tuple[str,...], Counter]] = {k: defaultdict(Counter) for k in range(1, order+1)}
        self._trained = False

    def _tokenize(self, text: str) -> List[str]:
        if self.mode == "word":
            return text.split()
        else:
            return list(text)

    def train(self, text: str):
        tokens = self._tokenize(text)
        n = len(tokens)
        if n < 2:
            return
        for k in range(1, self.order+1):
            if n <= k: 
                continue
            window = deque(maxlen=k)
            # initialize window with first k tokens
            for i in range(k):
                window.append(tokens[i])
            for i in range(k, n):
                state = tuple(window)
                nxt = tokens[i]
                self.transitions[k][state][nxt] += 1
                window.append(nxt)
        self._trained = True

    def train_files(self, filepaths: Iterable[str], encoding: str = "utf-8"):
        for fp in filepaths:
            with open(fp, "r", encoding=encoding) as f:
                self.train(f.read())

    def prune(self, min_count: int = 2):
        """Remove transitions with counts < min_count"""
        for k in range(1, self.order+1):
            to_delete = []
            for state, counter in list(self.transitions[k].items()):
                for tok, cnt in list(counter.items()):
                    if cnt < min_count:
                        del counter[tok]
                if not counter:
                    del self.transitions[k][state]

    def _get_counter(self, state: Tuple[str,...]) -> Optional[Counter]:
        k = len(state)
        if k == 0: 
            return None
        return self.transitions.get(k, {}).get(state)

    def _choose_next_with_backoff(self, state: Tuple[str,...], add_k: float = 0.0) -> Optional[str]:
        """
        Try to choose next token using state; if no data, backoff to lower-order states.
        add_k: Laplace smoothing constant. If >0, applies smoothing across observed vocabulary for that order.
        """
        k = len(state)
        for cur_k in range(k, 0, -1):
            cur_state = state[-cur_k:] if cur_k <= k else state
            counter = self.transitions.get(cur_k, {}).get(tuple(cur_state))
            if counter and sum(counter.values()) > 0:
                # perform weighted random with optional smoothing
                choices = list(counter.keys())
                counts = [counter[c] for c in choices]
                if add_k > 0.0:
                    # add-k smoothing: add k to each count, and extend choices by unseen tokens (approx by counting vocab)
                    vocab = set()
                    for ctr in self.transitions[cur_k].values():
                        vocab.update(ctr.keys())
                    V = max(1, len(vocab))
                    counts = [c + add_k for c in counts]
                total = sum(counts)
                r = random.random() * total
                upto = 0.0
                for tok, cnt in zip(choices, counts):
                    upto += cnt
                    if r <= upto:
                        return tok
        return None

    def generate(self, length: int = 100, seed: Optional[int] = None, start_state: Optional[Tuple[str,...]] = None, add_k: float = 0.0) -> str:
        if not self._trained:
            raise RuntimeError("Model not trained.")
        if seed is not None:
            random.seed(seed)
        # pick random start state of highest order
        if start_state is None:
            states = list(self.transitions[self.order].keys())
            if not states:
                # fall back to any available state
                for k in range(self.order-1, 0, -1):
                    if self.transitions[k]:
                        states = list(self.transitions[k].keys()); break
            start_state = tuple(random.choice(states))
        state = tuple(start_state)
        out = list(state)
        for _ in range(length):
            nxt = self._choose_next_with_backoff(state, add_k=add_k)
            if nxt is None:
                break
            out.append(nxt)
            # shift state (append next token)
            if self.mode == "word":
                if len(state) >= self.order:
                    state = tuple((list(state) + [nxt])[-self.order:])
                else:
                    state = tuple(list(state) + [nxt])
            else:
                if len(state) >= self.order:
                    state = tuple((list(state) + [nxt])[-self.order:])
                else:
                    state = tuple(list(state) + [nxt])
        if self.mode == "word":
            return " ".join(out)
        else:
            return "".join(out)

    def _probability_of_sequence(self, tokens: List[str], add_k: float = 0.0) -> float:
        """Compute log-probability of tokens under model using backoff and add-k smoothing when requested.
           Returns log probability (natural log)."""
        if len(tokens) <= 1:
            return 0.0
        logp = 0.0
        for i in range(1, len(tokens)):
            # context length up to order
            start = max(0, i - self.order)
            found = False
            for k in range(self.order, 0, -1):
                if i - k < 0:
                    continue
                state = tuple(tokens[i-k:i])
                counter = self.transitions.get(k, {}).get(state)
                if counter and sum(counter.values()) > 0:
                    numerator = counter.get(tokens[i], 0) + add_k
                    # compute denominator: sum counts + V*add_k
                    vocab = set()
                    for ctr in self.transitions[k].values():
                        vocab.update(ctr.keys())
                    V = max(1, len(vocab))
                    denominator = sum(counter.values()) + add_k * V
                    prob = numerator / denominator if denominator > 0 else 1e-12
                    logp += math.log(prob)
                    found = True
                    break
            if not found:
                # assign tiny probability if unseen
                logp += math.log(1e-12)
        return logp

    def perplexity(self, text: str, add_k: float = 0.0) -> float:
        tokens = self._tokenize(text)
        if not tokens:
            return float('inf')
        logp = self._probability_of_sequence(tokens, add_k=add_k)
        N = len(tokens) - 1 if len(tokens) > 1 else 1
        ppl = math.exp(-logp / N)
        return ppl

    def save(self, path: str):
        # Serialize transitions to JSON-friendly dict
        serial = {
            "order": self.order,
            "mode": self.mode,
            "transitions": {
                str(k): { "|".join(state): dict(counter) for state, counter in self.transitions[k].items() }
                for k in self.transitions
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "AdvancedMarkov":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mc = cls(order=int(data["order"]), mode=data["mode"])
        for k_str, mapping in data["transitions"].items():
            k = int(k_str)
            for state_k, counter in mapping.items():
                state = tuple(state_k.split("|")) if state_k else tuple()
                mc.transitions[k][state] = Counter(counter)
        mc._trained = True
        return mc
