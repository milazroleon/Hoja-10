import numpy as np

class Policy:
    def __init__(self, init_mean_value=0):
        self.init_mean_value = init_mean_value
        self.num_arms = None
        self.counts = None
        self.values = None
        self.t = 0

    def setup(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.ones(num_arms) * self.init_mean_value
        self.t = 0

    def choose(self):
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float) -> None:
        """Actualiza el promedio incremental y el contador de tiempo."""
        self.counts[arm] += 1
        n = self.counts[arm]
        # promedio incremental
        self.values[arm] += (reward - self.values[arm]) / n
        # el tiempo solo avanza cuando hay feedback
        self.t += 1

    @property
    def mean_estimates(self):
        return self.values


# =====================================================
# ε-GREEDY POLICY
# =====================================================
class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, init_mean_value=0):
        super().__init__(init_mean_value)
        self.epsilon = epsilon

    def setup(self, num_arms):
        super().setup(num_arms)

    def choose(self):
        # epsilon actual
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

        # exploración aleatoria con probabilidad ε
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)

        # explotación determinista: argmax (índice más bajo si hay empate)
        return int(np.argmax(self.values))

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)


# =====================================================
# UCB1 POLICY
# =====================================================
class UCB(Policy):
    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value)
        self.c = c

    def setup(self, num_arms):
        super().setup(num_arms)

    @property
    def exploration_terms(self):
        """Devuelve los términos de exploración de cada brazo."""
        exploration = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self.counts[arm] > 0:
                # usamos log(t+1) para evitar log(0)
                exploration[arm] = np.sqrt(np.log(self.t + 1) / self.counts[arm])
            else:
                exploration[arm] = np.inf
        return exploration

    def choose(self):
        # explorar cada brazo una vez (en orden)
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # calcular valores UCB
        ucb_values = self.values + self.c * self.exploration_terms
        # elegir índice determinista del máximo
        return int(np.argmax(ucb_values))

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
