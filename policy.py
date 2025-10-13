import numpy as np

class Policy:
    def __init__(self, init_mean_value=0):
        self.init_mean_value = init_mean_value

    def setup(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.ones(num_arms) * self.init_mean_value
        self.t = 0  # número de actualizaciones (rondas completadas)

    def choose(self):
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float):
        # promedio incremental
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        # avanzar el tiempo solo cuando recibimos feedback
        self.t += 1

    @property
    def mean_estimates(self):
        return self.values


# --------------------------------------------------------
# ε-greedy policy
# --------------------------------------------------------
class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, init_mean_value=0):
        super().__init__(init_mean_value)
        self.epsilon = epsilon

    def choose(self):
        # epsilon depende del tiempo actual t
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

        # exploración aleatoria
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)

        # explotación determinista: índice más bajo del máximo
        return int(np.argmax(self.values))

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)


# --------------------------------------------------------
# UCB1 policy
# --------------------------------------------------------
class UCB(Policy):
    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value)
        self.c = c

    def choose(self):
        # probar todos los brazos una vez primero
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # calcular término UCB
        exploration = np.sqrt((2 * np.log(self.t)) / self.counts)
        ucb_values = self.values + self.c * exploration
        return int(np.argmax(ucb_values))

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
