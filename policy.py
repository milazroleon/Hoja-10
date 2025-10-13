import numpy as np

class Policy:
    def __init__(self, init_mean_value=0):
        self.init_mean_value = init_mean_value

    def setup(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.ones(num_arms) * self.init_mean_value
        self.t = 0  # tiempo total

    def choose(self):
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float):
        """Actualiza valores y avanza el tiempo."""
        # primero avanzar el tiempo (para que t cuente desde 1)
        self.t += 1
        # promedio incremental
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

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

    def choose(self):
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

        # con prob ε, explora
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)

        # con prob 1-ε, elige el valor máximo (índice más bajo si empate)
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

    def choose(self):
        # probar cada brazo una vez al principio
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # cálculo clásico de UCB1
        exploration = np.sqrt((2 * np.log(self.t + 1)) / self.counts)
        ucb_values = self.values + self.c * exploration
        return int(np.argmax(ucb_values))

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
