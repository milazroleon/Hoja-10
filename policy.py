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
        """Actualiza la media incremental y el contador de pasos."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.t += 1

    @property
    def mean_estimates(self):
        return self.values


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon, init_mean_value=0):
        super().__init__(init_mean_value)
        self.epsilon = epsilon

    def setup(self, num_arms):
        super().setup(num_arms)

    def choose(self):
        eps = self.epsilon(self.t + 1) if callable(self.epsilon) else self.epsilon

        # Explorar con probabilidad epsilon
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)

        # Explotar: elegir el índice del valor máximo (rompe empates al más alto índice)
        max_value = np.max(self.values)
        best_arms = np.where(self.values == max_value)[0]
        return int(best_arms[-1])  # usa el último índice en caso de empate

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)


class UCB(Policy):
    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value)
        self.c = c

    def setup(self, num_arms):
        super().setup(num_arms)

    @property
    def exploration_terms(self):
        exploration = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self.counts[arm] > 0:
                exploration[arm] = np.sqrt(np.log(self.t + 1) / self.counts[arm])
            else:
                exploration[arm] = np.inf
        return exploration

    def choose(self):
        # Elegir primero los brazos no probados (en orden)
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + self.c * self.exploration_terms
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        return int(best_arms[-1])  # usa el último índice en caso de empate

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
