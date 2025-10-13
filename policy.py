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
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

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
        # Incrementamos el contador de pasos
        self.t += 1
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

        # Con probabilidad ε exploramos
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)
        else:
            # Con probabilidad 1-ε elegimos el índice máximo más bajo
            return int(np.argmax(self.values))

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
                exploration[arm] = np.sqrt(np.log(self.t) / self.counts[arm])
            else:
                exploration[arm] = np.inf
        return exploration

    def choose(self):
        # Si hay algún brazo sin probar, pruébalo primero
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculamos UCB para los demás
        ucb_values = self.values + self.c * self.exploration_terms
        return int(np.argmax(ucb_values))

    def tell_reward(self, arm, reward):
        # Primero actualizamos, luego incrementamos t
        super().tell_reward(arm, reward)
        self.t += 1
