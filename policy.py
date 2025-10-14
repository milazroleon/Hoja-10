import numpy as np

class Policy:
    def __init__(self, init_mean_value=0):
        self.init_mean_value = init_mean_value
        self.num_arms = None
        self.counts = None
        self.values = None
        self.t = 0  # número de recompensas recibidas hasta ahora

    def setup(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms, dtype=int)
        self.values = np.ones(num_arms) * self.init_mean_value
        self.t = 0

    def choose(self):
        raise NotImplementedError

    def tell_reward(self, arm: int, reward: float) -> None:
        # actualizar promedio incremental y contador de recompensas/t
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

    def choose(self):
        # 1) fase inicial: probar cada brazo no probado en orden
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # 2) calcular epsilon (epsilon puede depender de t)
        eps = self.epsilon(self.t) if callable(self.epsilon) else self.epsilon

        # 3) explorar con probabilidad eps
        if np.random.rand() < eps:
            return np.random.randint(0, self.num_arms)

        # 4) explotar: elegir el brazo con mayor estimado
        # desempate: elegir el menor índice (primer máximo)
        max_value = np.max(self.values)
        best_arms = np.where(self.values == max_value)[0]
        return int(best_arms[0])

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)


class UCB(Policy):
    def __init__(self, c, init_mean_value=0):
        super().__init__(init_mean_value)
        self.c = c

    def choose(self):
        # 1) fase inicial: probar cada brazo no probado en orden
        for arm in range(self.num_arms):
            if self.counts[arm] == 0:
                return arm

        # 2) calcular término de exploración (evitar log(0) con t+1)
        #    usamos np.log(self.t + 1) por seguridad si t empieza en 0/1
        exploration = np.sqrt((2 * np.log(self.t + 1)) / self.counts)

        # 3) calcular UCB y elegir mejor (desempate: menor índice)
        ucb_values = self.values + self.c * exploration
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        return int(best_arms[0])

    def tell_reward(self, arm, reward):
        super().tell_reward(arm, reward)
