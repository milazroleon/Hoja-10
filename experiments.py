import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from policy import EpsilonGreedyPolicy, UCB
from environment import BanditProblem

def run_experiment(bandit_problem, T=1000, R=100, init_mean_value=0):


    policies = {
        "ε-Greedy (ε=0.1)": lambda: EpsilonGreedyPolicy(0.1, init_mean_value),
        "ε-Greedy (ε=1/t)": lambda: EpsilonGreedyPolicy(lambda t: 1 / t, init_mean_value),
        "UCB1": lambda: UCB(c=2, init_mean_value=init_mean_value)
    }

    results = {name: [] for name in policies}

    for name, make_policy in policies.items():
        print(f"Running policy: {name}")
        for _ in range(R):
            policy = make_policy()
            env = BanditProblem(bandit_problem)
            df = env.simulate_policy(policy, T)
            results[name].append(df)

    metrics = ["avg_reward", "optimal_rate", "cum_regret"]
    avg_results = {metric: {} for metric in metrics}

    for metric in metrics:
        for name, runs in results.items():
            avg_df = pd.concat([df[metric] for df in runs], axis=1).mean(axis=1)
            avg_results[metric][name] = avg_df

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, curve in avg_results[metric].items():
            plt.plot(curve, label=name)
        plt.title(metric.replace("_", " ").capitalize())
        plt.xlabel("Timestep")
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return results

if __name__ == "__main__":

    for x in range(4):
        print(f"\n========== Problem x={x} ==========")
        arm1 = ([1], [1.0])
        arm2 = ([10 * (x + 1), 0], [10 ** (-x), 1 - 10 ** (-x)])
        problem = [arm1, arm2]

        print(f"Running with init_mean_value = 0")
        run_experiment(problem, T=1000, R=50, init_mean_value=0)

        print(f"Running with init_mean_value = {10 * (x + 1)}")
        run_experiment(problem, T=1000, R=50, init_mean_value=10 * (x + 1))

    # 3 brazos
    arms_3 = [([1, 2], [0.5, 0.5]), ([3, 0], [0.2, 0.8]), ([5, 0], [0.1, 0.9])]
    print("\n========== Problem with 3 arms ==========")
    run_experiment(arms_3, T=1000, R=50, init_mean_value=0)

    # 5 brazos
    arms_5 = [([1, 0], [0.7, 0.3]), ([3, 0], [0.4, 0.6]), ([5, 0], [0.2, 0.8]),
              ([8, 0], [0.1, 0.9]), ([10, 0], [0.05, 0.95])]
    print("\n========== Problem with 5 arms ==========")
    run_experiment(arms_5, T=1000, R=50, init_mean_value=0)

    # 10 brazos
    arms_10 = [([i + 1, 0], [0.1, 0.9]) for i in range(10)]
    print("\n========== Problem with 10 arms ==========")
    run_experiment(arms_10, T=1000, R=50, init_mean_value=0)

    # 20 brazos
    arms_20 = [([i + 1, 0], [0.05, 0.95]) for i in range(20)]
    print("\n========== Problem with 20 arms ==========")
    run_experiment(arms_20, T=1000, R=50, init_mean_value=0)
