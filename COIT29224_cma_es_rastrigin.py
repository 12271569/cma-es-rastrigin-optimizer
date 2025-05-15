
"""
Applies CMA-ES to minimize the Rastrigin function.
"""

import numpy as np

# Rastrigin function definition
def rastrigin(x):
    """
    This is the Rastrigin function.
    We are trying to minimize it using CMA-ES.
    """
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


class CMA_ES:
    """
    This class implements the CMA-ES algorithm.
    It adapts mean and covariance matrix to find the minimum.
    """

    def __init__(self, func, dim=10, sigma=1.0, population_size=40, max_gen=200):
        """
        Constructor to initialize the algorithm parameters.
        """
        self.func = func
        self.dim = dim
        self.sigma = sigma
        self.lambda_ = population_size
        self.max_gen = max_gen
        self.mean = np.random.uniform(-5.12, 5.12, dim)  # corrected range
        self.cov = np.identity(dim)  # identity matrix
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def optimize(self):
        """
        This function runs the optimization loop.
        It updates the mean and covariance to move towards global minimum.
        """
        for gen in range(self.max_gen):
            # Step 1: Sample lambda solutions
            population = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov, self.lambda_)

            # Step 2: Evaluate fitness
            fitness = np.array([self.func(ind) for ind in population])

            # Step 3: Select best individuals
            idx_sorted = np.argsort(fitness)
            selected = population[idx_sorted[:self.lambda_ // 2]]

            # Step 4: Update mean and covariance
            self.mean = np.mean(selected, axis=0)
            self.cov = np.cov(selected.T)

            # Step 5: Track best solution
            if fitness[idx_sorted[0]] < self.best_fitness:
                self.best_fitness = fitness[idx_sorted[0]]
                self.best_solution = population[idx_sorted[0]]

            self.history.append(self.best_fitness)

            # Print for every 10 generations
            if gen % 10 == 0 or gen == self.max_gen - 1:
                print(f"Generation {gen}: Best Fitness = {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness

    def plot_results(self):
        """This function plots convergence graph after optimization"""
        import matplotlib.pyplot as plt

        # Main convergence graph
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, label='Best Fitness per Generation', color='blue')
        plt.title('Convergence of CMA-ES on Rastrigin Function')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.legend()
        plt.savefig("convergence_graph.png")
        plt.close()

        # Zoomed-in for Gen 0–30
        plt.figure(figsize=(10, 6))
        plt.plot(self.history[:30], label='Initial Convergence (Gen 0–30)', color='green')
        plt.title('Early Stage Convergence of CMA-ES')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.legend()
        plt.savefig("early_convergence_graph.png")
        plt.close()

        # Scatter plot of fitness points
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.history)), self.history, c='red', s=10, label='Fitness Points')
        plt.plot(self.history, color='gray', linestyle='--')
        plt.title('Fitness Trajectory (Point-wise) of CMA-ES')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.legend()
        plt.savefig("fitness_trajectory_scatter.png")
        plt.close()

        print("Graphs saved: convergence_graph.png, early_convergence_graph.png, fitness_trajectory_scatter.png")

# Instruction to run the code
if __name__ == "__main__":
    print("Starting CMA-ES optimization on Rastrigin function...")
    optimizer = CMA_ES(func=rastrigin, dim=10, sigma=1.0, population_size=40, max_gen=200)
    best_sol, best_fit = optimizer.optimize()
    print("\nFinal Best Solution:", best_sol)
    print("Final Best Fitness Value:", best_fit)
    optimizer.plot_results()
