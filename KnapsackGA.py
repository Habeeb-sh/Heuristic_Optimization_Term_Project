import random
from typing import List, Tuple, Dict

class KnapsackGA:
    def __init__(self, population_size=100, mutation_rate=0.01, 
                 crossover_rate=0.8, max_generations=500,
                 alpha: float = None):
        """
        Optimized Genetic Algorithm for Knapsack Problem
        
        Args:
            population_size: Number of chromosomes in each generation
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
            max_generations: Maximum number of generations to evolve
            alpha: Penalty coefficient for constraint violation
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.alpha = alpha  # if None, we'll set it after loading data
        
        # Problem data (will be set when loading input)
        self.n_items = 0
        self.max_weight = 0
        self.values = []
        self.weights = []
        self.value_weight_ratio = []
        
    def load_input(self, filename: str) -> None:
        """
        Step 1: Read the input file
        
        File format:
        - First line: n_items max_weight
        - Next n lines: value weight
        """
        try:
            with open(filename, 'r') as f:
                # Read first line: number of items and max weight
                first_line = f.readline().strip().split()
                self.n_items = int(first_line[0])
                self.max_weight = int(first_line[1])
                
                # Read items (value, weight pairs)
                self.values = []
                self.weights = []
                
                for _ in range(self.n_items):
                    line = f.readline().strip().split()
                    value = int(line[0])
                    weight = int(line[1])
                    self.values.append(value)
                    self.weights.append(weight)
                
                # Precompute value-to-weight ratios
                self.value_weight_ratio = [v/w for v, w in zip(self.values, self.weights)]
                
                # Set penalty coefficient if not provided
                if self.alpha is None:
                    max_value = max(self.values)
                    self.alpha = max_value / self.max_weight * 10  # Strong penalty
                    print(f"Setting penalty α = {self.alpha:.2f}")
                    
                print(f"Loaded problem: {self.n_items} items, max weight: {self.max_weight}")
                print(f"Value range: {min(self.values)} - {max(self.values)}")
                print(f"Weight range: {min(self.weights)} - {max(self.weights)}")
                
        except FileNotFoundError:
            print(f"File {filename} not found!")
            # Create sample data for testing
            self.create_sample_data()
            
    def create_sample_data(self):
        """Create sample data if file is not found"""
        print("Creating sample data for testing...")
        self.n_items = 4
        self.max_weight = 5
        self.values = [10, 40, 30, 50]
        self.weights = [5, 4, 6, 3]
        self.value_weight_ratio = [v/w for v, w in zip(self.values, self.weights)]
        if self.alpha is None:
            self.alpha = 20.0  # Strong penalty for sample data
        print(f"Sample problem: {self.n_items} items, max weight: {self.max_weight}")
        print(f"Items: {list(zip(self.values, self.weights))}")
    
    def create_smart_chromosome(self) -> List[int]:
        """
        Create a chromosome using greedy heuristic as starting point
        """
        # Sort items by value-to-weight ratio descending
        sorted_indices = sorted(range(self.n_items), 
                               key=lambda i: self.value_weight_ratio[i], 
                               reverse=True)
        
        chromosome = [0] * self.n_items
        current_weight = 0
        
        # Greedily select items with best ratios that fit
        for item_idx in sorted_indices:
            if current_weight + self.weights[item_idx] <= self.max_weight:
                chromosome[item_idx] = 1
                current_weight += self.weights[item_idx]
        
        # Add some randomness - randomly flip some bits
        for i in range(self.n_items):
            if random.random() < 0.1:  # 10% chance to flip each bit
                chromosome[i] = 1 - chromosome[i]
        
        return chromosome
    
    def create_chromosome(self) -> List[int]:
        """
        Create a chromosome - mix of random and smart initialization
        """
        if random.random() < 0.3:  # 30% smart initialization
            return self.create_smart_chromosome()
        else:  # 70% random initialization but with bias toward lighter items
            chromosome = []
            for i in range(self.n_items):
                # Bias toward selecting lighter items (higher probability for lower weight)
                weight_factor = 1 - (self.weights[i] / max(self.weights))
                prob = 0.3 + 0.4 * weight_factor  # Probability between 0.3 and 0.7
                chromosome.append(1 if random.random() < prob else 0)
            return chromosome
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness with penalty for constraint violation
        Optimized version using list comprehensions and zip
        """
        total_value = sum(v * bit for v, bit in zip(self.values, chromosome))
        total_weight = sum(w * bit for w, bit in zip(self.weights, chromosome))
        
        if total_weight <= self.max_weight:
            return float(total_value)
        else:
            excess = total_weight - self.max_weight
            penalty = self.alpha * excess
            return float(total_value) - penalty
    
    def repair_chromosome(self, chromosome: List[int]) -> List[int]:
        """
        Optimized repair of an infeasible chromosome
        """
        total_weight = sum(w * bit for w, bit in zip(self.weights, chromosome))
        if total_weight <= self.max_weight:
            return chromosome
        
        repaired = chromosome.copy()
        selected = [i for i in range(self.n_items) if repaired[i] == 1]
        
        # Sort selected items by value-to-weight ratio (worst first)
        selected.sort(key=lambda i: self.value_weight_ratio[i])
        
        # Remove items until feasible
        for item_idx in selected:
            repaired[item_idx] = 0
            total_weight -= self.weights[item_idx]
            if total_weight <= self.max_weight:
                break
        
        return repaired
    
    def initialize_population(self) -> List[List[int]]:
        """Generate initial population with mix of strategies"""
        population = []
        for i in range(self.population_size):
            chromosome = self.create_chromosome()
            if i < self.population_size // 4:  # Repair 25% of initial population
                chromosome = self.repair_chromosome(chromosome)
            population.append(chromosome)
        return population
    
    def selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[int]:
        """
        Tournament selection with optimized implementation
        """
        tournament_size = min(5, len(population))
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Optimized two-point crossover
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point1 = random.randint(1, self.n_items - 2)
        point2 = random.randint(point1 + 1, self.n_items - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Optimized mutation implementation
        """
        mutated = chromosome.copy()
        
        # Bit-flip mutation
        for i in range(self.n_items):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Occasionally do swap mutation
        if random.random() < 0.1:
            i, j = random.sample(range(self.n_items), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def evolve_generation(self, population: List[List[int]]) -> List[List[int]]:
        """
        Optimized generation evolution
        """
        # Calculate fitness for all chromosomes
        fitness_scores = [self.calculate_fitness(chrom) for chrom in population]
        
        # Create new population
        new_population = []
        
        # Elitism - keep best chromosomes
        elite_count = max(1, self.population_size // 10)  # Keep top 10%
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.selection(population, fitness_scores)
            parent2 = self.selection(population, fitness_scores)
            
           # Select parents
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Occasionally repair children
            if random.random() < 0.2:  # 20% chance to repair
                child1 = self.repair_chromosome(child1)
            if random.random() < 0.2:
                child2 = self.repair_chromosome(child2)
            
            # Add to new population
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def solve(self) -> Tuple[List[int], int, Dict]:
        """
        Run the Genetic Algorithm with optimized implementation
        """
        print(f"\nStarting Genetic Algorithm...")
        print(f"Population size: {self.population_size}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"Max generations: {self.max_generations}")
        print(f"Penalty coefficient α: {self.alpha:.2f}")
        
        # Initialize population
        population = self.initialize_population()
        
        best_fitness_history = []
        avg_fitness_history = []
        feasible_count_history = []
        
        best_overall_fitness = float('-inf')
        best_overall_solution = None
        generations_without_improvement = 0
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Calculate fitness for current population
            fitness_scores = [self.calculate_fitness(chrom) for chrom in population]
            
            # Count feasible solutions
            feasible_count = sum(
                1 for chrom in population 
                if sum(w * bit for w, bit in zip(self.weights, chrom)) <= self.max_weight
            )
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            feasible_count_history.append(feasible_count)
            
            # Update best overall solution
            if best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_idx = fitness_scores.index(best_fitness)
                best_overall_solution = population[best_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Print progress
            if generation % 50 == 0 or generation == self.max_generations - 1:
                print(f"Generation {generation:3d}: Best={best_fitness:8.1f}, "
                      f"Avg={avg_fitness:8.1f}, Feasible={feasible_count}/{self.population_size}")
            
            # Early stopping
            if generations_without_improvement > 100:
                print(f"Early stopping at generation {generation} (no improvement for {generations_without_improvement} generations)")
                break
            
            # Evolve to next generation
            population = self.evolve_generation(population)
        
        # Ensure we return the best feasible solution found
        best_solution = best_overall_solution
        best_weight = sum(w * bit for w, bit in zip(self.weights, best_solution))
        
        # If best solution is infeasible, try to repair it
        if best_weight > self.max_weight:
            best_solution = self.repair_chromosome(best_solution)
        
        # Calculate final value (actual value, not fitness)
        best_value = sum(v * bit for v, bit in zip(self.values, best_solution))
        
        # Statistics
        stats = {
            'generations_run': generation + 1,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'feasible_count_history': feasible_count_history
        }
        
        return best_solution, best_value, stats
    
    def print_solution(self, solution: List[int], total_value: int) -> None:
        """
        Print the solution with detailed analysis
        (Identical to original version)
        """
        print(f"\nBEST SOLUTION FOUND:")
        print(f"Total Value: {total_value}")
        print(f"Solution: {' '.join(map(str, solution))}")
        
        total_weight = sum(solution[i] * self.weights[i] for i in range(self.n_items))
        selected_items = [i for i in range(self.n_items) if solution[i] == 1]
        
        print(f"\nDetails:")
        print(f"Total Weight: {total_weight}/{self.max_weight}")
        print(f"Selected Items: {selected_items}")
        print(f"Weight utilization: {total_weight/self.max_weight*100:.1f}%")
        print(f"Constraint satisfied: {'Yes' if total_weight <= self.max_weight else 'No'}")
        
        if selected_items:
            print(f"\nSelected Items Details:")
            for i in selected_items:
                ratio = self.values[i] / self.weights[i]
                print(f"  Item {i}: value={self.values[i]}, weight={self.weights[i]}, ratio={ratio:.3f}")

def write_results_to_file(filename: str, solution: List[int], total_value: int, stats: Dict, ga):
    """
    Append results to a file named filename + '.txt'.
    Creates file if it doesn't exist.
    """
    result_filename = f"{filename}Results.txt"
    try:
        with open(result_filename, 'a') as f:
            f.write("=" * 50 + "\n")
            f.write(f"Run Results for {filename}\n")
            f.write(f"Total Value: {total_value}\n")

            total_weight = sum(solution[i] * ga.weights[i] for i in range(ga.n_items))
            f.write(f"Total Weight: {total_weight}/{ga.max_weight}\n")
            f.write(f"Solution: {' '.join(map(str, solution))}\n")

            selected_items = [i for i in range(ga.n_items) if solution[i] == 1]
            f.write(f"Selected Items: {selected_items}\n")
            f.write(f"Weight utilization: {total_weight / ga.max_weight * 100:.1f}%\n")
            f.write(f"Constraint satisfied: {'✅ Yes' if total_weight <= ga.max_weight else '❌ No'}\n\n")

            if selected_items:
                f.write("Selected Items Details:\n")
                for i in selected_items:
                    ratio = ga.values[i] / ga.weights[i]
                    f.write(f"  Item {i}: value={ga.values[i]}, weight={ga.weights[i]}, ratio={ratio:.3f}\n")

            f.write(f"\nGenerations Run: {stats['generations_run']}\n")
            f.write(f"Best Fitness History (last 5): {stats['best_fitness_history'][-5:]}\n")
            f.write(f"Average Fitness History (last 5): {stats['avg_fitness_history'][-5:]}\n")
            f.write(f"Feasible Count History (last 5): {stats['feasible_count_history'][-5:]}\n")
            f.write("=" * 50 + "\n\n")

        print(f"✅ Results appended to {result_filename}")
    except Exception as e:
        print(f"❌ Error writing to {result_filename}: {e}")


    

def main():
    """Main function to run the genetic algorithm"""
    print("KNAPSACK PROBLEM - GENETIC ALGORITHM SOLVER")
    print("=" * 50)
    
    # Initialize GA with optimized parameters
    ga = KnapsackGA(
        population_size=200,    # Balanced population size
        mutation_rate=0.02,     # Moderate mutation rate
        crossover_rate=0.8,
        max_generations=1000    # Sufficient generations
    )
    
    # Load problem
    filename = "ks_40_0"  # Example file; change as needed
    ga.load_input(filename)
    
    # Solve the problem
    best_solution, best_value, stats = ga.solve()
    
    # Print results
    ga.print_solution(best_solution, best_value)
    
    print(f"\nAlgorithm ran for {stats['generations_run']} generations")
    print("Done!")
    
    # Save results
    write_results_to_file(filename, best_solution, best_value, stats, ga)


if __name__ == "__main__":
    main()