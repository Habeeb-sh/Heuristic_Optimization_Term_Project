# Knapsack Problem Solver using Genetic Algorithm
A highly optimized Python implementation of a Genetic Algorithm for solving the 0/1 Knapsack Problem. This project includes both a standalone Python class and an interactive Jupyter notebook for experimentation and visualization.
## Table of Contents
•	Overview\n
•	Features
•	Installation
•	Usage
•	Algorithm Details
•	File Structure
•	Input Format
•	Output Format
•	Parameters
•	Examples
## Overview
The 0/1 Knapsack Problem is a classic optimization problem where you must select a subset of items with given weights and values to maximize total value while staying within a weight constraint. This implementation uses a Genetic Algorithm with several optimization techniques to find high-quality solutions efficiently.
##Features
### Core Algorithm Features
•	Hybrid Initialization: Combines random and greedy heuristic-based chromosome creation
•	Penalty-based Fitness: Handles constraint violations with adaptive penalty coefficients
•	Smart Repair Mechanism: Converts infeasible solutions to feasible ones
•	Elitism: Preserves best solutions across generations
•	Multiple Mutation Types: Bit-flip and swap mutations for diversity
•	Tournament Selection: Efficient parent selection mechanism
•	Early Stopping: Prevents unnecessary computation when convergence is detected
### Implementation Features
•	Optimized Performance: List comprehensions and efficient data structures
•	Comprehensive Logging: Detailed progress tracking and statistics
•	Flexible Input/Output: Supports various file formats and result export
•	Sample Data Generation: Built-in test data when input files are unavailable
•	Professional Documentation: Well-documented code with type hints
## Installation
### Prerequisites
•	Python 3.7 or higher
•	NumPy (for Jupyter notebook)
•	Matplotlib (for visualization in notebook)
•	Jupyter Notebook (for interactive version)
### Setup
Clone the repository
`git clone https://github.com/Habeeb-sh/Heuristic_Optimization_Term_Project.git`
`cd HProject`
`pip install numpy matplotlib jupyter`

## Usage
### Standalone Python Script (Option 1 ) 
from knapsack_ga import KnapsackGA

`# Initialize the genetic algorithm
ga = KnapsackGA(
    population_size=200,
    mutation_rate=0.02,
    crossover_rate=0.8,
    max_generations=1000
)
`
`# Load problem from file
ga.load_input("ks_40_0")`

`# Solve the problem
best_solution, best_value, stats = ga.solve()`

`# Display results
ga.print_solution(best_solution, best_value)`
### Jupyter Notebook ( Option 2) 
Launch the interactive notebook for experimentation:
jupyter notebook Knapsack.ipynb
Command Line Usage
`python or python3 Knapsack.ipynb`
## Algorithm Details
### Genetic Algorithm Components
1.	Initialization
•	30% smart initialization using greedy heuristic
•	70% random initialization with weight-based bias
•	Automatic repair of 25% of initial population
2.	Selection
•	Tournament selection with configurable tournament size
•	Elitism preserving top 10% of solutions
3.	Crossover
•	Two-point crossover with 80% probability
•	Maintains genetic diversity while preserving building blocks
4.	Mutation
•	Bit-flip mutation with adaptive rates
•	Occasional swap mutation for structural changes
•	20% probability of repair after mutation
5.	Fitness Function
6.	fitness = value - α × max(0, weight - capacity)
Where α is the adaptive penalty coefficient
### Optimization Techniques
•	Adaptive Penalty: Penalty coefficient scales with problem characteristics
•	Smart Repair: Removes items with worst value-to-weight ratios first
•	Early Stopping: Terminates when no improvement for 100 generations
•	Efficient Data Structures: Optimized for memory and speed
## File Structure
knapsack-genetic-algorithm/
├── Knapsack.py              # Main implementation
├── Knapsack.ipynb  # Interactive Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Sample input files
│   ├── ks_40_0                 # 40-item knapsack problem
│   ├── ks_100_0
  ├── ks_40_0Results.txt                                    # 100-item knapsack problem
│   └── ks_200_0                # 200-item knapsack problem
## Input Format
Input files should follow this format:
```
n_items max_weight
value1 weight1
value2 weight2
...
valueN weightN
```
Example Input File (ks_40_0):
```
4 5
10 5
40 4
30 6
50 3
```

Output Format
Console Output
```
Generation   0: Best=   120.0, Avg=    85.3, Feasible=45/200
Generation  50: Best=   140.0, Avg=   125.7, Feasible=178/200
...

BEST SOLUTION FOUND:
Total Value: 140
Solution: 0 1 0 1
Total Weight: 7/10
Selected Items: [1, 3]
Weight utilization: 70.0%
Constraint satisfied: Yes
```
### Results File
Detailed results are automatically saved to {filename}Results.txt with:
•	Solution vector and selected items
•	Weight utilization and constraint satisfaction
•	Performance statistics and convergence history
•	Detailed item analysis
### Parameters
Parameter	Default	Description
population_size	200	Number of chromosomes per generation
mutation_rate	0.02	Probability of bit mutation
crossover_rate	0.8	Probability of crossover
max_generations	1000	Maximum number of generations
alpha	Auto	Penalty coefficient (auto-calculated if None)
Parameter Tuning Guidelines
•	Small problems (< 50 items): Use smaller population (50-100)
•	Large problems (> 200 items): Increase population (300-500)
•	High constraint satisfaction needed: Increase alpha penalty
•	Exploration vs Exploitation: Adjust mutation rate (0.01-0.05)

### Examples
Basic Usage
```
# Quick solve with default parameters
ga = KnapsackGA()
ga.load_input("ks_40_0")
solution, value, stats = ga.solve()
```
Custom Configuration
```
# High-precision solving
ga = KnapsackGA(
    population_size=500,
    mutation_rate=0.01,
    max_generations=2000,
    alpha=50.0
)
```
Batch Processing
```
# Solve multiple problems
problems = ["ks_40_0", "ks_100_0", "ks_200_0"]
for problem in problems:
    ga = KnapsackGA()
    ga.load_input(problem)
    solution, value, stats = ga.solve()
    print(f"{problem}: Best value = {value}")
```
