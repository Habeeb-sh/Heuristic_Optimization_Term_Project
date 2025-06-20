# Knapsack Problem Solver using Genetic Algorithm

A highly optimized Python implementation of a Genetic Algorithm for solving the **0/1 Knapsack Problem**. This project includes both a standalone Python class and an interactive Jupyter notebook for experimentation and visualization.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Algorithm Details](#algorithm-details)
* [File Structure](#file-structure)
* [Input Format](#input-format)
* [Output Format](#output-format)
* [Parameters](#parameters)
* [Examples](#examples)

---

## Overview

The **0/1 Knapsack Problem** is a classic optimization problem where you must select a subset of items with given weights and values to maximize total value while staying within a weight constraint. This implementation uses a **Genetic Algorithm** with several optimization techniques to find high-quality solutions efficiently.

---

## Features

### Core Algorithm Features

* Hybrid Initialization: Combines random and greedy heuristic-based chromosome creation
* Penalty-based Fitness: Handles constraint violations with adaptive penalty coefficients
* Smart Repair Mechanism: Converts infeasible solutions to feasible ones
* Elitism: Preserves best solutions across generations
* Multiple Mutation Types: Bit-flip and swap mutations for diversity
* Tournament Selection: Efficient parent selection mechanism
* Early Stopping: Prevents unnecessary computation when convergence is detected

### Implementation Features

* Optimized Performance: List comprehensions and efficient data structures
* Comprehensive Logging: Detailed progress tracking and statistics
* Flexible Input/Output: Supports various file formats and result export
* Sample Data Generation: Built-in test data when input files are unavailable
* Professional Documentation: Well-documented code with type hints

---

## Installation

### Prerequisites

* Python 3.7 or higher
* `numpy` (for notebook)
* `matplotlib` (for visualization)
* `jupyter` (for interactive version)

### Setup

```bash
git clone https://github.com/Habeeb-sh/Heuristic_Optimization_Term_Project.git
cd HProject
pip install numpy matplotlib jupyter
```

---

## Usage

### Option 1: Standalone Python Script

```python
from knapsack_ga import KnapsackGA

ga = KnapsackGA(
    population_size=200,
    mutation_rate=0.02,
    crossover_rate=0.8,
    max_generations=1000
)

ga.load_input("ks_40_0")
best_solution, best_value, stats = ga.solve()
ga.print_solution(best_solution, best_value)
```

### Option 2: Jupyter Notebook

Launch the interactive notebook:

```bash
jupyter notebook Knapsack.ipynb
```

### Command Line

```bash
python Knapsack.ipynb
```

---

## Algorithm Details

### Genetic Algorithm Components

1. **Initialization**

   * 30% smart initialization using greedy heuristic
   * 70% random with weight-bias
   * 25% population auto-repaired

2. **Selection**

   * Tournament selection
   * Elitism (top 10% preserved)

3. **Crossover**

   * Two-point crossover (80% probability)

4. **Mutation**

   * Bit-flip with adaptive rates
   * Swap mutation occasionally
   * 20% chance of post-mutation repair

5. **Fitness Function**

```python
fitness = value - α × max(0, weight - capacity)
```

Where `α` is the adaptive penalty coefficient.

### Optimization Techniques

* Adaptive Penalty
* Smart Repair (remove worst value/weight items)
* Early Stopping (after 100 stagnant generations)
* Efficient Data Structures

---

## File Structure

```
knapsack-genetic-algorithm/
├── Knapsack.py              # Main implementation
├── Knapsack.ipynb           # Jupyter notebook
├── requirements.txt         # Dependencies
├── README.md                # Documentation
├── data/
│   ├── ks_40_0              # 40-item problem
│   ├── ks_100_0
│   ├── ks_200_0
│   └── ks_40_0Results.txt   # Output file
```

---

## Input Format

Each input file must follow this format:

```
n_items max_weight
value1 weight1
value2 weight2
...
valueN weightN
```

**Example:**

```
4 5
10 5
40 4
30 6
50 3
```

---

## Output Format

### Console Output

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

### Results File Output (`ks_40_0Results.txt`)

* Solution vector
* Selected items
* Total weight
* Value and weight utilization
* Performance stats and history
* Item analysis summary

---

## Parameters

| Parameter        | Default | Description                            |
| ---------------- | ------- | -------------------------------------- |
| population\_size | 200     | Chromosomes per generation             |
| mutation\_rate   | 0.02    | Bit-flip mutation chance               |
| crossover\_rate  | 0.8     | Crossover operation chance             |
| max\_generations | 1000    | Total evolutionary iterations          |
| alpha            | Auto    | Penalty coefficient (adaptive if None) |

### Tuning Tips

* Small problems (< 50 items): population = 50–100
* Large problems (> 200 items): population = 300–500
* Strong constraint satisfaction: increase `alpha`
* Explore more: increase mutation to 0.03–0.05

---

## Examples

### Quick Solve

```python
ga = KnapsackGA()
ga.load_input("ks_40_0")
solution, value, stats = ga.solve()
```

### Custom Configuration

```python
ga = KnapsackGA(
    population_size=500,
    mutation_rate=0.01,
    max_generations=2000,
    alpha=50.0
)
```

### Batch Processing

```python
problems = ["ks_40_0", "ks_100_0", "ks_200_0"]
for problem in problems:
    ga = KnapsackGA()
    ga.load_input(problem)
    solution, value, stats = ga.solve()
    print(f"{problem}: Best value = {value}")
```
