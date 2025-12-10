# Quick Start Guide

Get up and running with the Tunix Reasoning Agent in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/ujjawalkaushik1110/tunix-reasoning-agent.git
cd tunix-reasoning-agent

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Your First Problem

Create a file `my_first_agent.py`:

```python
from tunix_reasoning_agent import ReasoningAgent

# Initialize the agent
agent = ReasoningAgent(verbose=True)

# Solve a problem
problem = "What is 15% of 240?"
result = agent.solve(problem)

# Print the solution
print(f"\nSolution: {result['solution']}")
print(f"Verified: {result['metadata']['verified']}")
print(f"Confidence: {result['metadata']['confidence']:.2%}")
```

Run it:

```bash
python my_first_agent.py
```

You'll see the agent work through the four reasoning steps:
1. **Understand**: Analyzing the problem
2. **Plan**: Creating a solution strategy
3. **Execute**: Implementing the solution
4. **Verify**: Validating the results

## Explore Different Problem Types

### Mathematical Problems

```python
agent = ReasoningAgent(verbose=False)

# Speed, distance, time
result1 = agent.solve("A train travels 120 km in 2 hours. What is its average speed?")

# Percentages
result2 = agent.solve("What is 25% of 200?")

# Geometry
result3 = agent.solve("Calculate the area of a circle with radius 7 cm.")
```

### Logical Problems

```python
problem = """
Three people - Alice, Bob, and Charlie - are standing in a line.
Alice is not first. Charlie is not last. Bob is not in the middle.
What is the order of the people?
"""

result = agent.solve(problem)
print(result['solution'])
```

### Analytical Problems

```python
problem = "What are the main differences between Python and JavaScript?"
result = agent.solve(problem)
```

## Problem Decomposition

For complex problems, enable decomposition:

```python
from tunix_reasoning_agent import ReasoningAgent, ProblemDecomposer

agent = ReasoningAgent(verbose=True)

complex_problem = """
Design a complete e-commerce platform with user authentication,
product catalog, shopping cart, payment processing, order tracking,
and admin dashboard. Plan the architecture and implementation steps.
"""

# Solve with decomposition
result = agent.solve(
    complex_problem,
    decompose=True,
    decomposition_strategy="hierarchical"
)

# Visualize the decomposition
decomposer = ProblemDecomposer()
decomposition = result['decomposition']
print(decomposer.visualize_decomposition(decomposition))
```

### Decomposition Strategies

```python
# Sequential: Steps that must be done in order
agent.solve(problem, decompose=True, decomposition_strategy="sequential")

# Hierarchical: Tree structure with parent-child relationships
agent.solve(problem, decompose=True, decomposition_strategy="hierarchical")

# Parallel: Independent aspects that can be solved simultaneously
agent.solve(problem, decompose=True, decomposition_strategy="parallel")
```

## Evaluate Solution Quality

```python
from tunix_reasoning_agent.evaluation import MetricsCalculator

calculator = MetricsCalculator()

# Solve multiple problems
problems = [
    "What is 2 + 2?",
    "Calculate 15% of 300",
    "If x + 5 = 12, what is x?"
]

results = [agent.solve(p) for p in problems]

# Evaluate
batch_eval = calculator.evaluate_batch(results)

# Get summary
print(calculator.get_summary_report())
```

The metrics include:
- **Correctness**: Is the solution accurate?
- **Completeness**: Are all requirements addressed?
- **Coherence**: Is the reasoning logical?
- **Efficiency**: How quickly was it solved?
- **Clarity**: Is the explanation clear?

## Visualize Results

```python
from tunix_reasoning_agent.utils import ReasoningVisualizer

visualizer = ReasoningVisualizer()

# Detailed trace visualization
print(visualizer.visualize_result(result, detailed=True))

# Compare multiple results
print(visualizer.create_comparison_table(results))

# Summary report
print(visualizer.create_summary_report(results))
```

## Batch Processing

Process multiple problems efficiently:

```python
agent = ReasoningAgent(verbose=False)

problems = [
    "Problem 1...",
    "Problem 2...",
    "Problem 3...",
]

# Solve all
results = []
for i, problem in enumerate(problems, 1):
    print(f"Solving problem {i}/{len(problems)}...")
    result = agent.solve(problem)
    results.append(result)

# Get statistics
stats = agent.get_statistics()
print(f"Total problems: {stats['total_problems']}")
print(f"Verification rate: {stats['verification_rate']:.1%}")
print(f"Average duration: {stats['average_duration']:.2f}s")
```

## Access Reasoning Trace

Get detailed reasoning information:

```python
# Solve a problem
result = agent.solve("Your problem here")

# Get trace as dictionary
trace_dict = agent.get_reasoning_trace(format="dict")

# Get trace as JSON
trace_json = agent.get_reasoning_trace(format="json")

# Get trace as formatted text
trace_text = agent.get_reasoning_trace(format="text")
print(trace_text)
```

## Fine-tuning (Advanced)

Prepare data for fine-tuning:

```python
from tunix_reasoning_agent import TunixTrainer
from tunix_reasoning_agent.fine_tuning import DatasetBuilder

# Your training data
problems = ["Problem 1", "Problem 2", "Problem 3"]
solutions = ["Solution 1", "Solution 2", "Solution 3"]

# Initialize trainer
trainer = TunixTrainer(output_dir="./models")

# Prepare data
training_data = trainer.prepare_training_data(
    problems=problems,
    solutions=solutions,
    output_file="training_data.json"
)

# Create train/validation split
builder = DatasetBuilder()
train_data, val_data = builder.create_from_examples(
    training_data,
    train_ratio=0.8
)

# Fine-tune (simulated for now)
results = trainer.fine_tune(
    training_data=train_data,
    validation_data=val_data,
    num_epochs=3
)

print(trainer.get_training_summary())
```

## Next Steps

- 📖 Read the [Architecture Documentation](ARCHITECTURE.md)
- 📚 Check the [API Reference](API.md)
- 💻 Try the [Demo Notebook](../notebooks/demo_reasoning_agent.ipynb)
- 🔍 Explore [Example Scripts](../examples/)
- 🧪 Run the [Test Suite](../tests/)

## Common Patterns

### Pattern 1: Simple Problem Solving

```python
agent = ReasoningAgent(verbose=False)
result = agent.solve("Your problem")
print(result['solution'])
```

### Pattern 2: With Evaluation

```python
from tunix_reasoning_agent import ReasoningAgent
from tunix_reasoning_agent.evaluation import MetricsCalculator

agent = ReasoningAgent(verbose=False)
calculator = MetricsCalculator()

result = agent.solve("Your problem")
evaluation = calculator.evaluate_result(result)

print(f"Overall Score: {evaluation['metrics']['overall']:.3f}")
```

### Pattern 3: With Visualization

```python
from tunix_reasoning_agent import ReasoningAgent
from tunix_reasoning_agent.utils import ReasoningVisualizer

agent = ReasoningAgent(verbose=False)
visualizer = ReasoningVisualizer()

result = agent.solve("Your problem")
print(visualizer.visualize_result(result, detailed=True))
```

### Pattern 4: Complete Workflow

```python
from tunix_reasoning_agent import ReasoningAgent
from tunix_reasoning_agent.evaluation import MetricsCalculator
from tunix_reasoning_agent.utils import ReasoningVisualizer

# Initialize
agent = ReasoningAgent(verbose=True)
calculator = MetricsCalculator()
visualizer = ReasoningVisualizer()

# Solve
result = agent.solve("Your problem", decompose=True)

# Evaluate
evaluation = calculator.evaluate_result(result)

# Visualize
print(visualizer.visualize_result(result, detailed=True))
print(f"\nOverall Score: {evaluation['metrics']['overall']:.3f}")
```

## Tips

1. **Use verbose mode** during development to understand the reasoning process
2. **Enable decomposition** for complex, multi-part problems
3. **Batch process** multiple similar problems for efficiency
4. **Check evaluation metrics** to assess solution quality
5. **Visualize traces** when debugging or explaining results
6. **Store history** to track performance over time

## Troubleshooting

### Import Error

If you get an import error:
```bash
# Make sure you're in the right directory
cd tunix-reasoning-agent

# Install the package
pip install -e .
```

### No Module Named 'tunix_reasoning_agent'

Add the src directory to your Python path:
```python
import sys
sys.path.insert(0, 'path/to/tunix-reasoning-agent/src')
```

### Tests Not Running

Install pytest:
```bash
pip install pytest
pytest tests/ -v
```

## Getting Help

- 📖 Read the [full documentation](../README.md)
- 💬 Open an [issue on GitHub](https://github.com/ujjawalkaushik1110/tunix-reasoning-agent/issues)
- 📧 Contact the maintainer

Happy reasoning! 🚀
