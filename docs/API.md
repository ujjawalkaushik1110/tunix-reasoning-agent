# API Documentation

Complete API reference for the Tunix Reasoning Agent.

## Table of Contents

- [ReasoningAgent](#reasoningagent)
- [ProblemDecomposer](#problemdecomposer)
- [ReasoningMetrics](#reasoningmetrics)
- [TunixTrainer](#tunixtrainer)
- [DatasetBuilder](#datasetbuilder)
- [Utilities](#utilities)

---

## ReasoningAgent

Main class for solving problems using multi-step reasoning.

### Constructor

```python
ReasoningAgent(
    model_name: str = "gemini-pro",
    api_key: Optional[str] = None,
    verbose: bool = True
)
```

**Parameters:**
- `model_name` (str): Name of the LLM model to use
- `api_key` (str, optional): API key for the model
- `verbose` (bool): Whether to print detailed progress

### Methods

#### solve()

```python
solve(
    problem: str,
    decompose: bool = True,
    decomposition_strategy: str = "sequential"
) -> Dict[str, Any]
```

Solve a problem using multi-step reasoning.

**Parameters:**
- `problem` (str): The problem statement to solve
- `decompose` (bool): Whether to decompose complex problems
- `decomposition_strategy` (str): Strategy for decomposition ('sequential', 'hierarchical', 'parallel')

**Returns:**
- Dictionary containing:
  - `problem`: Original problem
  - `solution`: Generated solution
  - `reasoning_trace`: Complete trace of all reasoning steps
  - `decomposition`: Problem decomposition if enabled
  - `metadata`: Metadata including verification status, confidence, duration

**Example:**
```python
agent = ReasoningAgent(verbose=True)
result = agent.solve("What is 15% of 240?")
print(result['solution'])
```

#### get_reasoning_trace()

```python
get_reasoning_trace(format: str = "dict") -> Any
```

Get the reasoning trace from the last solution.

**Parameters:**
- `format` (str): Output format ('dict', 'json', 'text')

**Returns:**
- Reasoning trace in specified format

#### get_statistics()

```python
get_statistics() -> Dict[str, Any]
```

Get statistics about the agent's performance.

**Returns:**
- Dictionary with statistics including:
  - `total_problems`: Total problems solved
  - `verified_solutions`: Number of verified solutions
  - `verification_rate`: Percentage of verified solutions
  - `average_duration`: Average solving time
  - `average_confidence`: Average confidence score

#### clear_history()

```python
clear_history() -> None
```

Clear the reasoning history.

---

## ProblemDecomposer

Decomposes complex problems into manageable sub-problems.

### Constructor

```python
ProblemDecomposer()
```

### Methods

#### decompose()

```python
decompose(
    problem: str,
    strategy: str = "sequential",
    max_depth: int = 3
) -> Dict[str, Any]
```

Decompose a problem into sub-problems.

**Parameters:**
- `problem` (str): The problem statement to decompose
- `strategy` (str): Decomposition strategy
  - `"sequential"`: Sequential steps
  - `"hierarchical"`: Tree structure with parent-child relationships
  - `"parallel"`: Independent parallel aspects
- `max_depth` (int): Maximum depth for hierarchical decomposition

**Returns:**
- Dictionary containing:
  - `original_problem`: Original problem text
  - `strategy`: Strategy used
  - `sub_problems`: List of sub-problem dictionaries
  - `count`: Number of sub-problems
  - `structure`: Structure analysis

**Example:**
```python
decomposer = ProblemDecomposer()
result = decomposer.decompose(
    "Complex problem...",
    strategy="hierarchical",
    max_depth=3
)
```

#### visualize_decomposition()

```python
visualize_decomposition(decomposition: Dict[str, Any]) -> str
```

Create a text-based visualization of the decomposition.

**Parameters:**
- `decomposition` (dict): Result from decompose() method

**Returns:**
- String representation of the decomposition tree

---

## ReasoningMetrics

Calculate quality metrics for reasoning.

### Constructor

```python
ReasoningMetrics()
```

### Methods

#### calculate_correctness()

```python
calculate_correctness(
    predicted_solution: str,
    ground_truth: Optional[str] = None,
    verification_result: Optional[Dict[str, Any]] = None
) -> float
```

Calculate correctness score (0-1).

#### calculate_completeness()

```python
calculate_completeness(
    reasoning_trace: Dict[str, Any],
    requirements: Optional[List[str]] = None
) -> float
```

Calculate completeness score (0-1).

#### calculate_coherence()

```python
calculate_coherence(reasoning_trace: Dict[str, Any]) -> float
```

Calculate coherence score (0-1).

#### calculate_efficiency()

```python
calculate_efficiency(
    reasoning_trace: Dict[str, Any],
    duration: float,
    problem_complexity: str = "medium"
) -> float
```

Calculate efficiency score (0-1).

#### calculate_clarity()

```python
calculate_clarity(
    solution: str,
    reasoning_trace: Dict[str, Any]
) -> float
```

Calculate clarity score (0-1).

#### calculate_all_metrics()

```python
calculate_all_metrics(
    problem: str,
    solution: str,
    reasoning_trace: Dict[str, Any],
    duration: float,
    ground_truth: Optional[str] = None
) -> Dict[str, float]
```

Calculate all metrics at once.

**Returns:**
- Dictionary with all metric scores plus overall score

**Example:**
```python
metrics = ReasoningMetrics()
scores = metrics.calculate_all_metrics(
    problem="What is 2+2?",
    solution="4",
    reasoning_trace=trace,
    duration=1.5
)
print(f"Overall: {scores['overall']:.3f}")
```

---

## MetricsCalculator

Utility class for batch metrics calculation.

### Constructor

```python
MetricsCalculator()
```

### Methods

#### evaluate_result()

```python
evaluate_result(
    result: Dict[str, Any],
    ground_truth: Optional[str] = None
) -> Dict[str, Any]
```

Evaluate a single reasoning result.

**Parameters:**
- `result` (dict): Result from ReasoningAgent.solve()
- `ground_truth` (str, optional): Ground truth solution

**Returns:**
- Dictionary containing metrics and analysis

#### evaluate_batch()

```python
evaluate_batch(
    results: List[Dict[str, Any]],
    ground_truths: Optional[List[str]] = None
) -> Dict[str, Any]
```

Evaluate multiple reasoning results.

**Returns:**
- Aggregate metrics and individual evaluations

#### get_summary_report()

```python
get_summary_report() -> str
```

Generate a summary report of all evaluations.

**Returns:**
- Formatted string report

---

## TunixTrainer

Fine-tuning trainer using Tunix/Gemini API.

### Constructor

```python
TunixTrainer(
    model_name: str = "gemini-pro",
    api_key: Optional[str] = None,
    output_dir: str = "./models",
    verbose: bool = True
)
```

### Methods

#### prepare_training_data()

```python
prepare_training_data(
    problems: List[str],
    solutions: List[str],
    reasoning_traces: Optional[List[Dict[str, Any]]] = None,
    output_file: Optional[str] = None
) -> List[Dict[str, Any]]
```

Prepare training data for fine-tuning.

**Parameters:**
- `problems` (list): List of problem statements
- `solutions` (list): List of solutions
- `reasoning_traces` (list, optional): Reasoning traces for each problem
- `output_file` (str, optional): File to save prepared data

**Returns:**
- List of training examples

#### fine_tune()

```python
fine_tune(
    training_data: List[Dict[str, Any]],
    validation_data: Optional[List[Dict[str, Any]]] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    eval_steps: int = 500,
    save_steps: int = 1000,
    max_length: int = 512
) -> Dict[str, Any]
```

Fine-tune the model on reasoning tasks.

**Returns:**
- Training results and metrics

#### save_checkpoint()

```python
save_checkpoint(checkpoint_name: Optional[str] = None) -> str
```

Save a training checkpoint.

**Returns:**
- Path to saved checkpoint

#### load_checkpoint()

```python
load_checkpoint(checkpoint_path: str) -> None
```

Load a training checkpoint.

---

## DatasetBuilder

Build and manage datasets for fine-tuning.

### Constructor

```python
DatasetBuilder(output_dir: str = "./data")
```

### Methods

#### create_from_examples()

```python
create_from_examples(
    examples: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
```

Create train/validation split from examples.

**Returns:**
- Tuple of (training_data, validation_data)

#### augment_dataset()

```python
augment_dataset(
    examples: List[Dict[str, Any]],
    augmentation_factor: int = 2
) -> List[Dict[str, Any]]
```

Augment dataset with variations.

#### get_statistics()

```python
get_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]
```

Get statistics about a dataset.

---

## Utilities

### ReasoningVisualizer

Visualize reasoning traces and results.

#### Constructor

```python
ReasoningVisualizer(style: str = "detailed")
```

#### Methods

##### visualize_trace()

```python
visualize_trace(
    reasoning_trace: Dict[str, Any],
    show_metadata: bool = True
) -> str
```

Create a text visualization of a reasoning trace.

##### visualize_result()

```python
visualize_result(
    result: Dict[str, Any],
    detailed: bool = False
) -> str
```

Visualize a complete reasoning result.

##### create_comparison_table()

```python
create_comparison_table(
    results: List[Dict[str, Any]],
    metrics: List[str] = ["verified", "confidence", "duration_seconds"]
) -> str
```

Create a comparison table for multiple results.

##### create_summary_report()

```python
create_summary_report(
    results: List[Dict[str, Any]],
    evaluations: Optional[List[Dict[str, Any]]] = None
) -> str
```

Create a comprehensive summary report.

---

## Example Workflows

### Complete Reasoning Workflow

```python
from tunix_reasoning_agent import ReasoningAgent
from tunix_reasoning_agent.evaluation import MetricsCalculator
from tunix_reasoning_agent.utils import ReasoningVisualizer

# Initialize
agent = ReasoningAgent(verbose=True)
calculator = MetricsCalculator()
visualizer = ReasoningVisualizer()

# Solve problem
result = agent.solve("Your problem here")

# Evaluate
evaluation = calculator.evaluate_result(result)

# Visualize
print(visualizer.visualize_result(result, detailed=True))
print(f"Overall Score: {evaluation['metrics']['overall']:.3f}")
```

### Fine-tuning Workflow

```python
from tunix_reasoning_agent import TunixTrainer
from tunix_reasoning_agent.fine_tuning import DatasetBuilder

# Prepare data
trainer = TunixTrainer(output_dir="./models")
builder = DatasetBuilder()

training_data = trainer.prepare_training_data(problems, solutions)
train_data, val_data = builder.create_from_examples(training_data)

# Fine-tune
results = trainer.fine_tune(
    training_data=train_data,
    validation_data=val_data,
    num_epochs=3
)

# Save
checkpoint = trainer.save_checkpoint("my_model_v1")
```
