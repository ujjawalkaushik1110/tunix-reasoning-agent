# Tunix Reasoning Agent

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)

A Python-based LLM reasoning agent using Google's Tunix library to solve problems through structured multi-step reasoning. Built for the Google Tunix Hack hackathon.

## 🌟 Features

- **Multi-step Reasoning Framework**: Understand → Plan → Execute → Verify
- **Problem Decomposition**: Break complex problems into manageable sub-problems
- **Comprehensive Evaluation**: Quality metrics for reasoning assessment
- **Fine-tuning Support**: Train models on reasoning tasks using Tunix
- **Transparent Reasoning**: Full trace visibility for explainability
- **Multiple Strategies**: Sequential, hierarchical, and parallel decomposition

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ujjawalkaushik1110/tunix-reasoning-agent.git
cd tunix-reasoning-agent

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from tunix_reasoning_agent import ReasoningAgent

# Initialize the agent
agent = ReasoningAgent(verbose=True)

# Solve a problem
problem = "If a car travels at 60 km/h for 2.5 hours, how far does it travel?"
result = agent.solve(problem)

print(f"Solution: {result['solution']}")
print(f"Verified: {result['metadata']['verified']}")
print(f"Confidence: {result['metadata']['confidence']:.2%}")
```

## 📖 Documentation

### Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│         Tunix Reasoning Agent                │
├─────────────────────────────────────────────┤
│  Core Modules                                │
│  ├─ Reasoning Engine (4-step process)       │
│  ├─ Problem Decomposer (3 strategies)       │
│  ├─ Evaluation Metrics (5 dimensions)       │
│  └─ Fine-tuning Trainer (Tunix-based)       │
└─────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

### The Four-Step Reasoning Process

1. **Understand**: Analyze the problem, extract key information, identify constraints
2. **Plan**: Create a solution strategy, break into sub-steps, determine dependencies
3. **Execute**: Implement the solution, track intermediate results, handle edge cases
4. **Verify**: Validate correctness, check constraints, assess consistency

### Problem Decomposition Strategies

#### Sequential Decomposition
Problems are broken down into sequential steps that must be completed in order.

```python
decomposer = ProblemDecomposer()
result = decomposer.decompose(problem, strategy="sequential")
```

#### Hierarchical Decomposition
Problems are organized in a tree structure with parent-child relationships.

```python
result = decomposer.decompose(problem, strategy="hierarchical", max_depth=3)
```

#### Parallel Decomposition
Independent aspects of the problem that can be solved in parallel.

```python
result = decomposer.decompose(problem, strategy="parallel")
```

## 📊 Evaluation Metrics

The agent provides comprehensive evaluation across five dimensions:

- **Correctness**: Is the solution accurate and valid?
- **Completeness**: Does it address all requirements?
- **Coherence**: Is the reasoning logically consistent?
- **Efficiency**: How quickly was the problem solved?
- **Clarity**: Is the explanation clear and understandable?

```python
from tunix_reasoning_agent.evaluation import MetricsCalculator

calculator = MetricsCalculator()
evaluation = calculator.evaluate_result(result)

print(f"Correctness: {evaluation['metrics']['correctness']:.3f}")
print(f"Overall Score: {evaluation['metrics']['overall']:.3f}")
```

## 🔧 Fine-tuning

Train models on reasoning tasks:

```python
from tunix_reasoning_agent import TunixTrainer

trainer = TunixTrainer(output_dir="./models")

# Prepare training data
training_data = trainer.prepare_training_data(
    problems=problems,
    solutions=solutions,
    reasoning_traces=traces
)

# Fine-tune the model
results = trainer.fine_tune(
    training_data=training_data,
    num_epochs=3,
    batch_size=8
)
```

## 📓 Examples

### Example 1: Mathematical Problem

```python
agent = ReasoningAgent(verbose=True)
problem = "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
result = agent.solve(problem)
```

### Example 2: Complex Problem with Decomposition

```python
complex_problem = """
Design an efficient supply chain distribution strategy for a company 
with 5 warehouses, 20 distribution centers, and 100 retail stores.
"""

result = agent.solve(complex_problem, decompose=True, decomposition_strategy="hierarchical")
```

### Example 3: Batch Processing

```python
problems = [
    "What is 25% of 200?",
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "Explain the difference between machine learning and deep learning."
]

results = [agent.solve(p) for p in problems]

# Evaluate all results
from tunix_reasoning_agent.evaluation import MetricsCalculator
calculator = MetricsCalculator()
batch_eval = calculator.evaluate_batch(results)
print(calculator.get_summary_report())
```

## 📚 Notebooks

Interactive Jupyter notebooks with examples:

- [Demo Notebook](notebooks/demo_reasoning_agent.ipynb): Comprehensive examples and use cases

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_reasoning_agent.py -v

# Run with coverage
pytest tests/ --cov=src/tunix_reasoning_agent --cov-report=html
```

## 📦 Project Structure

```
tunix-reasoning-agent/
├── src/
│   └── tunix_reasoning_agent/
│       ├── core/                 # Core reasoning engine
│       ├── evaluation/           # Metrics and evaluation
│       ├── fine_tuning/         # Training utilities
│       └── utils/               # Helper utilities
├── tests/                       # Test suite
├── notebooks/                   # Jupyter notebooks
├── examples/                    # Example scripts
├── docs/                        # Documentation
└── data/                        # Data directory
```

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for the Google Tunix Hack hackathon
- Inspired by chain-of-thought reasoning and problem decomposition techniques
- Uses Google's Tunix/Gemini API for LLM capabilities

## 📧 Contact

Ujjawal Kaushik - [@ujjawalkaushik1110](https://github.com/ujjawalkaushik1110)

Project Link: [https://github.com/ujjawalkaushik1110/tunix-reasoning-agent](https://github.com/ujjawalkaushik1110/tunix-reasoning-agent)

## 🗺️ Roadmap

- [ ] Full Tunix/Gemini API integration
- [ ] Multi-modal reasoning support (text, images, code)
- [ ] Interactive reasoning with user feedback
- [ ] Distributed training capabilities
- [ ] Web interface for easy access
- [ ] Extended evaluation benchmarks
- [ ] Support for custom reasoning strategies

---

**Note**: This is an alpha version built for the hackathon. Some features (like actual Tunix API integration) are currently simulated and will be fully implemented in future releases.
