"""
Basic usage example for Tunix Reasoning Agent.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tunix_reasoning_agent import ReasoningAgent
from tunix_reasoning_agent.evaluation import MetricsCalculator
from tunix_reasoning_agent.utils import ReasoningVisualizer


def main():
    """Run basic usage examples."""
    print("=" * 70)
    print("Tunix Reasoning Agent - Basic Usage Example")
    print("=" * 70)
    
    # Initialize agent
    agent = ReasoningAgent(verbose=True)
    
    # Example 1: Simple problem
    print("\n\n### Example 1: Simple Mathematical Problem ###\n")
    problem1 = "If a car travels at 60 km/h for 2.5 hours, how far does it travel?"
    result1 = agent.solve(problem1)
    
    # Example 2: Logical problem
    print("\n\n### Example 2: Logical Reasoning Problem ###\n")
    problem2 = """
    Three friends - Alice, Bob, and Charlie - have different favorite colors: red, blue, and green.
    Alice doesn't like red. Bob's favorite color is not blue. What is each person's favorite color?
    """
    result2 = agent.solve(problem2)
    
    # Example 3: Analytical problem
    print("\n\n### Example 3: Analytical Problem ###\n")
    problem3 = "What are the main differences between Python and JavaScript programming languages?"
    result3 = agent.solve(problem3)
    
    # Evaluate results
    print("\n\n### Evaluation ###\n")
    calculator = MetricsCalculator()
    
    results = [result1, result2, result3]
    batch_eval = calculator.evaluate_batch(results)
    
    print(calculator.get_summary_report())
    
    # Visualize
    visualizer = ReasoningVisualizer()
    print(visualizer.create_comparison_table(results))
    
    # Get agent statistics
    print("\n\n### Agent Statistics ###\n")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
