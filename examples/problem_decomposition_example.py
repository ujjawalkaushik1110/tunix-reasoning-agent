"""
Example demonstrating problem decomposition strategies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tunix_reasoning_agent import ProblemDecomposer


def main():
    """Demonstrate problem decomposition."""
    print("=" * 70)
    print("Problem Decomposition Example")
    print("=" * 70)
    
    # Complex problem
    problem = """
    A software company needs to develop a new mobile application with the following requirements:
    1. User authentication and profile management
    2. Real-time messaging between users
    3. Photo and video sharing capabilities
    4. Push notifications
    5. Analytics dashboard for administrators
    The project must be completed in 6 months with a team of 5 developers.
    Create a comprehensive development plan.
    """
    
    decomposer = ProblemDecomposer()
    
    # Try different decomposition strategies
    strategies = ["sequential", "hierarchical", "parallel"]
    
    for strategy in strategies:
        print(f"\n\n{'=' * 70}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'=' * 70}")
        
        decomposition = decomposer.decompose(
            problem,
            strategy=strategy,
            max_depth=3
        )
        
        print(decomposer.visualize_decomposition(decomposition))
        
        print(f"\nStructure Analysis:")
        print(f"-" * 70)
        structure = decomposition["structure"]
        for key, value in structure.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Decomposition examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
