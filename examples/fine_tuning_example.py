"""
Example demonstrating fine-tuning workflow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tunix_reasoning_agent import TunixTrainer
from tunix_reasoning_agent.fine_tuning import DatasetBuilder


def main():
    """Demonstrate fine-tuning workflow."""
    print("=" * 70)
    print("Fine-tuning Workflow Example")
    print("=" * 70)
    
    # Sample training data
    problems = [
        "What is 15% of 240?",
        "A train travels 180 km in 3 hours. What is its speed?",
        "Calculate the area of a circle with radius 7 cm.",
        "If x + 5 = 12, what is x?",
        "What is the square root of 144?",
    ]
    
    solutions = [
        "15% of 240 = 0.15 × 240 = 36",
        "Speed = Distance / Time = 180 km / 3 hours = 60 km/h",
        "Area = π × r² = π × 7² = π × 49 ≈ 153.94 cm²",
        "x + 5 = 12, therefore x = 12 - 5 = 7",
        "√144 = 12",
    ]
    
    # Initialize trainer
    trainer = TunixTrainer(output_dir="./models", verbose=True)
    
    # Prepare training data
    print("\n### Step 1: Preparing Training Data ###\n")
    training_data = trainer.prepare_training_data(
        problems=problems,
        solutions=solutions,
        output_file="math_problems_training.json"
    )
    
    print(f"Prepared {len(training_data)} training examples")
    
    # Build dataset with train/val split
    print("\n### Step 2: Building Dataset ###\n")
    builder = DatasetBuilder(output_dir="./data/processed")
    
    # Create more examples for demonstration
    extended_problems = problems * 5  # Repeat to have more data
    extended_solutions = solutions * 5
    
    training_data = trainer.prepare_training_data(
        problems=extended_problems,
        solutions=extended_solutions
    )
    
    train_data, val_data = builder.create_from_examples(
        training_data,
        train_ratio=0.8,
        shuffle=True
    )
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Get dataset statistics
    stats = builder.get_statistics(train_data)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Fine-tune model (simulated)
    print("\n### Step 3: Fine-tuning Model ###\n")
    results = trainer.fine_tune(
        training_data=train_data,
        validation_data=val_data,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
    
    print("\n### Training Complete ###")
    print(trainer.get_training_summary())
    
    # Save checkpoint
    print("\n### Step 4: Saving Checkpoint ###\n")
    checkpoint_path = trainer.save_checkpoint("math_model_v1")
    print(f"Checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("Fine-tuning example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
