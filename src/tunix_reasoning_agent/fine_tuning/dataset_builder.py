"""
Dataset builder for creating training datasets for reasoning tasks.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import random


class DatasetBuilder:
    """
    Builds and manages datasets for fine-tuning reasoning models.
    """
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize dataset builder.
        
        Args:
            output_dir: Directory for dataset outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_from_examples(
        self,
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create train/validation split from examples.
        
        Args:
            examples: List of problem-solution examples
            train_ratio: Ratio of training examples
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        if shuffle:
            random.seed(seed)
            examples = examples.copy()
            random.shuffle(examples)
        
        split_idx = int(len(examples) * train_ratio)
        train_data = examples[:split_idx]
        val_data = examples[split_idx:]
        
        return train_data, val_data
    
    def augment_dataset(
        self,
        examples: List[Dict[str, Any]],
        augmentation_factor: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Augment dataset with variations.
        
        Args:
            examples: Original examples
            augmentation_factor: How many variations per example
            
        Returns:
            Augmented dataset
        """
        augmented = examples.copy()
        
        for example in examples:
            for i in range(augmentation_factor - 1):
                # Create variation by rephrasing
                variation = self._create_variation(example, i)
                augmented.append(variation)
        
        return augmented
    
    def _create_variation(self, example: Dict[str, Any], variation_id: int) -> Dict[str, Any]:
        """Create a variation of an example."""
        variation = example.copy()
        
        # Add variation markers
        variation["variation_id"] = variation_id
        variation["original_problem"] = example["problem"]
        
        # Simple variation: add context or rephrase
        prefixes = [
            "Consider the following problem: ",
            "Here's a problem to solve: ",
            "Problem statement: ",
        ]
        
        prefix = prefixes[variation_id % len(prefixes)]
        variation["problem"] = prefix + example["problem"]
        
        return variation
    
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of examples
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def save_to_file(
        self,
        data: List[Dict[str, Any]],
        file_name: str,
        pretty: bool = True
    ) -> str:
        """
        Save dataset to JSON file.
        
        Args:
            data: Dataset to save
            file_name: Name of output file
            pretty: Whether to use pretty printing
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / file_name
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
        
        return str(output_path)
    
    def create_reasoning_dataset(
        self,
        problems: List[str],
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create a reasoning dataset with multi-step templates.
        
        Args:
            problems: List of problem statements
            categories: Optional categories for each problem
            
        Returns:
            Dataset with reasoning templates
        """
        dataset = []
        
        for i, problem in enumerate(problems):
            category = categories[i] if categories and i < len(categories) else "general"
            
            example = {
                "id": f"reasoning_{i+1}",
                "problem": problem,
                "category": category,
                "reasoning_template": {
                    "understand": "Analyze the problem and identify key requirements",
                    "plan": "Create a step-by-step solution strategy",
                    "execute": "Implement the solution with clear steps",
                    "verify": "Check the solution for correctness and completeness"
                },
                "expected_format": "Provide a solution following the 4-step reasoning process"
            }
            
            dataset.append(example)
        
        return dataset
    
    def get_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about a dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not dataset:
            return {"size": 0}
        
        # Count categories if present
        categories = {}
        problem_lengths = []
        
        for example in dataset:
            # Category distribution
            category = example.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
            
            # Problem lengths
            problem = example.get("problem", "")
            problem_lengths.append(len(problem))
        
        stats = {
            "size": len(dataset),
            "categories": categories,
            "avg_problem_length": sum(problem_lengths) / len(problem_lengths) if problem_lengths else 0,
            "min_problem_length": min(problem_lengths) if problem_lengths else 0,
            "max_problem_length": max(problem_lengths) if problem_lengths else 0,
        }
        
        return stats
