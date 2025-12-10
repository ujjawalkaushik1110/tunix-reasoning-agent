"""
Problem Decomposer - Breaks down complex problems into manageable sub-problems.
"""

from typing import List, Dict, Any, Optional
import re

# Decomposition configuration constants
MAX_CHILDREN_PER_PARENT = 3
MIN_CHILDREN_PER_PARENT = 2
CHARS_PER_CHILD = 50


class ProblemDecomposer:
    """
    Decomposes complex problems into smaller, manageable sub-problems
    using the Understand -> Plan -> Execute -> Verify framework.
    """
    
    def __init__(self):
        self.decomposition_strategies = {
            "sequential": self._sequential_decomposition,
            "hierarchical": self._hierarchical_decomposition,
            "parallel": self._parallel_decomposition,
        }
    
    def decompose(
        self, 
        problem: str, 
        strategy: str = "sequential",
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Decompose a problem into sub-problems.
        
        Args:
            problem: The problem statement to decompose
            strategy: Decomposition strategy ('sequential', 'hierarchical', 'parallel')
            max_depth: Maximum depth for hierarchical decomposition
            
        Returns:
            Dictionary containing decomposed sub-problems and structure
        """
        if strategy not in self.decomposition_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.decomposition_strategies.keys())}")
        
        decomposition_func = self.decomposition_strategies[strategy]
        sub_problems = decomposition_func(problem, max_depth)
        
        return {
            "original_problem": problem,
            "strategy": strategy,
            "sub_problems": sub_problems,
            "count": len(sub_problems),
            "structure": self._analyze_structure(sub_problems, strategy)
        }
    
    def _sequential_decomposition(self, problem: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Decompose problem into sequential steps.
        Each step must be completed before the next.
        """
        # Split by sentences or logical breaks
        sentences = self._split_into_sentences(problem)
        
        sub_problems = []
        for i, sentence in enumerate(sentences[:max_depth * 2], 1):
            if sentence.strip():
                sub_problems.append({
                    "id": f"seq_{i}",
                    "description": sentence.strip(),
                    "order": i,
                    "type": "sequential",
                    "dependencies": [f"seq_{i-1}"] if i > 1 else []
                })
        
        # Ensure we have at least basic steps
        if not sub_problems:
            sub_problems = self._create_default_steps()
        
        return sub_problems
    
    def _hierarchical_decomposition(self, problem: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Decompose problem hierarchically with parent-child relationships.
        """
        sub_problems = []
        
        # Create main problem as root
        root = {
            "id": "root",
            "description": problem,
            "level": 0,
            "type": "hierarchical",
            "parent": None,
            "children": []
        }
        sub_problems.append(root)
        
        # Decompose into levels
        current_level_problems = [root]
        
        for level in range(1, max_depth + 1):
            next_level = []
            
            for parent_problem in current_level_problems:
                # Create 2-3 sub-problems for each parent based on complexity
                child_count = min(
                    MAX_CHILDREN_PER_PARENT,
                    max(MIN_CHILDREN_PER_PARENT, len(parent_problem["description"]) // CHARS_PER_CHILD)
                )
                
                for i in range(child_count):
                    child_id = f"{parent_problem['id']}_child_{i+1}"
                    child = {
                        "id": child_id,
                        "description": f"Sub-problem {i+1} of: {parent_problem['description'][:50]}...",
                        "level": level,
                        "type": "hierarchical",
                        "parent": parent_problem["id"],
                        "children": []
                    }
                    sub_problems.append(child)
                    parent_problem["children"].append(child_id)
                    next_level.append(child)
            
            current_level_problems = next_level
            if not current_level_problems:
                break
        
        return sub_problems
    
    def _parallel_decomposition(self, problem: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Decompose problem into independent parallel sub-problems.
        """
        # Identify independent aspects
        aspects = self._identify_aspects(problem)
        
        sub_problems = []
        for i, aspect in enumerate(aspects, 1):
            sub_problems.append({
                "id": f"par_{i}",
                "description": aspect,
                "type": "parallel",
                "dependencies": [],  # No dependencies for parallel execution
                "can_parallelize": True
            })
        
        if not sub_problems:
            sub_problems = self._create_default_parallel_steps()
        
        return sub_problems
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_aspects(self, problem: str) -> List[str]:
        """Identify different aspects of the problem."""
        aspects = []
        
        # Look for conjunctions that might indicate different aspects
        if ' and ' in problem.lower():
            parts = problem.split(' and ')
            aspects.extend([p.strip() for p in parts if p.strip()])
        elif ',' in problem:
            parts = problem.split(',')
            aspects.extend([p.strip() for p in parts if p.strip()])
        else:
            # Create default aspects based on problem analysis
            aspects = [
                f"Aspect 1: Initial analysis of the problem",
                f"Aspect 2: Core solution development",
                f"Aspect 3: Validation and verification"
            ]
        
        return aspects[:5]  # Limit to 5 aspects
    
    def _create_default_steps(self) -> List[Dict[str, Any]]:
        """Create default sequential steps."""
        return [
            {
                "id": "seq_1",
                "description": "Understand the problem requirements",
                "order": 1,
                "type": "sequential",
                "dependencies": []
            },
            {
                "id": "seq_2",
                "description": "Plan the solution approach",
                "order": 2,
                "type": "sequential",
                "dependencies": ["seq_1"]
            },
            {
                "id": "seq_3",
                "description": "Execute the solution",
                "order": 3,
                "type": "sequential",
                "dependencies": ["seq_2"]
            },
            {
                "id": "seq_4",
                "description": "Verify the results",
                "order": 4,
                "type": "sequential",
                "dependencies": ["seq_3"]
            }
        ]
    
    def _create_default_parallel_steps(self) -> List[Dict[str, Any]]:
        """Create default parallel steps."""
        return [
            {
                "id": "par_1",
                "description": "Analyze problem requirements",
                "type": "parallel",
                "dependencies": [],
                "can_parallelize": True
            },
            {
                "id": "par_2",
                "description": "Research solution approaches",
                "type": "parallel",
                "dependencies": [],
                "can_parallelize": True
            },
            {
                "id": "par_3",
                "description": "Identify edge cases",
                "type": "parallel",
                "dependencies": [],
                "can_parallelize": True
            }
        ]
    
    def _analyze_structure(self, sub_problems: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """Analyze the structure of decomposed problems."""
        structure = {
            "strategy": strategy,
            "total_problems": len(sub_problems),
            "max_dependencies": 0,
            "parallelizable": strategy == "parallel"
        }
        
        if strategy == "hierarchical":
            levels = set(sp.get("level", 0) for sp in sub_problems)
            structure["depth"] = max(levels) if levels else 0
            structure["levels"] = len(levels)
        
        # Calculate maximum dependencies
        for sp in sub_problems:
            deps = len(sp.get("dependencies", []))
            structure["max_dependencies"] = max(structure["max_dependencies"], deps)
        
        return structure
    
    def visualize_decomposition(self, decomposition: Dict[str, Any]) -> str:
        """
        Create a text-based visualization of the decomposition.
        
        Args:
            decomposition: Result from decompose() method
            
        Returns:
            String representation of the decomposition tree
        """
        strategy = decomposition["strategy"]
        sub_problems = decomposition["sub_problems"]
        
        lines = [f"\nProblem Decomposition ({strategy})"]
        lines.append("=" * 60)
        lines.append(f"Original: {decomposition['original_problem'][:100]}...")
        lines.append(f"Total sub-problems: {decomposition['count']}")
        lines.append("-" * 60)
        
        if strategy == "sequential":
            for sp in sub_problems:
                indent = "  " * len(sp.get("dependencies", []))
                lines.append(f"{indent}[{sp['id']}] {sp['description']}")
        
        elif strategy == "hierarchical":
            def print_tree(sp_id: str, indent: int = 0):
                sp = next((p for p in sub_problems if p["id"] == sp_id), None)
                if sp:
                    lines.append(f"{'  ' * indent}├─ [{sp['id']}] Level {sp['level']}: {sp['description'][:60]}...")
                    for child_id in sp.get("children", []):
                        print_tree(child_id, indent + 1)
            
            root = next((p for p in sub_problems if p.get("parent") is None), None)
            if root:
                print_tree(root["id"])
        
        elif strategy == "parallel":
            for sp in sub_problems:
                lines.append(f"║ [{sp['id']}] {sp['description']}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
