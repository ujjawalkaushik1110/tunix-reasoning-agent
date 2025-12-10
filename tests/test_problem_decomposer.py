"""Tests for ProblemDecomposer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from tunix_reasoning_agent import ProblemDecomposer


class TestProblemDecomposer:
    """Test suite for ProblemDecomposer."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = ProblemDecomposer()
        assert len(decomposer.decomposition_strategies) == 3
    
    def test_sequential_decomposition(self):
        """Test sequential decomposition strategy."""
        decomposer = ProblemDecomposer()
        problem = "This is a test problem. It has multiple sentences. Each should be analyzed."
        
        result = decomposer.decompose(problem, strategy="sequential")
        
        assert result["strategy"] == "sequential"
        assert "sub_problems" in result
        assert len(result["sub_problems"]) > 0
        assert result["count"] > 0
    
    def test_hierarchical_decomposition(self):
        """Test hierarchical decomposition strategy."""
        decomposer = ProblemDecomposer()
        problem = "Complex hierarchical problem"
        
        result = decomposer.decompose(problem, strategy="hierarchical", max_depth=2)
        
        assert result["strategy"] == "hierarchical"
        assert "structure" in result
        assert "depth" in result["structure"]
    
    def test_parallel_decomposition(self):
        """Test parallel decomposition strategy."""
        decomposer = ProblemDecomposer()
        problem = "Problem with aspect A and aspect B, plus aspect C"
        
        result = decomposer.decompose(problem, strategy="parallel")
        
        assert result["strategy"] == "parallel"
        assert result["structure"]["parallelizable"] is True
    
    def test_invalid_strategy(self):
        """Test handling of invalid strategy."""
        decomposer = ProblemDecomposer()
        
        with pytest.raises(ValueError):
            decomposer.decompose("Test problem", strategy="invalid_strategy")
    
    def test_visualization(self):
        """Test decomposition visualization."""
        decomposer = ProblemDecomposer()
        problem = "Test problem for visualization"
        
        result = decomposer.decompose(problem, strategy="sequential")
        visualization = decomposer.visualize_decomposition(result)
        
        assert isinstance(visualization, str)
        assert "Problem Decomposition" in visualization
        assert result["strategy"] in visualization
    
    def test_structure_analysis(self):
        """Test structure analysis."""
        decomposer = ProblemDecomposer()
        problem = "Test problem"
        
        result = decomposer.decompose(problem, strategy="sequential")
        structure = result["structure"]
        
        assert "strategy" in structure
        assert "total_problems" in structure
        assert structure["total_problems"] == len(result["sub_problems"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
