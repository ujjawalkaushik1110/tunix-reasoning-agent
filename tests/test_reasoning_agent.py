"""Tests for the main ReasoningAgent class."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from tunix_reasoning_agent import ReasoningAgent


class TestReasoningAgent:
    """Test suite for ReasoningAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = ReasoningAgent(verbose=False)
        assert agent.model_name == "gemini-pro"
        assert len(agent.history) == 0
    
    def test_simple_solve(self):
        """Test solving a simple problem."""
        agent = ReasoningAgent(verbose=False)
        problem = "What is 2 + 2?"
        result = agent.solve(problem, decompose=False)
        
        assert "problem" in result
        assert "solution" in result
        assert "reasoning_trace" in result
        assert "metadata" in result
        assert result["problem"] == problem
    
    def test_reasoning_steps_present(self):
        """Test that all reasoning steps are present."""
        agent = ReasoningAgent(verbose=False)
        problem = "Calculate 10 * 5"
        result = agent.solve(problem, decompose=False)
        
        trace = result["reasoning_trace"]
        assert "understand" in trace
        assert "plan" in trace
        assert "execute" in trace
        assert "verify" in trace
    
    def test_solve_with_decomposition(self):
        """Test solving with problem decomposition."""
        agent = ReasoningAgent(verbose=False)
        problem = "This is a complex problem that requires decomposition. " * 5
        result = agent.solve(problem, decompose=True)
        
        assert "decomposition" in result
        if result["decomposition"]:
            assert "sub_problems" in result["decomposition"]
    
    def test_history_tracking(self):
        """Test that agent tracks solution history."""
        agent = ReasoningAgent(verbose=False)
        
        problems = ["Problem 1", "Problem 2", "Problem 3"]
        for problem in problems:
            agent.solve(problem, decompose=False)
        
        assert len(agent.history) == 3
    
    def test_get_statistics(self):
        """Test statistics generation."""
        agent = ReasoningAgent(verbose=False)
        agent.solve("Test problem", decompose=False)
        
        stats = agent.get_statistics()
        assert "total_problems" in stats
        assert stats["total_problems"] == 1
        assert "verified_solutions" in stats
        assert "average_duration" in stats
    
    def test_clear_history(self):
        """Test clearing history."""
        agent = ReasoningAgent(verbose=False)
        agent.solve("Test problem", decompose=False)
        assert len(agent.history) > 0
        
        agent.clear_history()
        assert len(agent.history) == 0
    
    def test_reasoning_trace_format(self):
        """Test reasoning trace format."""
        agent = ReasoningAgent(verbose=False)
        result = agent.solve("Test problem", decompose=False)
        
        trace_dict = agent.get_reasoning_trace(format="dict")
        assert isinstance(trace_dict, dict)
        
        trace_text = agent.get_reasoning_trace(format="text")
        assert isinstance(trace_text, str)
        assert "REASONING TRACE" in trace_text.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
