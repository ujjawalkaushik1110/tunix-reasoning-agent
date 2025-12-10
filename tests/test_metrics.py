"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from tunix_reasoning_agent.evaluation import ReasoningMetrics, MetricsCalculator


class TestReasoningMetrics:
    """Test suite for ReasoningMetrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = ReasoningMetrics()
        assert metrics.metrics == {}
    
    def test_calculate_correctness(self):
        """Test correctness calculation."""
        metrics = ReasoningMetrics()
        
        # With verification result
        verify_result = {"valid": True, "confidence": 0.9}
        score = metrics.calculate_correctness("Solution text", verification_result=verify_result)
        assert 0 <= score <= 1
        assert score > 0.5
    
    def test_calculate_completeness(self):
        """Test completeness calculation."""
        metrics = ReasoningMetrics()
        
        trace = {
            "understand": {"result": {"problem_type": "test"}},
            "plan": {"result": {"approach": "test"}},
            "execute": {"result": {"solution": "test"}},
            "verify": {"result": {"valid": True}}
        }
        
        score = metrics.calculate_completeness(trace)
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high for complete trace
    
    def test_calculate_coherence(self):
        """Test coherence calculation."""
        metrics = ReasoningMetrics()
        
        trace = {
            "understand": {"result": "analysis"},
            "plan": {"result": "strategy"},
            "execute": {"result": "solution"},
            "verify": {"result": {"consistency": {"consistent": True}}}
        }
        
        score = metrics.calculate_coherence(trace)
        assert 0 <= score <= 1
    
    def test_calculate_efficiency(self):
        """Test efficiency calculation."""
        metrics = ReasoningMetrics()
        
        trace = {
            "execute": {
                "result": {
                    "execution_trace": [
                        {"success": True},
                        {"success": True}
                    ]
                }
            }
        }
        
        score = metrics.calculate_efficiency(trace, duration=5.0, problem_complexity="low")
        assert 0 <= score <= 1
    
    def test_calculate_clarity(self):
        """Test clarity calculation."""
        metrics = ReasoningMetrics()
        
        solution = "This is a clear solution. It explains the problem well."
        trace = {
            "understand": {"result": {"problem_type": "test"}},
            "plan": {"result": {"approach": "systematic"}}
        }
        
        score = metrics.calculate_clarity(solution, trace)
        assert 0 <= score <= 1
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        metrics = ReasoningMetrics()
        
        problem = "Test problem"
        solution = "Test solution"
        trace = {
            "understand": {
                "result": {"problem_type": "general"},
                "metadata": {"complexity": "low"}
            },
            "plan": {"result": {"approach": "direct"}},
            "execute": {"result": {"execution_trace": []}},
            "verify": {"result": {"valid": True, "confidence": 0.8}}
        }
        
        all_metrics = metrics.calculate_all_metrics(problem, solution, trace, duration=3.0)
        
        assert "correctness" in all_metrics
        assert "completeness" in all_metrics
        assert "coherence" in all_metrics
        assert "efficiency" in all_metrics
        assert "clarity" in all_metrics
        assert "overall" in all_metrics


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = MetricsCalculator()
        assert len(calculator.results_history) == 0
    
    def test_evaluate_result(self):
        """Test evaluating a single result."""
        calculator = MetricsCalculator()
        
        result = {
            "problem": "Test problem",
            "solution": "Test solution",
            "reasoning_trace": {
                "understand": {"result": {}, "metadata": {}},
                "plan": {"result": {}},
                "execute": {"result": {}},
                "verify": {"result": {"valid": True}}
            },
            "metadata": {
                "verified": True,
                "confidence": 0.8,
                "duration_seconds": 2.0,
                "timestamp": "2024-01-01T00:00:00"
            }
        }
        
        evaluation = calculator.evaluate_result(result)
        
        assert "metrics" in evaluation
        assert "verified" in evaluation
        assert len(calculator.results_history) == 1
    
    def test_clear_history(self):
        """Test clearing calculator history."""
        calculator = MetricsCalculator()
        
        result = {
            "problem": "Test",
            "solution": "Solution",
            "reasoning_trace": {"understand": {"result": {}}},
            "metadata": {"verified": True, "confidence": 0.8, "duration_seconds": 1.0, "timestamp": "2024-01-01"}
        }
        
        calculator.evaluate_result(result)
        assert len(calculator.results_history) > 0
        
        calculator.clear_history()
        assert len(calculator.results_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
