"""
Evaluation metrics for assessing reasoning quality.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required for evaluation metrics. "
        "Install it with: pip install numpy"
    )


class ReasoningMetrics:
    """
    Comprehensive metrics for evaluating reasoning quality.
    
    Metrics include:
    - Correctness: Is the solution correct?
    - Completeness: Does it address all requirements?
    - Coherence: Is the reasoning logical and consistent?
    - Efficiency: How efficiently was the problem solved?
    - Clarity: Is the explanation clear and understandable?
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_correctness(
        self,
        predicted_solution: str,
        ground_truth: Optional[str] = None,
        verification_result: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate correctness score.
        
        Args:
            predicted_solution: The generated solution
            ground_truth: Optional ground truth solution
            verification_result: Result from verification step
            
        Returns:
            Correctness score between 0 and 1
        """
        if ground_truth:
            # Compare with ground truth
            similarity = self._calculate_similarity(predicted_solution, ground_truth)
            return similarity
        elif verification_result:
            # Use verification results
            is_valid = verification_result.get("valid", False)
            confidence = verification_result.get("confidence", 0.5)
            return confidence if is_valid else confidence * 0.5
        else:
            # Basic heuristics
            if not predicted_solution or len(predicted_solution) < 10:
                return 0.0
            
            # Check for error indicators
            error_terms = ["error", "cannot", "unable", "failed", "invalid"]
            has_errors = any(term in predicted_solution.lower() for term in error_terms)
            
            return 0.3 if has_errors else 0.7
    
    def calculate_completeness(
        self,
        reasoning_trace: Dict[str, Any],
        requirements: Optional[List[str]] = None
    ) -> float:
        """
        Calculate completeness score.
        
        Args:
            reasoning_trace: Complete reasoning trace
            requirements: Optional list of requirements
            
        Returns:
            Completeness score between 0 and 1
        """
        required_steps = ["understand", "plan", "execute", "verify"]
        completed_steps = sum(1 for step in required_steps if step in reasoning_trace)
        
        base_score = completed_steps / len(required_steps)
        
        # Check if each step produced meaningful output
        quality_scores = []
        for step in required_steps:
            if step in reasoning_trace:
                step_result = reasoning_trace[step].get("result", {})
                if isinstance(step_result, dict) and step_result:
                    quality_scores.append(1.0)
                elif isinstance(step_result, str) and len(step_result) > 20:
                    quality_scores.append(0.8)
                else:
                    quality_scores.append(0.5)
        
        quality_score = np.mean(quality_scores) if quality_scores else 0.5
        
        return (base_score * 0.6 + quality_score * 0.4)
    
    def calculate_coherence(self, reasoning_trace: Dict[str, Any]) -> float:
        """
        Calculate coherence score - how logically connected the reasoning steps are.
        
        Args:
            reasoning_trace: Complete reasoning trace
            
        Returns:
            Coherence score between 0 and 1
        """
        scores = []
        
        # Check for logical flow between steps
        step_sequence = ["understand", "plan", "execute", "verify"]
        
        for i in range(len(step_sequence) - 1):
            current_step = step_sequence[i]
            next_step = step_sequence[i + 1]
            
            if current_step in reasoning_trace and next_step in reasoning_trace:
                # Check if next step references or builds on current step
                current_result = str(reasoning_trace[current_step].get("result", ""))
                next_result = str(reasoning_trace[next_step].get("result", ""))
                
                if current_result and next_result:
                    scores.append(1.0)
                else:
                    scores.append(0.5)
        
        # Check for consistency
        verify_result = reasoning_trace.get("verify", {}).get("result", {})
        if verify_result.get("consistency", {}).get("consistent", False):
            scores.append(1.0)
        else:
            scores.append(0.6)
        
        return np.mean(scores) if scores else 0.5
    
    def calculate_efficiency(
        self,
        reasoning_trace: Dict[str, Any],
        duration: float,
        problem_complexity: str = "medium"
    ) -> float:
        """
        Calculate efficiency score.
        
        Args:
            reasoning_trace: Complete reasoning trace
            duration: Time taken in seconds
            problem_complexity: Estimated problem complexity
            
        Returns:
            Efficiency score between 0 and 1
        """
        # Complexity-based thresholds
        time_thresholds = {
            "low": 5.0,
            "medium": 15.0,
            "high": 30.0
        }
        
        threshold = time_thresholds.get(problem_complexity, 15.0)
        
        # Calculate time efficiency
        if duration <= threshold:
            time_score = 1.0
        elif duration <= threshold * 2:
            time_score = 0.7
        else:
            time_score = 0.4
        
        # Calculate step efficiency (fewer steps with good results = better)
        execute_trace = reasoning_trace.get("execute", {}).get("result", {})
        execution_results = execute_trace.get("execution_trace", [])
        
        if execution_results:
            successful_steps = sum(1 for r in execution_results if r.get("success", False))
            step_efficiency = successful_steps / len(execution_results)
        else:
            step_efficiency = 0.5
        
        return (time_score * 0.6 + step_efficiency * 0.4)
    
    def calculate_clarity(
        self,
        solution: str,
        reasoning_trace: Dict[str, Any]
    ) -> float:
        """
        Calculate clarity score - how clear and understandable the reasoning is.
        
        Args:
            solution: The final solution
            reasoning_trace: Complete reasoning trace
            
        Returns:
            Clarity score between 0 and 1
        """
        scores = []
        
        # Check solution clarity
        if solution:
            # Length check (not too short, not too verbose)
            length = len(solution)
            if 50 <= length <= 1000:
                length_score = 1.0
            elif length < 50:
                length_score = 0.4
            else:
                length_score = 0.8
            scores.append(length_score)
            
            # Structure check (has sentences, punctuation)
            has_structure = '.' in solution or '\n' in solution
            scores.append(1.0 if has_structure else 0.6)
        
        # Check reasoning trace clarity
        for step_name, step_data in reasoning_trace.items():
            result = step_data.get("result", {})
            if isinstance(result, dict) and result:
                # Well-structured output
                scores.append(1.0)
            elif isinstance(result, str) and result:
                scores.append(0.8)
            else:
                scores.append(0.5)
        
        return np.mean(scores) if scores else 0.5
    
    def calculate_all_metrics(
        self,
        problem: str,
        solution: str,
        reasoning_trace: Dict[str, Any],
        duration: float,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics at once.
        
        Args:
            problem: Original problem
            solution: Generated solution
            reasoning_trace: Complete reasoning trace
            duration: Time taken
            ground_truth: Optional ground truth solution
            
        Returns:
            Dictionary of all metric scores
        """
        # Get problem complexity from understand step
        complexity = reasoning_trace.get("understand", {}).get("metadata", {}).get("complexity", "medium")
        
        # Get verification result
        verify_result = reasoning_trace.get("verify", {}).get("result", {})
        
        metrics = {
            "correctness": self.calculate_correctness(solution, ground_truth, verify_result),
            "completeness": self.calculate_completeness(reasoning_trace),
            "coherence": self.calculate_coherence(reasoning_trace),
            "efficiency": self.calculate_efficiency(reasoning_trace, duration, complexity),
            "clarity": self.calculate_clarity(solution, reasoning_trace)
        }
        
        # Calculate overall score
        metrics["overall"] = np.mean(list(metrics.values()))
        
        return metrics


class MetricsCalculator:
    """
    Utility class for batch metrics calculation and analysis.
    """
    
    def __init__(self):
        self.reasoning_metrics = ReasoningMetrics()
        self.results_history: List[Dict[str, Any]] = []
    
    def evaluate_result(
        self,
        result: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single reasoning result.
        
        Args:
            result: Result from ReasoningAgent.solve()
            ground_truth: Optional ground truth solution
            
        Returns:
            Dictionary containing all metrics and analysis
        """
        metrics = self.reasoning_metrics.calculate_all_metrics(
            problem=result["problem"],
            solution=result["solution"],
            reasoning_trace=result["reasoning_trace"],
            duration=result["metadata"]["duration_seconds"],
            ground_truth=ground_truth
        )
        
        evaluation = {
            "problem": result["problem"][:100] + "...",
            "metrics": metrics,
            "verified": result["metadata"]["verified"],
            "confidence": result["metadata"]["confidence"],
            "duration": result["metadata"]["duration_seconds"],
            "timestamp": result["metadata"]["timestamp"]
        }
        
        self.results_history.append(evaluation)
        
        return evaluation
    
    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple reasoning results.
        
        Args:
            results: List of results from ReasoningAgent.solve()
            ground_truths: Optional list of ground truth solutions
            
        Returns:
            Aggregate metrics and statistics
        """
        evaluations = []
        
        for i, result in enumerate(results):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            evaluation = self.evaluate_result(result, gt)
            evaluations.append(evaluation)
        
        # Aggregate metrics
        aggregate = self._aggregate_metrics(evaluations)
        
        return {
            "total_results": len(evaluations),
            "aggregate_metrics": aggregate,
            "individual_evaluations": evaluations
        }
    
    def _aggregate_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple evaluations."""
        if not evaluations:
            return {}
        
        metric_names = ["correctness", "completeness", "coherence", "efficiency", "clarity", "overall"]
        aggregated = {}
        
        for metric in metric_names:
            values = [e["metrics"][metric] for e in evaluations if metric in e["metrics"]]
            if values:
                aggregated[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }
        
        # Additional statistics
        aggregated["verification_rate"] = sum(1 for e in evaluations if e["verified"]) / len(evaluations)
        aggregated["avg_confidence"] = np.mean([e["confidence"] for e in evaluations])
        aggregated["avg_duration"] = np.mean([e["duration"] for e in evaluations])
        
        return aggregated
    
    def get_summary_report(self) -> str:
        """Generate a summary report of all evaluations."""
        if not self.results_history:
            return "No evaluations performed yet."
        
        aggregate = self._aggregate_metrics(self.results_history)
        
        lines = [
            "\n" + "=" * 70,
            "REASONING EVALUATION SUMMARY",
            "=" * 70,
            f"Total Evaluations: {len(self.results_history)}",
            "",
            "Aggregate Metrics:",
            "-" * 70
        ]
        
        metric_names = ["correctness", "completeness", "coherence", "efficiency", "clarity", "overall"]
        for metric in metric_names:
            if metric in aggregate:
                stats = aggregate[metric]
                lines.append(f"{metric.capitalize():15} → Mean: {stats['mean']:.3f}, "
                           f"Std: {stats['std']:.3f}, Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        lines.extend([
            "",
            "Additional Statistics:",
            "-" * 70,
            f"Verification Rate: {aggregate['verification_rate']:.2%}",
            f"Average Confidence: {aggregate['avg_confidence']:.3f}",
            f"Average Duration: {aggregate['avg_duration']:.2f}s",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.results_history = []
