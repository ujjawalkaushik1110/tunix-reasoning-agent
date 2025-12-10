"""
Main Reasoning Agent - Orchestrates the multi-step reasoning process.
"""

from typing import Dict, Any, Optional, List
import time
from datetime import datetime

from .reasoning_steps import UnderstandStep, PlanStep, ExecuteStep, VerifyStep
from .problem_decomposer import ProblemDecomposer


class ReasoningAgent:
    """
    Multi-step reasoning agent that solves problems using the
    Understand -> Plan -> Execute -> Verify framework.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the model (optional, can use env var)
            verbose: Whether to print detailed progress
        """
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose
        
        # Initialize reasoning steps
        self.understand_step = UnderstandStep()
        self.plan_step = PlanStep()
        self.execute_step = ExecuteStep()
        self.verify_step = VerifyStep()
        
        # Initialize problem decomposer
        self.decomposer = ProblemDecomposer()
        
        # Reasoning history
        self.history: List[Dict[str, Any]] = []
    
    def solve(
        self,
        problem: str,
        decompose: bool = True,
        decomposition_strategy: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Solve a problem using multi-step reasoning.
        
        Args:
            problem: The problem statement to solve
            decompose: Whether to decompose complex problems
            decomposition_strategy: Strategy for problem decomposition
            
        Returns:
            Dictionary containing complete reasoning trace and solution
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Starting reasoning process for problem:")
            print(f"{problem[:200]}{'...' if len(problem) > 200 else ''}")
            print(f"{'='*70}\n")
        
        # Optional: Decompose problem first
        decomposition = None
        if decompose and len(problem) > 100:
            decomposition = self._decompose_problem(problem, decomposition_strategy)
        
        # Execute reasoning steps
        context = {}
        
        # Step 1: Understand
        understand_result = self._execute_understand(problem, context)
        context["understand"] = understand_result
        
        # Step 2: Plan
        plan_result = self._execute_plan(problem, context)
        context["plan"] = plan_result
        
        # Step 3: Execute
        execute_result = self._execute_execute(problem, context)
        context["execute"] = execute_result
        
        # Step 4: Verify
        verify_result = self._execute_verify(problem, context)
        context["verify"] = verify_result
        
        # Compile final result
        end_time = time.time()
        
        result = {
            "problem": problem,
            "solution": execute_result["result"]["solution"],
            "reasoning_trace": {
                "understand": understand_result,
                "plan": plan_result,
                "execute": execute_result,
                "verify": verify_result
            },
            "decomposition": decomposition,
            "metadata": {
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": end_time - start_time,
                "verified": verify_result["result"]["valid"],
                "confidence": verify_result["metadata"]["confidence"]
            }
        }
        
        # Store in history
        self.history.append(result)
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _decompose_problem(self, problem: str, strategy: str) -> Dict[str, Any]:
        """Decompose the problem into sub-problems."""
        if self.verbose:
            print(f"[DECOMPOSE] Using {strategy} strategy...")
        
        decomposition = self.decomposer.decompose(problem, strategy=strategy)
        
        if self.verbose:
            print(self.decomposer.visualize_decomposition(decomposition))
        
        return decomposition
    
    def _execute_understand(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the understand step."""
        if self.verbose:
            print(f"\n[UNDERSTAND] Analyzing problem...")
        
        result = self.understand_step.execute(problem, context)
        
        if self.verbose:
            print(f"  → Problem type: {result['result']['problem_type']}")
            print(f"  → Complexity: {result['metadata']['complexity']}")
            print(f"  → Key concepts: {', '.join(result['result']['key_concepts'][:5])}")
        
        return result
    
    def _execute_plan(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan step."""
        if self.verbose:
            print(f"\n[PLAN] Creating solution strategy...")
        
        result = self.plan_step.execute(problem, context)
        
        if self.verbose:
            print(f"  → Approach: {result['result']['approach']}")
            print(f"  → Sub-problems: {result['metadata']['num_sub_problems']}")
            print(f"  → Steps: {result['metadata']['estimated_steps']}")
        
        return result
    
    def _execute_execute(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the execute step."""
        if self.verbose:
            print(f"\n[EXECUTE] Implementing solution...")
        
        result = self.execute_step.execute(problem, context)
        
        if self.verbose:
            print(f"  → Steps executed: {result['metadata']['steps_executed']}")
            print(f"  → All successful: {result['metadata']['all_steps_successful']}")
        
        return result
    
    def _execute_verify(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the verify step."""
        if self.verbose:
            print(f"\n[VERIFY] Validating solution...")
        
        result = self.verify_step.execute(problem, context)
        
        if self.verbose:
            print(f"  → Valid: {result['result']['valid']}")
            print(f"  → Confidence: {result['metadata']['confidence']:.2%}")
            print(f"  → Recommendations: {', '.join(result['result']['recommendations'])}")
        
        return result
    
    def _print_summary(self, result: Dict[str, Any]) -> None:
        """Print a summary of the reasoning process."""
        print(f"\n{'='*70}")
        print(f"REASONING COMPLETE")
        print(f"{'='*70}")
        print(f"Solution: {result['solution'][:200]}{'...' if len(result['solution']) > 200 else ''}")
        print(f"Verified: {result['metadata']['verified']}")
        print(f"Confidence: {result['metadata']['confidence']:.2%}")
        print(f"Duration: {result['metadata']['duration_seconds']:.2f}s")
        print(f"{'='*70}\n")
    
    def get_reasoning_trace(self, format: str = "dict") -> Any:
        """
        Get the reasoning trace from the last solution.
        
        Args:
            format: Output format ('dict', 'json', 'text')
            
        Returns:
            Reasoning trace in specified format
        """
        if not self.history:
            return None
        
        last_result = self.history[-1]
        
        if format == "dict":
            return last_result["reasoning_trace"]
        elif format == "json":
            import json
            return json.dumps(last_result["reasoning_trace"], indent=2)
        elif format == "text":
            return self._format_trace_as_text(last_result["reasoning_trace"])
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _format_trace_as_text(self, trace: Dict[str, Any]) -> str:
        """Format reasoning trace as readable text."""
        lines = ["\nReasoning Trace:", "=" * 60]
        
        for step_name, step_data in trace.items():
            lines.append(f"\n{step_name.upper()}:")
            lines.append("-" * 40)
            
            result = step_data.get("result", {})
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool)):
                        lines.append(f"  {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 3:
                        lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def clear_history(self) -> None:
        """Clear the reasoning history."""
        self.history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent's performance."""
        if not self.history:
            return {"total_problems": 0}
        
        total = len(self.history)
        verified = sum(1 for r in self.history if r["metadata"]["verified"])
        avg_duration = sum(r["metadata"]["duration_seconds"] for r in self.history) / total
        avg_confidence = sum(r["metadata"]["confidence"] for r in self.history) / total
        
        return {
            "total_problems": total,
            "verified_solutions": verified,
            "verification_rate": verified / total if total > 0 else 0,
            "average_duration": avg_duration,
            "average_confidence": avg_confidence,
            "problem_types": self._get_problem_type_distribution()
        }
    
    def _get_problem_type_distribution(self) -> Dict[str, int]:
        """Get distribution of problem types solved."""
        distribution = {}
        
        for result in self.history:
            problem_type = result["reasoning_trace"]["understand"]["result"]["problem_type"]
            distribution[problem_type] = distribution.get(problem_type, 0) + 1
        
        return distribution
