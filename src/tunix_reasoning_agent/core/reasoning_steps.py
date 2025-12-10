"""
Individual reasoning step implementations following the 
Understand -> Plan -> Execute -> Verify framework.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class ReasoningStep(ABC):
    """Abstract base class for reasoning steps."""
    
    def __init__(self, name: str):
        self.name = name
        self.context: Dict[str, Any] = {}
    
    @abstractmethod
    def execute(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the reasoning step.
        
        Args:
            problem: The problem or input for this step
            context: Optional context from previous steps
            
        Returns:
            Dictionary containing step results and metadata
        """
        pass
    
    def _format_output(self, result: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format the output of a reasoning step."""
        return {
            "step": self.name,
            "result": result,
            "metadata": metadata,
            "success": metadata.get("success", True)
        }


class UnderstandStep(ReasoningStep):
    """
    Understand Step: Analyze and comprehend the problem.
    - Extract key information
    - Identify problem type
    - Determine constraints and requirements
    """
    
    def __init__(self):
        super().__init__("understand")
    
    def execute(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Understand the problem by analyzing its components.
        
        Args:
            problem: The problem statement
            context: Optional context
            
        Returns:
            Understanding results including problem analysis
        """
        # Extract key information
        analysis = self._analyze_problem(problem)
        
        # Identify problem type
        problem_type = self._classify_problem(problem)
        
        # Extract constraints
        constraints = self._extract_constraints(problem)
        
        # Identify required information
        requirements = self._extract_requirements(problem)
        
        metadata = {
            "success": True,
            "problem_length": len(problem),
            "problem_type": problem_type,
            "complexity": self._estimate_complexity(problem)
        }
        
        result = {
            "analysis": analysis,
            "problem_type": problem_type,
            "constraints": constraints,
            "requirements": requirements,
            "key_concepts": self._extract_concepts(problem)
        }
        
        return self._format_output(result, metadata)
    
    def _analyze_problem(self, problem: str) -> str:
        """Analyze the problem statement."""
        lines = problem.split('\n')
        sentences = problem.split('.')
        
        return f"Problem consists of {len(lines)} lines and {len(sentences)} sentences. " \
               f"Primary focus: {sentences[0][:100] if sentences else 'Unknown'}"
    
    def _classify_problem(self, problem: str) -> str:
        """Classify the type of problem."""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['calculate', 'compute', 'solve', 'find']):
            return "mathematical"
        elif any(word in problem_lower for word in ['explain', 'describe', 'what is']):
            return "explanatory"
        elif any(word in problem_lower for word in ['compare', 'contrast', 'difference']):
            return "comparative"
        elif any(word in problem_lower for word in ['analyze', 'evaluate', 'assess']):
            return "analytical"
        else:
            return "general"
    
    def _extract_constraints(self, problem: str) -> List[str]:
        """Extract constraints from the problem."""
        constraints = []
        problem_lower = problem.lower()
        
        # Look for common constraint indicators
        constraint_keywords = ['must', 'should', 'cannot', 'only', 'within', 'less than', 'greater than']
        
        for keyword in constraint_keywords:
            if keyword in problem_lower:
                # Find sentences containing constraints
                for sentence in problem.split('.'):
                    if keyword in sentence.lower():
                        constraints.append(sentence.strip())
        
        return constraints if constraints else ["No explicit constraints found"]
    
    def _extract_requirements(self, problem: str) -> List[str]:
        """Extract requirements from the problem."""
        requirements = []
        
        # Look for questions and requirements
        for sentence in problem.split('.'):
            sentence = sentence.strip()
            if '?' in sentence or any(word in sentence.lower() for word in ['find', 'determine', 'calculate']):
                requirements.append(sentence)
        
        return requirements if requirements else ["Understand and solve the given problem"]
    
    def _extract_concepts(self, problem: str) -> List[str]:
        """Extract key concepts from the problem."""
        # Simple keyword extraction
        words = problem.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        concepts = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique concepts
        return list(set(concepts))[:10]  # Limit to top 10
    
    def _estimate_complexity(self, problem: str) -> str:
        """Estimate problem complexity."""
        length = len(problem)
        concepts = len(self._extract_concepts(problem))
        
        if length < 100 and concepts < 5:
            return "low"
        elif length < 300 and concepts < 15:
            return "medium"
        else:
            return "high"


class PlanStep(ReasoningStep):
    """
    Plan Step: Create a strategy to solve the problem.
    - Break down into sub-problems
    - Identify solution approach
    - Determine steps and order
    """
    
    def __init__(self):
        super().__init__("plan")
    
    def execute(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a plan to solve the problem.
        
        Args:
            problem: The problem statement
            context: Context from understand step
            
        Returns:
            Plan including sub-problems and approach
        """
        understanding = context.get("understand", {}).get("result", {}) if context else {}
        problem_type = understanding.get("problem_type", "general")
        
        # Generate solution approach
        approach = self._generate_approach(problem, problem_type)
        
        # Break into sub-problems
        sub_problems = self._decompose_problem(problem, understanding)
        
        # Create step sequence
        steps = self._create_step_sequence(sub_problems, approach)
        
        metadata = {
            "success": True,
            "approach": approach,
            "num_sub_problems": len(sub_problems),
            "estimated_steps": len(steps)
        }
        
        result = {
            "approach": approach,
            "sub_problems": sub_problems,
            "steps": steps,
            "dependencies": self._identify_dependencies(steps)
        }
        
        return self._format_output(result, metadata)
    
    def _generate_approach(self, problem: str, problem_type: str) -> str:
        """Generate solution approach based on problem type."""
        approaches = {
            "mathematical": "Use mathematical reasoning and calculations",
            "explanatory": "Provide detailed explanation with examples",
            "comparative": "Compare and contrast key aspects systematically",
            "analytical": "Analyze components and their relationships",
            "general": "Break down the problem and solve step by step"
        }
        return approaches.get(problem_type, approaches["general"])
    
    def _decompose_problem(self, problem: str, understanding: Dict[str, Any]) -> List[str]:
        """Decompose problem into sub-problems."""
        requirements = understanding.get("requirements", [])
        
        if len(requirements) > 1:
            return requirements
        
        # Default decomposition
        return [
            "Identify the main question or goal",
            "Gather necessary information",
            "Apply relevant methods or formulas",
            "Validate the solution"
        ]
    
    def _create_step_sequence(self, sub_problems: List[str], approach: str) -> List[Dict[str, str]]:
        """Create ordered sequence of steps."""
        steps = []
        for i, sub_problem in enumerate(sub_problems, 1):
            steps.append({
                "step_number": i,
                "description": sub_problem,
                "status": "pending"
            })
        return steps
    
    def _identify_dependencies(self, steps: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Identify dependencies between steps."""
        dependencies = []
        for i in range(1, len(steps)):
            dependencies.append({
                "step": i + 1,
                "depends_on": [i],
                "type": "sequential"
            })
        return dependencies


class ExecuteStep(ReasoningStep):
    """
    Execute Step: Implement the plan and solve the problem.
    - Execute each sub-step
    - Show intermediate results
    - Handle errors and edge cases
    """
    
    def __init__(self):
        super().__init__("execute")
    
    def execute(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the plan to solve the problem.
        
        Args:
            problem: The problem statement
            context: Context from previous steps
            
        Returns:
            Execution results with intermediate steps
        """
        plan = context.get("plan", {}).get("result", {}) if context else {}
        steps = plan.get("steps", [])
        
        # Execute each step
        execution_results = []
        for step in steps:
            step_result = self._execute_step(step, problem, context)
            execution_results.append(step_result)
        
        # Compile final solution
        solution = self._compile_solution(execution_results, problem)
        
        metadata = {
            "success": True,
            "steps_executed": len(execution_results),
            "all_steps_successful": all(r.get("success", False) for r in execution_results)
        }
        
        result = {
            "solution": solution,
            "execution_trace": execution_results,
            "intermediate_results": [r.get("result", "") for r in execution_results]
        }
        
        return self._format_output(result, metadata)
    
    def _execute_step(self, step: Dict[str, str], problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step."""
        step_num = step.get("step_number", 0)
        description = step.get("description", "")
        
        # Simulate step execution
        result = f"Completed: {description}"
        
        return {
            "step_number": step_num,
            "description": description,
            "result": result,
            "success": True
        }
    
    def _compile_solution(self, execution_results: List[Dict[str, Any]], problem: str) -> str:
        """Compile final solution from execution results."""
        solution_parts = []
        
        for result in execution_results:
            if result.get("success"):
                solution_parts.append(result.get("result", ""))
        
        if solution_parts:
            return "\n".join(solution_parts)
        else:
            return "Solution could not be compiled from execution results"


class VerifyStep(ReasoningStep):
    """
    Verify Step: Validate the solution.
    - Check correctness
    - Validate against constraints
    - Identify potential issues
    """
    
    def __init__(self):
        super().__init__("verify")
    
    def execute(self, problem: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify the solution.
        
        Args:
            problem: The original problem
            context: Context from all previous steps
            
        Returns:
            Verification results
        """
        understanding = context.get("understand", {}).get("result", {}) if context else {}
        execution = context.get("execute", {}).get("result", {}) if context else {}
        
        solution = execution.get("solution", "")
        constraints = understanding.get("constraints", [])
        requirements = understanding.get("requirements", [])
        
        # Check solution completeness
        completeness = self._check_completeness(solution, requirements)
        
        # Verify constraints
        constraint_check = self._verify_constraints(solution, constraints)
        
        # Check for logical consistency
        consistency = self._check_consistency(solution)
        
        # Overall validation
        is_valid = completeness["complete"] and constraint_check["satisfied"] and consistency["consistent"]
        
        metadata = {
            "success": True,
            "solution_valid": is_valid,
            "confidence": self._calculate_confidence(completeness, constraint_check, consistency)
        }
        
        result = {
            "valid": is_valid,
            "completeness": completeness,
            "constraint_satisfaction": constraint_check,
            "consistency": consistency,
            "recommendations": self._generate_recommendations(completeness, constraint_check, consistency)
        }
        
        return self._format_output(result, metadata)
    
    def _check_completeness(self, solution: str, requirements: List[str]) -> Dict[str, Any]:
        """Check if solution addresses all requirements."""
        if not solution:
            return {"complete": False, "missing": requirements}
        
        # Simple check: solution should not be empty and should have content
        return {
            "complete": len(solution) > 0,
            "coverage": min(len(solution) / 100, 1.0),  # Normalized coverage
            "missing": []
        }
    
    def _verify_constraints(self, solution: str, constraints: List[str]) -> Dict[str, Any]:
        """Verify solution satisfies constraints."""
        if not constraints or constraints == ["No explicit constraints found"]:
            return {"satisfied": True, "violations": []}
        
        # Simple verification: assume satisfied if solution exists
        return {
            "satisfied": len(solution) > 0,
            "violations": []
        }
    
    def _check_consistency(self, solution: str) -> Dict[str, Any]:
        """Check logical consistency of solution."""
        # Basic consistency checks
        has_content = len(solution) > 0
        no_contradictions = "error" not in solution.lower() and "invalid" not in solution.lower()
        
        return {
            "consistent": has_content and no_contradictions,
            "issues": [] if has_content and no_contradictions else ["Solution may have issues"]
        }
    
    def _calculate_confidence(self, completeness: Dict, constraints: Dict, consistency: Dict) -> float:
        """Calculate confidence score."""
        scores = []
        
        if completeness.get("complete"):
            scores.append(completeness.get("coverage", 0.5))
        else:
            scores.append(0.3)
        
        if constraints.get("satisfied"):
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        if consistency.get("consistent"):
            scores.append(1.0)
        else:
            scores.append(0.4)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _generate_recommendations(self, completeness: Dict, constraints: Dict, consistency: Dict) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        if not completeness.get("complete"):
            recommendations.append("Address missing requirements")
        
        if not constraints.get("satisfied"):
            recommendations.append("Review constraint violations")
        
        if not consistency.get("consistent"):
            recommendations.append("Check for logical inconsistencies")
        
        if not recommendations:
            recommendations.append("Solution appears valid and complete")
        
        return recommendations
