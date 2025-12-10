"""Visualization utilities for reasoning traces."""

from typing import Dict, Any, List, Optional
import json


class ReasoningVisualizer:
    """Visualize reasoning traces and results."""
    
    def __init__(self, style: str = "detailed"):
        """
        Initialize visualizer.
        
        Args:
            style: Visualization style ('detailed', 'compact', 'minimal')
        """
        self.style = style
    
    def visualize_trace(
        self,
        reasoning_trace: Dict[str, Any],
        show_metadata: bool = True
    ) -> str:
        """
        Create a text visualization of a reasoning trace.
        
        Args:
            reasoning_trace: Reasoning trace to visualize
            show_metadata: Whether to show metadata
            
        Returns:
            Formatted string representation
        """
        lines = ["\n" + "=" * 80, "REASONING TRACE VISUALIZATION", "=" * 80]
        
        steps = ["understand", "plan", "execute", "verify"]
        step_icons = {
            "understand": "🔍",
            "plan": "📋",
            "execute": "⚙️",
            "verify": "✓"
        }
        
        for step_name in steps:
            if step_name not in reasoning_trace:
                continue
            
            step_data = reasoning_trace[step_name]
            icon = step_icons.get(step_name, "•")
            
            lines.append(f"\n{icon} {step_name.upper()}")
            lines.append("-" * 80)
            
            # Show result
            result = step_data.get("result", {})
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (str, int, float, bool)):
                        lines.append(f"  {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
            
            # Show metadata if requested
            if show_metadata and "metadata" in step_data:
                lines.append("\n  Metadata:")
                for key, value in step_data["metadata"].items():
                    lines.append(f"    {key}: {value}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def visualize_result(
        self,
        result: Dict[str, Any],
        detailed: bool = False
    ) -> str:
        """
        Visualize a complete reasoning result.
        
        Args:
            result: Result from ReasoningAgent.solve()
            detailed: Whether to show detailed trace
            
        Returns:
            Formatted visualization
        """
        lines = [
            "\n" + "=" * 80,
            "REASONING RESULT",
            "=" * 80,
            f"\nProblem:",
            f"  {result['problem'][:200]}{'...' if len(result['problem']) > 200 else ''}",
            f"\nSolution:",
            f"  {result['solution'][:300]}{'...' if len(result['solution']) > 300 else ''}",
            f"\nMetadata:",
            f"  Verified: {result['metadata']['verified']}",
            f"  Confidence: {result['metadata']['confidence']:.2%}",
            f"  Duration: {result['metadata']['duration_seconds']:.2f}s",
        ]
        
        if detailed:
            lines.append("\n" + self.visualize_trace(result['reasoning_trace']))
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def create_comparison_table(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str] = ["verified", "confidence", "duration_seconds"]
    ) -> str:
        """
        Create a comparison table for multiple results.
        
        Args:
            results: List of reasoning results
            metrics: Metrics to compare
            
        Returns:
            Formatted table
        """
        if not results:
            return "No results to compare"
        
        lines = ["\n" + "=" * 100, "RESULTS COMPARISON", "=" * 100]
        
        # Header
        header = f"{'ID':<5} | {'Problem':<40} | "
        for metric in metrics:
            header += f"{metric:<20} | "
        lines.append(header)
        lines.append("-" * 100)
        
        # Rows
        for i, result in enumerate(results, 1):
            problem = result["problem"][:37] + "..." if len(result["problem"]) > 40 else result["problem"]
            row = f"{i:<5} | {problem:<40} | "
            
            for metric in metrics:
                value = result["metadata"].get(metric, "N/A")
                if isinstance(value, float):
                    value_str = f"{value:.3f}"
                else:
                    value_str = str(value)
                row += f"{value_str:<20} | "
            
            lines.append(row)
        
        lines.append("=" * 100)
        return "\n".join(lines)
    
    def export_to_json(
        self,
        data: Any,
        file_path: str,
        pretty: bool = True
    ) -> None:
        """
        Export data to JSON file.
        
        Args:
            data: Data to export
            file_path: Output file path
            pretty: Whether to use pretty printing
        """
        with open(file_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
    
    def create_summary_report(
        self,
        results: List[Dict[str, Any]],
        evaluations: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Create a comprehensive summary report.
        
        Args:
            results: List of reasoning results
            evaluations: Optional evaluation results
            
        Returns:
            Formatted report
        """
        lines = [
            "\n" + "=" * 80,
            "REASONING AGENT SUMMARY REPORT",
            "=" * 80,
            f"\nTotal Problems Solved: {len(results)}",
            ""
        ]
        
        # Calculate statistics
        verified = sum(1 for r in results if r["metadata"]["verified"])
        avg_confidence = sum(r["metadata"]["confidence"] for r in results) / len(results)
        avg_duration = sum(r["metadata"]["duration_seconds"] for r in results) / len(results)
        
        lines.extend([
            "Performance Statistics:",
            "-" * 80,
            f"  Verified Solutions: {verified}/{len(results)} ({verified/len(results):.1%})",
            f"  Average Confidence: {avg_confidence:.2%}",
            f"  Average Duration: {avg_duration:.2f}s",
        ])
        
        # Problem type distribution
        problem_types = {}
        for result in results:
            ptype = result["reasoning_trace"]["understand"]["result"]["problem_type"]
            problem_types[ptype] = problem_types.get(ptype, 0) + 1
        
        lines.extend([
            "",
            "Problem Type Distribution:",
            "-" * 80
        ])
        for ptype, count in problem_types.items():
            lines.append(f"  {ptype}: {count} ({count/len(results):.1%})")
        
        # Evaluation metrics if provided
        if evaluations:
            lines.extend([
                "",
                "Evaluation Metrics:",
                "-" * 80
            ])
            
            metric_names = ["correctness", "completeness", "coherence", "efficiency", "clarity"]
            for metric in metric_names:
                values = [e["metrics"][metric] for e in evaluations if metric in e["metrics"]]
                if values:
                    avg = sum(values) / len(values)
                    lines.append(f"  {metric.capitalize()}: {avg:.3f}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
