import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import logging
from datetime import datetime
import wandb
from sklearn.metrics import accuracy_score
import re
import gc
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TunixTrainingConfig:
    """Configuration for Tunix reasoning agent training"""
    model_name: str = "microsoft/DialoGPT-medium"  # Base model
    max_length: int = 1024
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    output_dir: str = "./tunix_model_checkpoints"
    target_accuracy: float = 0.87
    use_wandb: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4

class TunixReasoningDataset(Dataset):
    """Dataset class for Tunix reasoning problems"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training examples")
        
        # Process data into training format
        self.processed_data = self._process_data()
    
    def _process_data(self) -> List[Dict]:
        """Process raw data into training format"""
        processed = []
        
        for item in self.data:
            # Format: Problem -> Step-by-step reasoning -> Final answer
            problem = item['problem']
            solution = item['solution']
            
            # Create training text with special tokens
            training_text = f"<|problem|>{problem}<|reasoning|>{solution}<|endoftext|>"
            
            processed.append({
                'text': training_text,
                'problem': problem,
                'solution': solution
            })
        
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten().clone()
        }

class TunixReasoningEvaluator:
    """Evaluator for Tunix reasoning capabilities"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_reasoning(self, test_problems: List[Dict]) -> Dict[str, float]:
        """Evaluate model on reasoning problems"""
        self.model.eval()
        
        correct_answers = 0
        total_problems = len(test_problems)
        step_accuracies = []
        
        with torch.no_grad():
            for problem_data in test_problems:
                problem = problem_data['problem']
                expected_solution = problem_data['solution']
                
                # Generate solution
                generated_solution = self._generate_solution(problem)
                
                # Evaluate accuracy
                is_correct = self._evaluate_solution_accuracy(
                    generated_solution, expected_solution
                )
                
                if is_correct:
                    correct_answers += 1
                
                # Evaluate step-by-step reasoning quality
                step_accuracy = self._evaluate_step_accuracy(
                    generated_solution, expected_solution
                )
                step_accuracies.append(step_accuracy)
        
        accuracy = correct_answers / total_problems
        avg_step_accuracy = np.mean(step_accuracies)
        
        return {
            'accuracy': accuracy,
            'step_accuracy': avg_step_accuracy,
            'correct_answers': correct_answers,
            'total_problems': total_problems
        }
    
    def _generate_solution(self, problem: str, max_new_tokens: int = 512) -> str:
        """Generate solution for a given problem"""
        prompt = f"<|problem|>{problem}<|reasoning|>"
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract reasoning part
        if "<|reasoning|>" in generated_text:
            reasoning = generated_text.split("<|reasoning|>")[1]
            if "<|endoftext|>" in reasoning:
                reasoning = reasoning.split("<|endoftext|>")[0]
            return reasoning.strip()
        
        return ""
    
    def _evaluate_solution_accuracy(self, generated: str, expected: str) -> bool:
        """Evaluate if generated solution is correct"""
        # Extract final numerical answer from both solutions
        gen_answer = self._extract_final_answer(generated)
        exp_answer = self._extract_final_answer(expected)
        
        if gen_answer is not None and exp_answer is not None:
            return abs(gen_answer - exp_answer) < 0.01
        
        # Fallback to string similarity for non-numerical answers
        return self._calculate_similarity(generated.lower(), expected.lower()) > 0.8
    
    def _extract_final_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from solution text"""
        # Look for patterns like "= 42", "answer: 42", "result is 42"
        patterns = [
            r'=\s*([+-]?\d*\.?\d+)',
            r'answer:?\s*([+-]?\d*\.?\d+)',
            r'result:?\s*([+-]?\d*\.?\d+)',
            r'solution:?\s*([+-]?\d*\.?\d+)',
            r'final answer:?\s*([+-]?\d*\.?\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return float(matches[-1])  # Take the last match
                except ValueError:
                    continue
        
        return None
    
    def _evaluate_step_accuracy(self, generated: str, expected: str) -> float:
        """Evaluate step-by-step reasoning accuracy"""
        # Simple heuristic: count similar reasoning steps
        gen_steps = self._extract_reasoning_steps(generated)
        exp_steps = self._extract_reasoning_steps(expected)
        
        if not exp_steps:
            return 1.0 if not gen_steps else 0.5
        
        matching_steps = 0
        for exp_step in exp_steps:
            for gen_step in gen_steps:
                if self._calculate_similarity(exp_step, gen_step) > 0.6:
                    matching_steps += 1
                    break
        
        return matching_steps / len(exp_steps)
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps from solution"""
        # Split by common step indicators
        steps = re.split(r'(?:step \d+|first|second|third|then|next|finally|therefore)', text.lower())
        return [step.strip() for step in steps if step.strip()]
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class TunixTrainer:
    """Main trainer class for Tunix reasoning agent"""
    
    def __init__(self, config: TunixTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project="tunix-reasoning-agent",
                config=config.__dict__,
                name=f"tunix_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Setup model and tokenizer
        self._setup_model_and_tokenizer()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info(f"Training setup complete. Using device: {self.device}")
    
    def _setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with optimizations"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add special tokens for Tunix format
        special_tokens = {
            "additional_special_tokens": ["<|problem|>", "<|reasoning|>", "<|endoftext|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        # Resize token embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
    
    def create_sample_training_data(self):
        """Create sample training data in Tunix format"""
        sample_data = [
            {
                "problem": "A rectangle has length 12 cm and width 8 cm. What is its area?",
                "solution": "Step 1: Identify the formula for rectangle area. Area = length × width. Step 2: Substitute the given values. Area = 12 cm × 8 cm. Step 3: Calculate the result. Area = 96 cm². Therefore, the area is 96 square centimeters."
            },
            {
                "problem": "If x + 5 = 12, what is the value of x?",
                "solution": "Step 1: Start with the equation x + 5 = 12. Step 2: Subtract 5 from both sides to isolate x. x + 5 - 5 = 12 - 5. Step 3: Simplify both sides. x = 7. Therefore, x = 7."
            },
            {
                "problem": "A train travels 240 km in 3 hours. What is its average speed?",
                "solution": "Step 1: Identify the formula for average speed. Speed = Distance ÷ Time. Step 2: Substitute the given values. Speed = 240 km ÷ 3 hours. Step 3: Calculate the result. Speed = 80 km/h. Therefore, the average speed is 80 kilometers per hour."
            },
            {
                "problem": "What is 15% of 200?",
                "solution": "Step 1: Convert percentage to decimal. 15% = 15/100 = 0.15. Step 2: Multiply the decimal by the number. 0.15 × 200. Step 3: Calculate the result. 0.15 × 200 = 30. Therefore, 15% of 200 is 30."
            },
            {
