#!/usr/bin/env python3
"""
Complete LLM Scoring Pipeline
Reads JSONL files containing multiple model outputs and uses an LLM to score each output multi-dimensionally.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class JudgePipelineRunner:
    def __init__(self, input_file, output_dir, judge_model):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.judge_model = judge_model
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all intermediate file paths
        self.files = {
            'input': input_file,
            'prepared': self.output_dir / 'judge_prepared.jsonl',
            'inference': self.output_dir / 'judge_inference.jsonl',
            'scores': self.output_dir / 'judge_scores.jsonl',
            'failed': self.output_dir / 'judge_failed.jsonl',
        }
        
        # Script directory
        self.script_dir = Path(__file__).parent
    
    def run_command(self, cmd, step_name):
        """Run command and handle errors"""
        print("\n" + "=" * 80)
        print(f"STEP: {step_name}")
        print("=" * 80)
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR: {step_name} failed with exit code {result.returncode}")
            sys.exit(1)
        
        print(f"\n✅ {step_name} completed")
        return result
    
    def step1_prepare(self):
        """Step 1: Prepare evaluation data"""
        cmd = [
            'python', str(self.script_dir / 'prepare_judge.py'),
            '--input', str(self.files['input']),
            '--output', str(self.files['prepared']),
            '--tokenizer', self.judge_model
        ]
        self.run_command(cmd, "Step 1: Prepare evaluation data")
    
    def step2_inference(self):
        """Step 2: Run LLM scoring"""
        cmd = [
            'python', str(self.script_dir / 'inference.py'),
            '--input', str(self.files['prepared']),
            '--output', str(self.files['inference']),
            '--model_path', self.judge_model,
            '--tp_size', '2',
            '--dp_size', '4',
            '--max_concurrency', '512',
            '--max_tokens', '2048',
            '--temp', '0.3',  # Use lower temperature for more stable scoring
        ]
        self.run_command(cmd, "Step 2: Run LLM scoring")
    
    def step3_extract(self):
        """Step 3: Extract and aggregate scoring results"""
        cmd = [
            'python', str(self.script_dir / 'extract_scores.py'),
            '--prepared', str(self.files['prepared']),
            '--response', str(self.files['inference']),
            '--output', str(self.files['scores']),
            '--failed', str(self.files['failed'])
        ]
        self.run_command(cmd, "Step 3: Extract scoring results")
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "🚀" * 40)
        print("Starting LLM Scoring Pipeline")
        print("🚀" * 40)
        print(f"\nInput file:         {self.files['input']}")
        print(f"Output directory:   {self.output_dir}")
        print(f"Scoring model:      {self.judge_model}")
        print(f"\nFinal outputs:")
        print(f"  - Scoring results: {self.files['scores']}")
        print(f"  - Failed records:  {self.files['failed']}")
        
        # Check input file
        if not os.path.exists(self.files['input']):
            print(f"\n❌ ERROR: Input file not found: {self.files['input']}")
            sys.exit(1)
        
        # Run all steps
        try:
            self.step1_prepare()
            self.step2_inference()
            self.step3_extract()
            
            print("\n" + "🎉" * 40)
            print("Scoring pipeline completed!")
            print("🎉" * 40)
            print(f"\nFinal outputs:")
            print(f"  ✅ Scoring results: {self.files['scores']}")
            print(f"  ⚠️  Failed records:  {self.files['failed']}")
            print(f"\nAll intermediate files saved in: {self.output_dir}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete LLM scoring pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_judge_pipeline.py \\
    --input /path/to/input.jsonl \\
    --output-dir /path/to/output \\
    --judge-model /path/to/judge/model

Input JSONL format requirements:
  Each line contains:
  - conversations: Question and reference answer
  - JoyAI_output_0 to JoyAI_output_7: 8 model outputs

Output format:
  Each line contains:
  - original_idx: Original data index
  - question: Question
  - reference_answer: Reference answer
  - scores: Scoring details for each output
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSONL file (containing conversations and JoyAI_output_0-7)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--judge-model',
        default="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Thinking-2507-FP8",
        help='Scoring model path'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    runner = JudgePipelineRunner(
        input_file=args.input,
        output_dir=args.output_dir,
        judge_model=args.judge_model
    )
    
    runner.run()


if __name__ == '__main__':
    main()
