#!/usr/bin/env python3
"""
Complete pipeline wrapper for instruct_llm_do_first data processing.
Runs all steps from input to final easy/hard split in one command.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


class PipelineRunner:
    def __init__(self, input_file, output_dir, k=4, threshold=0.51):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.k = k
        self.threshold = threshold
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all intermediate file paths
        self.files = {
            'input': input_file,
            'instruct_filter_pre': self.output_dir / 'raw_instruct_filter_pre.jsonl',
            'instruct_filter_infer': self.output_dir / 'raw_instruct_filter_infer.jsonl',
            'instruct_filter_judge_pre': self.output_dir / 'raw_instruct_filter_judge_pre.jsonl',
            'instruct_filter_judge_infer': self.output_dir / 'raw_instruct_filter_judge_infer.jsonl',
            'instruct_filter_judge_post': self.output_dir / 'raw_instruct_filter_judge_post.jsonl',
            'instruct_filter_judge_failed': self.output_dir / 'raw_instruct_filter_judge_failed.jsonl',
            'instruct_filter_pass_at_k': self.output_dir / 'raw_instruct_filter_pass_at_k.jsonl',
            'output_easy': self.output_dir / 'raw_instruct_filter_4b_easy.jsonl',
            'output_hard': self.output_dir / 'raw_instruct_filter_4b_hard.jsonl',
        }
        
        # Model and tokenizer paths (can be overridden via arguments)
        self.qwen_4b = "/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507"
        self.qwen_30b = "/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507"
        
        # Get script directory
        self.script_dir = Path(__file__).parent
    
    def run_command(self, cmd, step_name):
        """Run a command and handle errors."""
        print("\n" + "=" * 80)
        print(f"STEP: {step_name}")
        print("=" * 80)
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR: {step_name} failed with exit code {result.returncode}")
            sys.exit(1)
        
        print(f"\n✅ {step_name} completed successfully")
        return result
    
    def step1_preprocess(self):
        """Step 1: Preprocess data and expand to k rollouts."""
        cmd = [
            'python', str(self.script_dir / 'preprocess.py'),
            '--input', str(self.files['input']),
            '--output', str(self.files['instruct_filter_pre']),
            '--k', str(self.k),
            '--tokenizer', self.qwen_30b
        ]
        self.run_command(cmd, "Step 1: Preprocess (expand to k rollouts)")
    
    def step2_inference_4b(self):
        """Step 2: Run 4B model inference."""
        cmd = [
            'python', str(self.script_dir / 'inference.py'),
            '--input', str(self.files['instruct_filter_pre']),
            '--output', str(self.files['instruct_filter_infer']),
            '--model_path', self.qwen_4b,
            '--tp_size', '1',
            '--dp_size', '8',
            '--max_concurrency', '1024',
            '--max_tokens', '2048'
        ]
        self.run_command(cmd, "Step 2: 4B Model Inference")
    
    def step3_judge_prepare(self):
        """Step 3: Prepare data for judge model."""
        cmd = [
            'python', str(self.script_dir / 'llm_judge_prepare.py'),
            '--input', str(self.files['instruct_filter_infer']),
            '--output', str(self.files['instruct_filter_judge_pre']),
            '--tokenizer', self.qwen_30b
        ]
        self.run_command(cmd, "Step 3: Prepare for Judge Model")
    
    def step4_inference_judge(self):
        """Step 4: Run 30B judge model inference."""
        cmd = [
            'python', str(self.script_dir / 'inference.py'),
            '--input', str(self.files['instruct_filter_judge_pre']),
            '--output', str(self.files['instruct_filter_judge_infer']),
            '--model_path', self.qwen_30b,
            '--tp_size', '1',
            '--dp_size', '8',
            '--max_concurrency', '1024',
            '--max_tokens', '8192'
        ]
        self.run_command(cmd, "Step 4: 30B Judge Model Inference")
    
    def step5_judge_extract(self):
        """Step 5: Extract judge results."""
        cmd = [
            'python', str(self.script_dir / 'llm_judge_extract.py'),
            '--original', str(self.files['instruct_filter_pre']),
            '--response', str(self.files['instruct_filter_judge_infer']),
            '--output', str(self.files['instruct_filter_judge_post']),
            '--failed', str(self.files['instruct_filter_judge_failed'])
        ]
        self.run_command(cmd, "Step 5: Extract Judge Results")
    
    def step6_calculate_pass_at_k(self):
        """Step 6: Calculate pass@k statistics."""
        cmd = [
            'python', str(self.script_dir / 'calculate_pass_at_k.py'),
            '--rollout', str(self.files['instruct_filter_judge_post']),
            '--reference', str(self.files['input']),
            '--output', str(self.files['instruct_filter_pass_at_k'])
        ]
        self.run_command(cmd, "Step 6: Calculate Pass@K")
    
    def step7_filter_by_threshold(self):
        """Step 7: Filter by threshold into easy/hard splits."""
        cmd = [
            'python', str(self.script_dir / 'filter_by_pass_at_k.py'),
            '--input', str(self.files['instruct_filter_pass_at_k']),
            '--threshold', str(self.threshold),
            '--output-high', str(self.files['output_easy']),
            '--output-low', str(self.files['output_hard'])
        ]
        self.run_command(cmd, f"Step 7: Filter by Threshold ({self.threshold})")
    
    def run(self):
        """Run the complete pipeline."""
        print("\n" + "🚀" * 40)
        print("STARTING COMPLETE PIPELINE")
        print("🚀" * 40)
        print(f"\nInput file:     {self.files['input']}")
        print(f"Output dir:     {self.output_dir}")
        print(f"K rollouts:     {self.k}")
        print(f"Threshold:      {self.threshold}")
        print(f"\nFinal outputs:")
        print(f"  - Easy (≥{self.threshold}): {self.files['output_easy']}")
        print(f"  - Hard (<{self.threshold}): {self.files['output_hard']}")
        
        # Check if input file exists
        if not os.path.exists(self.files['input']):
            print(f"\n❌ ERROR: Input file not found: {self.files['input']}")
            sys.exit(1)
        
        # Run all steps
        try:
            self.step1_preprocess()
            self.step2_inference_4b()
            self.step3_judge_prepare()
            self.step4_inference_judge()
            self.step5_judge_extract()
            self.step6_calculate_pass_at_k()
            self.step7_filter_by_threshold()
            
            print("\n" + "🎉" * 40)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 40)
            print(f"\nFinal outputs:")
            print(f"  ✅ Easy: {self.files['output_easy']}")
            print(f"  ✅ Hard: {self.files['output_hard']}")
            print(f"\nIntermediate files saved in: {self.output_dir}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n\n❌ Pipeline failed with error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete instruct_llm_do_first pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_pipeline.py \\
    --input /path/to/input.jsonl \\
    --output-dir /path/to/output \\
    --k 4 \\
    --threshold 0.51

This will:
  1. Expand input to k rollouts
  2. Run 4B model inference
  3. Prepare for judge model
  4. Run 30B judge model
  5. Extract judge results
  6. Calculate pass@k statistics
  7. Split into easy/hard based on threshold
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input JSONL file (e.g., raw_relabel_post.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for all generated files'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        help='Number of rollouts to generate (default: 4)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.51,
        help='Pass rate threshold for easy/hard split (default: 0.51)'
    )
    parser.add_argument(
        '--qwen-4b',
        default="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B-Instruct-2507",
        help='Path to Qwen 4B model'
    )
    parser.add_argument(
        '--qwen-30b',
        default="/mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-30B-A3B-Instruct-2507",
        help='Path to Qwen 30B model'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Threshold must be between 0.0 and 1.0")
    
    # Create pipeline runner
    runner = PipelineRunner(
        input_file=args.input,
        output_dir=args.output_dir,
        k=args.k,
        threshold=args.threshold
    )
    
    # Override model paths if provided
    if args.qwen_4b:
        runner.qwen_4b = args.qwen_4b
    if args.qwen_30b:
        runner.qwen_30b = args.qwen_30b
    
    # Run the pipeline
    runner.run()


if __name__ == '__main__':
    main()
