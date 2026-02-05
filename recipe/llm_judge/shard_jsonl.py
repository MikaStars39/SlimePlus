#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import subprocess

def writer_worker(shard_path, queue):
    """Dedicated writer process for one shard"""
    with open(shard_path, 'w', encoding='utf-8') as f:
        while True:
            line = queue.get()
            if line is None:
                break
            f.write(line)

def reader_worker(input_file, start_line, end_line, queues, num_shards):
    """Reader process that handles a chunk of the file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip to start_line
        for _ in range(start_line):
            f.readline()
        
        # Process lines in this chunk
        for line_num in range(start_line, end_line):
            line = f.readline()
            if not line or not line.strip():
                continue
            
            # Fast extraction of original_idx
            try:
                start = line.find('"original_idx":')
                if start != -1:
                    end = line.find(',', start)
                    if end == -1: 
                        end = line.find('}', start)
                    val_str = line[start+15:end].strip().strip(':').strip()
                    original_idx = int(val_str)
                else:
                    data = json.loads(line)
                    original_idx = data.get('original_idx', 0)
            except:
                data = json.loads(line)
                original_idx = data.get('original_idx', 0)

            shard_idx = original_idx % num_shards
            queues[shard_idx].put(line)

def shard_jsonl(input_file, output_dir, num_shards, num_readers=None):
    """
    Multi-process sharding with parallel readers and writers.
    Ensures all 4 answers for each question stay together in the same shard.
    """
    if num_readers is None:
        num_readers = min(cpu_count(), 16)  # Use up to 16 reader processes
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sharding with {num_readers} reader workers and {num_shards} writer workers...")
    
    # Count total lines
    print("Counting lines...")
    try:
        total_lines = int(subprocess.check_output(['wc', '-l', input_file]).split()[0])
    except:
        with open(input_file, 'rb') as f:
            total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines}")
    
    # Create queues for each shard
    queues = [Queue(maxsize=2000) for _ in range(num_shards)]
    
    # Start writer processes
    writer_processes = []
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i}.jsonl"
        p = Process(target=writer_worker, args=(shard_path, queues[i]))
        p.start()
        writer_processes.append(p)
    
    # Start reader processes
    lines_per_reader = total_lines // num_readers
    reader_processes = []
    
    for i in range(num_readers):
        start_line = i * lines_per_reader
        end_line = (i + 1) * lines_per_reader if i < num_readers - 1 else total_lines
        
        p = Process(target=reader_worker, args=(input_file, start_line, end_line, queues, num_shards))
        p.start()
        reader_processes.append(p)
        print(f"Reader {i}: processing lines {start_line} to {end_line}")
    
    # Wait for all readers to finish
    for p in reader_processes:
        p.join()
    
    print("All readers finished, flushing writers...")
    
    # Send termination signal to all writers
    for q in queues:
        q.put(None)
    
    # Wait for all writers to finish
    for p in writer_processes:
        p.join()
    
    print(f"✅ Successfully sharded {total_lines} lines into {num_shards} files in {output_dir}")
    
    # Print shard statistics
    print("\nShard statistics:")
    for i in range(num_shards):
        shard_path = output_dir / f"shard_{i}.jsonl"
        try:
            lines = int(subprocess.check_output(['wc', '-l', str(shard_path)]).split()[0])
            print(f"  shard_{i}.jsonl: {lines} lines")
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast parallel sharding of prepared JSONL")
    parser.add_argument("--input", required=True, help="Path to judge_prepared.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to save shards")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards (usually number of nodes)")
    parser.add_argument("--num-readers", type=int, default=None, help="Number of reader processes (default: min(cpu_count, 16))")
    
    args = parser.parse_args()
    shard_jsonl(args.input, args.output_dir, args.num_shards, args.num_readers)
