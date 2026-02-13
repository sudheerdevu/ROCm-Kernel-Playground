#!/usr/bin/env python3
"""
ROCm Kernel Profiling Analyzer

Analyzes profiling results and generates comparison reports.
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class KernelStats:
    """Statistics for a single kernel"""
    name: str
    total_time_us: float
    call_count: int
    avg_time_us: float
    min_time_us: float
    max_time_us: float
    grid_size: tuple
    block_size: tuple


def parse_rocprof_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Parse rocprof CSV output"""
    results = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    return results


def analyze_kernels(results: List[Dict]) -> Dict[str, KernelStats]:
    """Analyze kernel execution data"""
    kernel_data: Dict[str, List[Dict]] = {}
    
    for row in results:
        name = row.get('Name') or row.get('KernelName', 'Unknown')
        if name not in kernel_data:
            kernel_data[name] = []
        kernel_data[name].append(row)
    
    stats = {}
    for name, executions in kernel_data.items():
        times = []
        grid = (1, 1, 1)
        block = (1, 1, 1)
        
        for exec_data in executions:
            # Parse duration
            duration = float(exec_data.get('DurationNs', 0) or 
                           exec_data.get('Duration', 0))
            times.append(duration / 1000)  # Convert to microseconds
            
            # Parse grid/block
            grid = (
                int(exec_data.get('grd0', 1) or 1),
                int(exec_data.get('grd1', 1) or 1),
                int(exec_data.get('grd2', 1) or 1),
            )
            block = (
                int(exec_data.get('wgr0', 1) or 1),
                int(exec_data.get('wgr1', 1) or 1),
                int(exec_data.get('wgr2', 1) or 1),
            )
        
        if times:
            stats[name] = KernelStats(
                name=name,
                total_time_us=sum(times),
                call_count=len(times),
                avg_time_us=sum(times) / len(times),
                min_time_us=min(times),
                max_time_us=max(times),
                grid_size=grid,
                block_size=block,
            )
    
    return stats


def generate_report(stats: Dict[str, KernelStats], format: str = 'text') -> str:
    """Generate analysis report"""
    if format == 'json':
        data = {
            name: {
                'total_time_us': s.total_time_us,
                'call_count': s.call_count,
                'avg_time_us': s.avg_time_us,
                'min_time_us': s.min_time_us,
                'max_time_us': s.max_time_us,
            }
            for name, s in stats.items()
        }
        return json.dumps(data, indent=2)
    
    # Text format
    lines = [
        "=" * 70,
        "Kernel Analysis Report",
        "=" * 70,
        "",
    ]
    
    # Sort by total time
    sorted_kernels = sorted(stats.values(), key=lambda s: s.total_time_us, reverse=True)
    total_time = sum(s.total_time_us for s in sorted_kernels)
    
    lines.append(f"Total GPU Time: {total_time:.2f} µs ({total_time/1000:.2f} ms)")
    lines.append(f"Total Kernels: {len(sorted_kernels)}")
    lines.append(f"Total Calls: {sum(s.call_count for s in sorted_kernels)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"{'Kernel':<35} {'Calls':>6} {'Total (µs)':>12} {'Avg (µs)':>10} {'%':>6}")
    lines.append("-" * 70)
    
    for s in sorted_kernels[:20]:  # Top 20
        pct = (s.total_time_us / total_time * 100) if total_time > 0 else 0
        lines.append(
            f"{s.name[:35]:<35} {s.call_count:>6} {s.total_time_us:>12.2f} "
            f"{s.avg_time_us:>10.2f} {pct:>5.1f}%"
        )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze ROCm profiling results')
    parser.add_argument('input', type=Path, help='Input CSV file or directory')
    parser.add_argument('-o', '--output', type=Path, help='Output file')
    parser.add_argument('-f', '--format', choices=['text', 'json', 'markdown'],
                        default='text', help='Output format')
    args = parser.parse_args()
    
    # Find CSV files
    if args.input.is_dir():
        csv_files = list(args.input.glob('**/*.csv'))
    else:
        csv_files = [args.input]
    
    # Combine results
    all_results = []
    for csv_file in csv_files:
        results = parse_rocprof_csv(csv_file)
        all_results.extend(results)
    
    if not all_results:
        print("No profiling data found")
        return
    
    # Analyze
    stats = analyze_kernels(all_results)
    report = generate_report(stats, args.format)
    
    # Output
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
