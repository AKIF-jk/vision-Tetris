#!/usr/bin/env python3
# =============================================================================
# analyze_results.py — Analyze validation results and generate report
#
# Usage:
#   python analyze_results.py                    # Analyze validation_results.csv
#   python analyze_results.py --input results.csv  # Analyze custom results file
# =============================================================================

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def load_results(csv_file: str) -> list:
    """Load validation results from CSV."""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'filename': row['filename'],
                'ground_truth': row['ground_truth_class'],
                'detected': row['detected_class'],
                'match': row['match'].lower() == 'true',
                'xmin': int(row['xmin']),
                'ymin': int(row['ymin']),
                'xmax': int(row['xmax']),
                'ymax': int(row['ymax']),
            })
    return results


def analyze_results(results: list):
    """Analyze and print detailed statistics."""
    
    # Overall stats
    total = len(results)
    correct = sum(1 for r in results if r['match'])
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Per-class stats
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    confusion = defaultdict(lambda: defaultdict(int))
    misclassifications = defaultdict(lambda: defaultdict(int))
    
    for result in results:
        gt = result['ground_truth']
        detected = result['detected']
        
        class_stats[gt]['total'] += 1
        confusion[gt][detected] += 1
        
        if result['match']:
            class_stats[gt]['correct'] += 1
        else:
            misclassifications[gt][detected] += 1
    
    # Print report
    print(f"\n{'='*80}")
    print(f"  VALIDATION ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    print(f"  OVERALL PERFORMANCE")
    print(f"  {'-'*80}")
    print(f"  Total Images:        {total}")
    print(f"  Correct:             {correct}")
    print(f"  Incorrect:           {total - correct}")
    print(f"  Overall Accuracy:    {accuracy:.1f}%\n")
    
    # Per-class breakdown
    print(f"  PER-CLASS ACCURACY")
    print(f"  {'-'*80}")
    print(f"  {'Class':<8} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Error Rate':<12}")
    print(f"  {'-'*80}")
    
    for class_name in sorted(class_stats.keys()):
        stat = class_stats[class_name]
        acc = 100 * stat['correct'] / stat['total'] if stat['total'] > 0 else 0
        err = 100 - acc
        print(f"  {class_name:<8} {stat['correct']:<10} {stat['total']:<10} "
              f"{acc:>6.1f}%      {err:>6.1f}%")
    print()
    
    # Most common misclassifications
    print(f"  MOST COMMON MISCLASSIFICATIONS")
    print(f"  {'-'*80}")
    
    # Flatten misclassifications
    flat_misclass = []
    for gt, detected_dict in misclassifications.items():
        for detected, count in detected_dict.items():
            flat_misclass.append((count, gt, detected))
    
    flat_misclass.sort(reverse=True)
    
    if flat_misclass:
        print(f"  {'Ground Truth':<18} {'Detected As':<18} {'Count':<8} {'% of GT':<8}")
        print(f"  {'-'*80}")
        
        for count, gt, detected in flat_misclass[:15]:
            total_gt = class_stats[gt]['total']
            pct = 100 * count / total_gt if total_gt > 0 else 0
            print(f"  {gt:<18} {detected:<18} {count:<8} {pct:>5.1f}%")
    print()
    
    # Confusion matrix
    print(f"  CONFUSION MATRIX")
    print(f"  {'-'*80}")
    print(f"  Rows: Ground Truth  |  Columns: Detected As\n")
    
    classes = sorted(class_stats.keys())
    
    # Header
    print(f"  {'':>6} " + "  ".join(f"{c:>6}" for c in classes))
    print(f"  {'-'*80}")
    
    # Rows
    for gt in classes:
        row = f"  {gt:>6} "
        for detected in classes:
            count = confusion[gt][detected]
            row += f"{count:>6}  "
        print(row)
    print()
    
    # Class pairs that are frequently confused
    print(f"  FREQUENTLY CONFUSED CLASS PAIRS")
    print(f"  {'-'*80}")
    
    pair_confusion = defaultdict(int)
    for gt in confusion:
        for detected in confusion[gt]:
            if gt != detected:
                pair = tuple(sorted([gt, detected]))
                pair_confusion[pair] += confusion[gt][detected] + confusion[detected][gt]
    
    sorted_pairs = sorted(pair_confusion.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_pairs:
        print(f"  {'Classes':<15} {'Confusions':<15} {'% of Errors':<15}")
        print(f"  {'-'*80}")
        
        total_errors = total - correct
        for pair, count in sorted_pairs[:10]:
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"  {f'{pair[0]} ↔ {pair[1]}':<15} {count:<15} {pct:>6.1f}%")
    print()
    
    # Recommendations
    print(f"  RECOMMENDATIONS")
    print(f"  {'-'*80}")
    
    worst_class = min(class_stats.items(), 
                      key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 1)
    best_class = max(class_stats.items(), 
                     key=lambda x: x[1]['correct'] / x[1]['total'] if x[1]['total'] > 0 else 0)
    
    worst_name, worst_stat = worst_class
    best_name, best_stat = best_class
    worst_acc = 100 * worst_stat['correct'] / worst_stat['total'] if worst_stat['total'] > 0 else 0
    best_acc = 100 * best_stat['correct'] / best_stat['total'] if best_stat['total'] > 0 else 0
    
    print(f"\n  • Worst performing class: {worst_name} ({worst_acc:.1f}% accuracy)")
    print(f"    Focus on improving {worst_name} detection")
    
    if sorted_pairs:
        pair, count = sorted_pairs[0]
        print(f"\n  • Most common confusion: {pair[0]} ↔ {pair[1]} ({count} times)")
        print(f"    These classes are frequently misclassified as each other")
    
    if accuracy < 50:
        print(f"\n  • Overall accuracy is low ({accuracy:.1f}%)")
        print(f"    Consider reviewing:")
        print(f"    - Piece shape detection algorithm")
        print(f"    - Feature extraction (Hu moments, color)")
        print(f"    - Classification threshold tuning")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze validation results")
    parser.add_argument(
        '--input', '-i',
        default='validation_results.csv',
        help='Input CSV file with results (default: validation_results.csv)'
    )
    parser.add_argument(
        '--export', '-e',
        help='Export analysis to text file'
    )
    
    args = parser.parse_args()
    
    csv_file = Path(args.input)
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        return
    
    results = load_results(str(csv_file))
    
    if args.export:
        # Redirect output to file
        with open(args.export, 'w') as f:
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            analyze_results(results)
            sys.stdout = old_stdout
        print(f"Analysis exported to: {args.export}")
    else:
        analyze_results(results)


if __name__ == '__main__':
    main()
