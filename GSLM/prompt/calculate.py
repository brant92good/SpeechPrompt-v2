import json
import numpy as np
import argparse
from pathlib import Path
import utils
from utils import compute_eer

storage_dir = utils.get_storage_dir()

def scan_labels(json_file_path):
    """
    Scan the JSON file to automatically detect the two class labels.
    Returns the two unique labels found in the dataset.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    labels = set()
    for key, sample in data.items():
        labels.add(sample['label'])
    
    unique_labels = list(labels)
    
    print("=" * 60)
    print("LABEL SCANNING RESULTS")
    print("=" * 60)
    print(f"Unique labels found: {unique_labels}")
    print(f"Number of unique labels: {len(unique_labels)}")
    
    if len(unique_labels) != 2:
        print(f"Warning: Expected 2 classes for EER calculation, but found {len(unique_labels)} classes")
        if len(unique_labels) < 2:
            raise ValueError("Need at least 2 classes for EER calculation")
    
    # For EER calculation, we'll treat the first label as "class_0" and second as "class_1"
    class_0, class_1 = unique_labels[0], unique_labels[1]
    print(f"Will use '{class_0}' as class_0 and '{class_1}' as class_1")
    print("Note: For EER calculation, the assignment doesn't matter as EER is symmetric")
    print("=" * 60)
    
    return class_0, class_1


def process_json_confidence(json_file_path, class_0=None, class_1=None):
    """
    Process JSON with correct confidence interpretation for EER.
    Converts all confidences to represent class_0 probability.
    """
    # Auto-detect labels if not provided
    if class_0 is None or class_1 is None:
        class_0, class_1 = scan_labels(json_file_path)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    target_scores = []      # class_0 confidence for true class_0 samples
    nontarget_scores = []   # class_0 confidence for true class_1 samples
    
    for key, sample in data.items():
        label = sample['label']
        predict = sample['predict']
        confidence = sample['confidence']
        
        # Convert confidence to "class_0 probability"
        if predict == class_0:
            class_0_prob = confidence
        elif predict == class_1:
            class_0_prob = 1 - confidence  # If model says 80% class_1, then 20% class_0
        else:
            # Model predicted something outside our two classes
            continue
        
        # Collect scores based on TRUE labels
        if label == class_0:
            target_scores.append(class_0_prob)
        elif label == class_1:
            nontarget_scores.append(class_0_prob)
    
    return np.array(target_scores), np.array(nontarget_scores), class_0, class_1



def get_json_file_path(args):
    """
    Generate the JSON file path based on the same argument logic as sample.py
    """
    json_file_path = storage_dir / "exp_results" / args.downstream / args.exp_name / "samples" / "samples.json"
    return json_file_path


def get_input_args():
    """
    Parse input arguments using the same logic as sample.py
    """
    parser = argparse.ArgumentParser(description="Calculate EER from sampling results")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name")
    parser.add_argument("--downstream", type=str, default="FSD_asvspoof",
                        help="Downstream task name")
    parser.add_argument("--json_file", type=str, default=None,
                        help="Custom JSON file path (overrides auto-generated path)")
    
    return parser.parse_args()


def print_eer_results(json_file_path):
    """
    Print EER results with auto-detected class labels and store them alongside the JSON file.
    """
    target_scores, nontarget_scores, class_0, class_1 = process_json_confidence(json_file_path)

    if len(target_scores) > 0 and len(nontarget_scores) > 0:
        eer, threshold = compute_eer(target_scores, nontarget_scores)
    else:
        eer, threshold = None, None

    json_path_obj = Path(json_file_path)
    output_lines = []

    def add_line(line=""):
        output_lines.append(line)
        print(line)

    add_line("EER COMPUTATION RESULTS")
    add_line("=" * 60)

    if eer is not None:
        add_line(f"Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
        add_line(f"EER Threshold: {threshold:.6f}")
        add_line(f"\nAt EER threshold {threshold:.6f}:")
        add_line(f"- Samples with {class_0}_prob ≥ {threshold:.6f} are classified as '{class_0}'")
        add_line(f"- Samples with {class_0}_prob < {threshold:.6f} are classified as '{class_1}'")
    else:
        add_line("Could not compute EER - insufficient data")

    add_line(f"\nDataset Statistics:")
    add_line(f"True '{class_0}' samples: {len(target_scores)}")
    add_line(f"True '{class_1}' samples: {len(nontarget_scores)}")

    if len(target_scores) > 0:
        add_line(f"'{class_0}' samples - {class_0}_prob stats:")
        add_line(f"  Mean: {np.mean(target_scores):.4f}")
        add_line(f"  Min: {np.min(target_scores):.4f}")
        add_line(f"  Max: {np.max(target_scores):.4f}")

    if len(nontarget_scores) > 0:
        add_line(f"'{class_1}' samples - {class_0}_prob stats:")
        add_line(f"  Mean: {np.mean(nontarget_scores):.4f}")
        add_line(f"  Min: {np.min(nontarget_scores):.4f}")
        add_line(f"  Max: {np.max(nontarget_scores):.4f}")

    output_path = json_path_obj.parent / "eer_results.txt"
    output_path.write_text("\n".join(output_lines) + "\n")
    print(f"\nResults saved to {output_path}")

# Usage example:
if __name__ == "__main__":
    args = get_input_args()
    
    # Use custom JSON file if provided, otherwise generate from args
    if args.json_file:
        json_file_path = Path(args.json_file)
    else:
        json_file_path = get_json_file_path(args)
    
    print(f"Processing JSON file: {json_file_path}")
    
    # Check if file exists
    if not json_file_path.exists():
        print(f"Error: JSON file not found at {json_file_path}")
        print(f"Make sure you have run sample.py with the same arguments first:")
        print(f"python sample.py --exp_name {args.exp_name} --downstream {args.downstream}")
        exit(1)
    
    # Print EER results with auto-detected labels
    print_eer_results(str(json_file_path))