"""
Result processing and aggregation utilities for evaluation pipeline.
Provides functions to process, aggregate, and summarize evaluation results.
"""
import os
from utils.files import load_json
import logging

logger = logging.getLogger(__name__)

def match_label_to_score(label):
    """
    Convert a match label to a numeric score.
    """
    if label == "Not Covered":
        return 0
    elif label == "Partially Covered":
        return 0.5
    elif label == "Fully Covered":
        return 1
    else:
        return 0

def process_inference_result(inference_result):
    """
    Process inference results to compute matched counts for precision/recall/F1.
    """
    n_true = len(inference_result['groundtruth']['checklist'])
    n_predicted = len(inference_result['inferred']['checklist'])
    true_matched = 0
    predicted_matched = 0
    for result in inference_result['match']['gt_to_infer']:
        true_matched += match_label_to_score(result['label'])
    for result in inference_result['match']['infer_to_gt']:
        predicted_matched += match_label_to_score(result['label'])
    return {
        "true_matched": true_matched,
        "predicted_matched": predicted_matched,
        "n_true": n_true,
        "n_predicted": n_predicted
    }

def process_generation_result(generation_result):
    """
    Process generation results to extract and normalize the alignment score.
    """
    score = generation_result['alignment']['score']
    if isinstance(score, str):
        score = score.strip()
        if "/" in score:
            score = score.split("/")[0]
        if "." in score:
            score = score.split(".")[0]
        if "\n\n" in score:
            score = score.split("\n\n")[0]
        if "**" in score:
            score = score.replace("**", "")
        try:
            score = float(score)
        except Exception as e:
            logger.error(f"Error parsing generation score: {e}\nScore value: {score}")
            score = 0
    return {
        "score": score
    }

def process_results(results_dir, model_name, task):
    """
    Process all result files for a given model and return a dictionary of results.
    """
    results_files = os.listdir(f"{results_dir}/{model_name}")
    results = {}
    for result_file in results_files:
        if result_file == "results.json":
            continue
        instance_name = result_file.split(".")[0]
        try:
            result = load_json(f"{results_dir}/{model_name}/{result_file}")
        except Exception as e:
            logger.error(f"Error loading result file {result_file}: {e}")
            continue
        results[instance_name] = {}
        if task in ["inference", "both"]:
            results[instance_name]['inference'] = process_inference_result(result['inference'])
        if task in ["generation", "both"]:
            results[instance_name]['generation'] = process_generation_result(result['generation'])
    return results

def calculate_prf(n_true, n_predicted, true_matched, predicted_matched):
    """
    Calculate precision, recall, and F1 score from matched counts.
    """
    precision = predicted_matched / n_predicted if n_predicted > 0 else 0
    recall = true_matched / n_true if n_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def aggregate_results(results_dir, model_name, task):
    """
    Aggregate all processed results for a model and return summary statistics.
    """
    results = process_results(results_dir, model_name, task)
    aggregated_results = {
        "inference": {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "n_true_matched": 0,
            "n_predicted_matched": 0,
            "n_true": 0,
            "n_predicted": 0
        },
        "generation": {
            "average_score": 0,
            "n_instances": 0,
            "total_score": 0
        }
    }
    for instance_name, result in results.items():
        if task in ["inference", "both"]:
            aggregated_results['inference']['n_true_matched'] += result['inference']['true_matched']
            aggregated_results['inference']['n_predicted_matched'] += result['inference']['predicted_matched']
            aggregated_results['inference']['n_true'] += result['inference']['n_true']
            aggregated_results['inference']['n_predicted'] += result['inference']['n_predicted']
        if task in ["generation", "both"]:
            aggregated_results['generation']['total_score'] += result['generation']['score']
            aggregated_results['generation']['n_instances'] += 1
    if task in ["inference", "both"]:
        inference_prf = calculate_prf(
            aggregated_results['inference']['n_true'],
            aggregated_results['inference']['n_predicted'],
            aggregated_results['inference']['n_true_matched'],
            aggregated_results['inference']['n_predicted_matched']
        )
        aggregated_results['inference']['precision'] = inference_prf['precision']
        aggregated_results['inference']['recall'] = inference_prf['recall']
        aggregated_results['inference']['f1'] = inference_prf['f1']

    if aggregated_results['generation']['n_instances'] > 0:
        aggregated_results['generation']['average_score'] = aggregated_results['generation']['total_score'] / aggregated_results['generation']['n_instances']
    del aggregated_results['generation']['total_score']

    if task == "generation":
        del aggregated_results['inference']
    elif task == "inference":
        del aggregated_results['generation']
    return aggregated_results