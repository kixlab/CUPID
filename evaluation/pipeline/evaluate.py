"""
Evaluation pipeline for running model inference, preference matching, response generation, and judging.
Provides evaluate and evaluate_parallel functions for single and multi-process evaluation.
"""
import os
import json
import logging
from multiprocessing import Pool
from datasets import load_dataset
from utils.files import load_json, save_json, ensure_directory
from utils.logging import setup_worker_logging
from config import DATASET_NAME, PREFMATCHER_MODEL_NAME
from evaluation.models.model import get_model_class
from evaluation.modules import PreferenceInferrer, PreferenceMatcher, ResponseGenerator, ResponseJudger

logger = logging.getLogger(__name__)

log_queue = None  # Global for worker processes

def worker_init(queue):
    global log_queue
    log_queue = queue
    setup_worker_logging(log_queue)

def evaluate(model_name, evaluator_model, instance_name, instance_data, results_dir, use_matcher=False, task="inference"):
    """
    Run the evaluation pipeline for a single data instance.
    Steps: inference, matching, response generation, and judging.
    Only runs the stages specified by task.
    """
    # No need to set up logging here; handled by worker_init
    ensure_directory(f"{results_dir}/{model_name}")
    results = load_json(f"{results_dir}/{model_name}/{instance_name}.json", default={})
    
    ModelClass = get_model_class(model_name)
    model = ModelClass()

    matcher = PreferenceMatcher(model_name=evaluator_model if not use_matcher else PREFMATCHER_MODEL_NAME)
    judger = ResponseJudger(model_name=evaluator_model)

    inferrer = PreferenceInferrer(model)
    generator = ResponseGenerator(model)

    if 'inference' not in results:
        results['inference'] = {}
    if 'generation' not in results:
        results['generation'] = {}

    # Inference + Matching
    if task in ["inference", "both"]:
        # 1. Inference: Infer the preference from the interactions
        if 'inferred' not in results['inference']:
            logger.info(f"[{model_name}] {instance_name}: Inference task...")
            try:
                preference, checklist = inferrer(
                    instance_data['current_request'], 
                    instance_data['prior_interactions']
                )
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    logger.warning(f"[{model_name}] {instance_name}: Context length exceeded during inference.")
                    preference = "ERROR: Context length exceeded"
                    checklist = []
                else:
                    logger.error(f"[{model_name}] {instance_name}: Inference task failed: {e}")
                    return
            results['inference'] = {
                "inferred": {
                    "preference": preference,
                    "checklist": checklist
                },
                "groundtruth": {
                    "preference": instance_data['current_contextual_preference'],
                    "checklist": instance_data['current_checklist']
                }
            }
            save_json(f"{results_dir}/{model_name}/{instance_name}.json", results, indent=4)

        # 2. Matching: Match the inferred preference to the groundtruth preference
        if 'match' not in results['inference']:
            logger.info(f"[{model_name}] {instance_name}: Evaluate match of inferred preference...")
            inferred = results['inference']['inferred']
            groundtruth = results['inference']['groundtruth']
            if len(inferred['checklist']) > 0:
                try:
                    match_infer_to_gt, _ = matcher(
                        inferred['checklist'],
                        groundtruth['preference']
                    )
                    match_gt_to_infer, _ = matcher(
                        groundtruth['checklist'],
                        inferred['preference']
                    )
                    results['inference']['match'] = {
                        "infer_to_gt": match_infer_to_gt,
                        "gt_to_infer": match_gt_to_infer
                    }
                except Exception as e:
                    logger.error(f"[{model_name}] {instance_name}: Matching task failed: {e}")
                    results['inference']['match'] = {
                        "infer_to_gt": [],
                        "gt_to_infer": []
                    }
            else:
                # if the inference is empty, set the match to all 0
                results['inference']['match'] = {
                    "infer_to_gt": [
                        {
                            "entry": entry,
                            "score": 0
                        } for entry in inferred['checklist'] 
                    ],
                    "gt_to_infer": [
                        {
                            "entry": entry,
                            "score": 0
                        } for entry in groundtruth['checklist'] 
                    ]
                }
            save_json(f"{results_dir}/{model_name}/{instance_name}.json", results, indent=4)

    # Generation + Judging
    if task in ["generation", "both"]:
        # 3. Response: Generate adapted response
        if 'ai_response' not in results['generation']:
            logger.info(f"[{model_name}] {instance_name}: Generate response...")
            try:
                request, response = generator(
                    instance_data['current_request'],
                    instance_data['prior_interactions']
                )
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    logger.warning(f"[{model_name}] {instance_name}: Context length exceeded during generation.")
                    response = "ERROR: Context length exceeded"
                else:
                    logger.error(f"[{model_name}] {instance_name}: Generation task failed: {e}")
                    return
            results['generation'] = {
                "user_request": request,
                "ai_response": response
            }
            save_json(f"{results_dir}/{model_name}/{instance_name}.json", results, indent=4)

        # 4. Judging: Evaluate the response
        if 'alignment' not in results['generation']:
            logger.info(f"[{model_name}] {instance_name}: Judge response...")
            if results['generation']['ai_response'] != "ERROR: Context length exceeded":
                try:
                    analysis, score = judger(
                        user_request=results['generation']['user_request'],
                        ai_response=results['generation']['ai_response'],
                        preference=instance_data['current_contextual_preference'],
                        checklist=instance_data['current_checklist']
                    )
                    results['generation']['alignment'] = {"score": score, "analysis": analysis}
                except Exception as e:
                    logger.error(f"[{model_name}] {instance_name}: Judging of generation failed: {e}")
                    results['generation']['alignment'] = {"score": 0, "analysis": str(e)}
            else:
                results['generation']['alignment'] = {"score": 0, "analysis": "Context length exceeded"}
            save_json(f"{results_dir}/{model_name}/{instance_name}.json", results, indent=4)


def evaluate_parallel(model_name, evaluator_model, results_dir, data_dir=None, use_matcher=False, n_workers=8, task="both", log_queue=None):
    """
    Run the evaluation pipeline in parallel for all data instances in a directory.
    Uses multiprocessing if n_workers > 1.
    Passes task to each evaluation.
    """
    args = []
    ensure_directory(results_dir)

    if data_dir is None:
        dataset = load_dataset(DATASET_NAME, split="test")
        for instance in dataset:
            instance_name = instance['persona_id']
            instance_data = {
                "current_request": instance['current_request'],
                "current_contextual_preference": instance['current_contextual_preference'],
                "current_checklist": instance['current_checklist'],
                "prior_interactions": instance['prior_interactions']
            }
            if n_workers == 1:
                evaluate(model_name, evaluator_model, instance_name, instance_data, results_dir, use_matcher=use_matcher, task=task)
            else:
                args.append((model_name, evaluator_model, instance_name, instance_data, results_dir, use_matcher, task))
    else:
        instances_filenames = os.listdir(f"{data_dir}")
        for i, instance_filename in enumerate(instances_filenames):
            instance_name = instance_filename.split(".")[0]
            instance_data = load_json(f"{data_dir}/{instance_filename}", default=None)
            if instance_data is None:
                logger.error(f"ERROR: Loading {data_dir}/{instance_filename}")
                continue
            if n_workers == 1:
                evaluate(model_name, evaluator_model, instance_name, instance_data, results_dir, use_matcher=use_matcher, task=task)
            else:
                args.append((model_name, evaluator_model, instance_name, instance_data, results_dir, use_matcher, task))
        
    if n_workers > 1:
        with Pool(n_workers, initializer=worker_init, initargs=(log_queue,)) as p:
            p.starmap(evaluate, args)