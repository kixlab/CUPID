"""
CLI entrypoint for evaluation pipeline.
Evaluates model performance on the CUPID dataset or locally synthesized data.

Usage: python evaluation/run.py --results_dir <results_dir> --model <model> [other options]
Run this script from the cupid root directory.
"""
import argparse
import logging
import os
from evaluation.pipeline.evaluate import evaluate_parallel
from evaluation.pipeline.result import aggregate_results
from utils.files import save_json
from utils.logging import setup_main_logging

def main():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline for CUPID.")
    parser.add_argument("--results_dir", type=str, help="Directory to store evaluation results", required=True)
    parser.add_argument("--model", type=str, help="Model to be evaluated (must match a class in evaluation/models/)", required=True)
    parser.add_argument("--data_dir", type=str, help="Directory containing data instances to evaluate if using locally synthesized data (default: evaluate using CUPID dataset)")
    parser.add_argument("--evaluator", type=str, default="gpt-4o-2024-11-20", help="Model used for evaluation functions (default: gpt-4o-2024-11-20)")
    parser.add_argument("--use_matcher", action="store_true", help="Use the preference matcher model (default: False)")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--log_file", type=str, default="log.txt", help="Path to log file (optional)")
    parser.add_argument("--task", type=str, choices=["inference", "generation", "both"], default="inference", help="Which evaluation stages to run: inference, generation, or both (default: inference)")
    args = parser.parse_args()
    results_dir = args.results_dir
    data_dir = args.data_dir

    # Setup structured logging
    if args.log_file:
        log_file_path = args.log_file
    else:
        # Create a timestamped log file in the current directory
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"evaluation_{timestamp}.log"
    
    # Multiprocessing-safe logging setup
    log_queue, log_listener = setup_main_logging(log_file_path)
    log_listener.start()

    # Get a logger for this main module
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting evaluation pipeline with parameters:")
    logger.info(f"  Results directory: {results_dir}")
    logger.info(f"  Data: {data_dir if data_dir else 'CUPID dataset'}")
    logger.info(f"  Evaluated Model: {args.model}")
    logger.info(f"  Evaluator Model: {args.evaluator}")
    logger.info(f"  Use PrefMatcher: {args.use_matcher}")
    logger.info(f"  Workers: {args.n_workers}")
    logger.info(f"  Log file: {log_file_path}")

    try:
        # Evaluate the model on the specified data
        logger.info("Phase 1: Evaluating model performance...")
        evaluate_parallel(
            args.model,
            args.evaluator,
            results_dir,
            data_dir,
            use_matcher=args.use_matcher,
            n_workers=args.n_workers,
            task=args.task,
            log_queue=log_queue,  # Pass the log queue
        )
        logger.info("Evaluation completed successfully!")

        results = aggregate_results(results_dir, args.model, args.task)
        save_json(f"{results_dir}/{args.model}/results.json", results, indent=2)
        logger.info("Aggregation completed successfully!")
        logger.info(f"Results saved to {results_dir}/{args.model}/results.json")
        # Print main results
        if args.task in ["inference", "both"]:
            logger.info(f"Inference Results:")
            logger.info(f"  Precision: {results['inference']['precision']*100:.2f}%")
            logger.info(f"  Recall: {results['inference']['recall']*100:.2f}%")
            logger.info(f"  F1: {results['inference']['f1']*100:.2f}%")
        if args.task in ["generation", "both"]:
            logger.info(f"Generation Results:")
            logger.info(f"  Average Score: {results['generation']['average_score']:.2f} / 10")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        logger.exception("Full error details:")
        raise
    finally:
        log_listener.stop()

if __name__ == "__main__":
    main()