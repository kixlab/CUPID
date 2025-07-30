"""
CLI entrypoint for synthetic data generation pipeline.
Generates personas, synthesizes data, and creates scenario instances.

Usage: python synthesis/run_synthesis.py --output_dir <output_dir> [other options]
Run this script from the cupid root directory.
"""
import argparse
import logging
import os
from synthesis.pipeline.persona import generate_personas
from synthesis.pipeline.synthesize import synthesize_data_parallel
from synthesis.pipeline.instances import create_instances
from utils.logging import setup_main_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Output directory for the synthesized data", required=True)
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", help="Model to use for the synthesis")
    parser.add_argument("--n_personas", type=int, default=4, help="Number of personas to generate (default: 4)")
    parser.add_argument("--n_factors", type=int, default=8, help="Number of context factors per persona (default: 8)")
    parser.add_argument("--n_sessions", type=int, default=13, help="Number of sessions per persona (default: 13)")
    parser.add_argument("--max_turns", type=int, default=16, help="Maximum number of turns per interaction (default: 16)")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--log_file", type=str, default="log.txt", help="Path to log file (optional)")
    args = parser.parse_args()
    output_dir = args.output_dir

    # Multiprocessing-safe logging setup
    log_queue, log_listener = setup_main_logging(args.log_file)
    log_listener.start()
    try:
        # Get a logger for this main module
        logger = logging.getLogger(__name__)
        logger.info(f"Starting synthesis pipeline with parameters:")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Personas: {args.n_personas}")
        logger.info(f"  Context factors: {args.n_factors}")
        logger.info(f"  Sessions: {args.n_sessions}")
        logger.info(f"  Max turns: {args.max_turns}")
        logger.info(f"  Workers: {args.n_workers}")
        logger.info(f"  Log file: {args.log_file}")

        # Generate given number of personas
        logger.info("Phase 1: Generating personas...")
        personas = generate_personas(args.model, args.n_personas, output_dir)
        logger.info(f"Successfully generated {len(personas)} personas")

        # Synthesize context factors, sessions, and interactions for each persona
        logger.info("Phase 2: Synthesizing data (context factors, sessions, interactions)...")
        synthesize_data_parallel(
            args.model,
            personas,
            output_dir,
            args.n_factors,
            args.n_sessions,
            args.max_turns,
            n_workers=args.n_workers,
            log_file_path=args.log_file,
            log_queue=log_queue,
        )
        logger.info("Data synthesis completed successfully")

        # Create data instances based on the synthesized data
        logger.info("Phase 3: Creating instances...")
        create_instances(output_dir)
        logger.info("Instance creation completed successfully")
        logger.info("Synthesis pipeline completed successfully!")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Synthesis pipeline failed: {str(e)}")
        logger.exception("Full error details:")
        raise
    finally:
        log_listener.stop()

if __name__ == "__main__":
    main() 