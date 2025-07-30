import logging
import os
from multiprocessing import Pool
from typing import List
from synthesis.modules import ContextFactorsGenerator, SessionsGenerator, PreferenceDecomposer, InteractionSimulator
from utils.validation import validate_context_factors, validate_sessions
from utils.files import load_json, save_json, ensure_directory
from utils.logging import setup_worker_logging

logger = logging.getLogger(__name__)

log_queue = None  # Global for worker processes

def worker_init(queue):
    global log_queue
    log_queue = queue
    setup_worker_logging(log_queue)

def synthesize_data_with_logging(args_tuple):
    """
    Wrapper function that sets up logging for each worker process.
    """
    model_name, persona, output_dir, n_factors, n_sessions, max_turns = args_tuple
    return synthesize_data(model_name, persona, output_dir, n_factors, n_sessions, max_turns)

def synthesize_data(model_name: str, persona: dict, output_dir: str, n_factors: int, n_sessions: int, max_turns: int) -> None:
    """
    For a given persona, generate context factors, sessions, and interactions, saving to output_dir/data.
    
    This function implements a three-phase data synthesis process:
    
    Phase 1: Context Factors Generation
    - Generates diverse context factors that influence how the persona behaves
    - Validates the generated factors to ensure they meet quality standards
    
    Phase 2: Sessions Generation  
    - Creates interaction sessions based on the persona and context factors
    - Each session represents a different conversational scenario
    - Decomposes preference for each session to guide interaction generation
    
    Phase 3: Interactions Generation
    - Simulates actual conversations between user and assistant for each session
    - Uses the persona and session context to generate realistic interactions
    - Handles variable conversation lengths (max_turns, with last session having 0 turns)
    
    The process is incremental - if any phase is already complete, it skips to the next phase.
    This allows for resuming interrupted synthesis and selective regeneration.
    """
    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize all generators needed for the synthesis process
        context_generator = ContextFactorsGenerator(model_name=model_name)
        sessions_generator = SessionsGenerator(model_name=model_name)
        preference_decomposer = PreferenceDecomposer(model_name=model_name)
        interaction_simulator = InteractionSimulator(model_name=model_name)

        # Create a clean filename from persona info
        persona_id = persona['id']
        occupation = persona['occupation'].replace('/', ' and ').replace(' ', '_').lower()  # Clean up occupation for filename
        filename = f"{output_dir}/data/{persona_id}+{occupation}.json"
        
        # Load existing data if available (for incremental processing)
        data = load_json(filename, default={})

        # Phase 1: Generate context factors if not already done
        if 'context_factors' not in data:
            logger.info(f"[{persona_id} {occupation}] - Generating context factors...")
            try:
                data['context_factors'] = context_generator(persona, n_factors=n_factors)
                
                # Validate the generated context factors
                is_valid, error_msg = validate_context_factors(data['context_factors'])
                if not is_valid:
                    logger.error(f"[{persona_id} {occupation}] - FAILED generating context factors - {error_msg}")
                    return
                
                logger.info(f"[{persona_id} {occupation}] - SUCCESS generated context factors")
                save_json(filename, data, indent=4)
            except Exception as e:
                logger.error(f"[{persona_id} {occupation}] - ERROR generating context factors: {str(e)}")
                logger.exception(f"[{persona_id} {occupation}] - Full traceback for context factors generation:")
                return

        # Phase 2: Generate sessions if not already done
        if 'sessions' not in data:
            logger.info(f"[{persona_id} {occupation}] - Generating interaction sessions...")
            try:
                data['sessions'] = sessions_generator(persona, data['context_factors'], n_sessions=n_sessions)
                
                # Validate the generated sessions
                is_valid, error_msg = validate_sessions(data['sessions'], data['context_factors'], n_sessions)
                if not is_valid:
                    logger.error(f"[{persona_id} {occupation}] - FAILED generating interaction sessions - {error_msg}")
                    return
                
                # Decompose preference for each session to guide interaction generation
                data['sessions'] = preference_decomposer.decompose_for_sessions(data['sessions'])
                logger.info(f"[{persona_id} {occupation}] - SUCCESS generated interaction sessions")
                save_json(filename, data, indent=4)
            except Exception as e:
                logger.error(f"[{persona_id} {occupation}] - ERROR generating sessions: {str(e)}")
                logger.exception(f"[{persona_id} {occupation}] - Full traceback for sessions generation:")
                return

        # Phase 3: Generate interactions if not already done or incomplete
        if 'interactions' not in data or len(data['interactions']) != len(data['sessions']):
            logger.info(f"[{persona_id} {occupation}] - Generating interactions...")
            interactions = data.get('interactions', [])
            
            # Generate interactions for each session
            for i, session in enumerate(data['sessions']):
                session_id = session['id']
                
                # Skip if interaction already exists for this session
                if any(interaction['session_id'] == session_id for interaction in interactions):
                    continue
                
                try:
                    logger.info(f"[{persona_id} {occupation}] - Generating interaction for session {session_id} ({i+1}/{len(data['sessions'])})")
                    
                    # Generate interaction with variable turn length
                    # Last session gets 0 turns (just the request), others get max_turns
                    interaction = interaction_simulator(
                        persona, 
                        session, 
                        max_turns=max_turns if i != len(data['sessions']) - 1 else 0
                    )
                    interactions.append(interaction)
                    
                    # Save interactions incrementally
                    interactions = sorted(interactions, key=lambda x: x['session_id'])
                    data['interactions'] = interactions
                    save_json(filename, data, indent=4)
                    
                    logger.info(f"[{persona_id} {occupation}] - SUCCESS generated interaction for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"[{persona_id} {occupation}] - ERROR generating interaction for session {session_id}: {str(e)}")
                    logger.exception(f"[{persona_id} {occupation}] - Full traceback for interaction generation (session {session_id}):")
                    # Continue with next session instead of failing completely
                    continue
            
            # Check if all interactions were generated successfully
            if len(data['interactions']) == len(data['sessions']):
                logger.info(f"[{persona_id} {occupation}] - SUCCESS generated all interactions")
            else:
                logger.warning(f"[{persona_id} {occupation}] - WARNING: Only generated {len(data['interactions'])}/{len(data['sessions'])} interactions")
                
    except Exception as e:
        logger.error(f"[{persona_id} {occupation}] - FATAL ERROR in synthesize_data: {str(e)}")
        logger.exception(f"[{persona_id} {occupation}] - Full traceback for synthesize_data:")
        raise

def synthesize_data_parallel(model_name, personas: List[dict], output_dir: str, n_factors: int, n_sessions: int, max_turns: int, n_workers: int = 8, log_file_path: str = None, log_queue=None) -> None:
    """
    Synthesize data for all personas in parallel.
    
    This function orchestrates the parallel synthesis of data for multiple personas.
    It can run either in parallel mode (using multiprocessing) or sequential mode
    depending on the n_workers parameter.
    
    Args:
        model_name: The model to use for generation
        personas: List of persona dictionaries to process
        output_dir: Directory to save synthesized data
        n_factors: Number of context factors to generate per persona
        n_sessions: Number of interaction sessions to generate per persona
        max_turns: Maximum turns per interaction (except last session)
        n_workers: Number of parallel workers (1 = sequential, >1 = parallel)
        log_file_path: Path to the main log file (for worker log files)
    """
    # Ensure data directory exists
    data_dir = f"{output_dir}/data"
    ensure_directory(data_dir)
    
    # Use default log file if not provided
    if log_file_path is None:
        log_file_path = "synthesis.log"
    
    # Run either in parallel or sequential mode
    if n_workers > 1:
        logger.info(f"Starting parallel synthesis with {n_workers} workers")
        
        # Prepare arguments for each persona with worker-specific logging
        args = []
        for i, persona in enumerate(personas):
            worker_id = i % n_workers  # Distribute personas across workers
            args.append((model_name, persona, output_dir, n_factors, n_sessions, max_turns))
        
        # Parallel processing using multiprocessing
        with Pool(n_workers, initializer=worker_init, initargs=(log_queue,)) as p:
            p.map(synthesize_data_with_logging, args)
            
        logger.info("Parallel synthesis completed")
    else:
        logger.info("Starting sequential synthesis")
        
        # Sequential processing
        for persona in personas:
            synthesize_data(model_name, persona, output_dir, n_factors, n_sessions, max_turns)
            
        logger.info("Sequential synthesis completed") 