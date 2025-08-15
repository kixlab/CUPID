import os
import random
import logging
from typing import List
from utils.files import load_json, save_json, ensure_directory

logger = logging.getLogger(__name__)

def format_prior_interaction(sessions: List[dict], interactions: List[dict], invalid_session_ids: List[int]) -> List[dict]:
    """
    Format prior interactions for a given session.
    """
    formatted_interactions = []
    for interaction in interactions[:-1]:
        session_id = interaction['session_id']
        session_data = next((s for s in sessions if s['id'] == session_id), None)
        if session_data is None:
            continue
        if session_id in invalid_session_ids:
            continue
        formatted_interactions.append({
            "context_factor": session_data['context_factor'],
            "contextual_preference": session_data['preference'],
            "dialogue": interaction['chat']
        })
    return formatted_interactions

def create_instances(output_dir: str) -> None:
    """
    Create data instances (consistent, contrastive, changing, etc.) from synthesized interaction sessions in output_dir/data.
    
    This function processes synthesized interaction session data to create different types of evaluation instances:
    - consistent: Uses interaction sessions with the same context factor as the main request, but without a change in the preference.
    - contrastive: Same as consistent, but also includes interaction sessions with the contrasting context factor (similar factor but different preference).
    - changing: Same as consistent, but also includes interaction sessions with same context factor as the main request and the changed preference.
    
    Saves instances to output_dir/instances.
    """
    data_dir = os.path.join(output_dir, "data")
    instances_dir = os.path.join(output_dir, "instances")
    
    # Create instances directory if it doesn't exist
    ensure_directory(instances_dir)
    
    data_filenames = os.listdir(data_dir)
    
    for i, data_filename in enumerate(data_filenames):
        data_path = os.path.join(data_dir, data_filename)
        data = load_json(data_path, default=None)
        data_name = data_filename.split(".")[0]
        
        # Skip files that don't have all required data
        if not data or 'context_factors' not in data or 'sessions' not in data or 'interactions' not in data:
            logger.warning(f"{data_filename}: Missing required keys, skipping.")
            continue
        
        context_factors = data['context_factors']
        sessions = data['sessions']
        interactions = data['interactions']
        
        # The main factor is from the last session (the main request)
        main_factor = sessions[-1]['context_factor']
        
        # Find the contrastive factor (one that's similar to the main factor but different in the preference)
        contrastive_factor = None
        for factor in context_factors:
            if factor['related_factor'] == main_factor:
                contrastive_factor = factor['factor']
                break
        
        # Number of sessions that we expect will include the same factor
        NUM_DUPLICATES = 2
        
        # Categorize sessions by their context factors
        consistent_session = None  # Session that includes the original preference of the main factor
        consistent_session_ids = []  # Sessions with same factor/preference as the main factor
        contrastive_session_ids = []  # Sessions with contrastive factor
        changing_session_ids = []  # Sessions that include the main factor but with changed preference
        
        # Process all sessions except the last one (which is the main request)
        for session in sessions[:-1]:
            if session['context_factor'] == main_factor and len(consistent_session_ids) < NUM_DUPLICATES:
                # First few sessions with main factor are the consistent sessions
                consistent_session = session
                consistent_session_ids.append(session['id'])
            elif session['context_factor'] == contrastive_factor:
                # Sessions with contrastive factor
                contrastive_session_ids.append(session['id'])
            elif session['context_factor'] == main_factor and len(consistent_session_ids) == NUM_DUPLICATES:
                # Additional sessions with main factor are the changing sessions
                changing_session_ids.append(session['id'])
        
        # Find remaining sessions for random sampling
        used_session_ids = consistent_session_ids + contrastive_session_ids + changing_session_ids
        remaining_session_ids = [session['id'] for session in sessions[:-1] if session['id'] not in used_session_ids]
        
        # Need enough sessions to create meaningful instances
        if len(remaining_session_ids) < NUM_DUPLICATES:
            logger.warning(f"{data_filename}: Not enough sessions to create instances")
            continue
        
        # Sample random sessions that will be removed from the instances
        random_session_ids = random.sample(remaining_session_ids, NUM_DUPLICATES)
        
        try:
            # 1) Consistent instance: uses the original preference of the main factor
            # - Excludes contrastive and changing interactions
            instance = {
                "persona_id": data_name,
                "instance_type": "consistent",
                "current_request": interactions[-1]['chat'][0]['content'],  # The final request to be answered
                "current_context_factor": main_factor,
                "current_contextual_preference": consistent_session['preference'],
                "current_checklist": consistent_session['checklist'],
                "prior_interactions": format_prior_interaction(sessions, interactions, random_session_ids + contrastive_session_ids + changing_session_ids)
            }
            save_json(os.path.join(instances_dir, f"{data_name}+consistent.json"), instance, indent=2)

            # 2) Contrastive instance: includes contrastive interactions
            # - Excludes changing and random sessions (for equal number of sessions across instances)
            instance['instance_type'] = "contrastive"
            instance['prior_interactions'] = format_prior_interaction(sessions, interactions, random_session_ids + changing_session_ids)
            save_json(os.path.join(instances_dir, f"{data_name}+contrastive.json"), instance, indent=2)

            # 3) Changing instance: includes the changing sessions with the new preference for main factor
            # - Excludes contrastive and random sessions (for equal number of sessions across instances)
            instance['instance_type'] = "changing"
            instance['current_contextual_preference'] = sessions[-1]['preference']
            instance['current_checklist'] = sessions[-1]['checklist']
            instance['prior_interactions'] = format_prior_interaction(sessions, interactions, random_session_ids + contrastive_session_ids)
            save_json(os.path.join(instances_dir, f"{data_name}+changing.json"), instance, indent=2)
            
        except Exception as e:
            logger.exception(f"{data_filename}: Failed to create one or more instances: {e}") 