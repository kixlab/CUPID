import logging
from utils.parsing import parse_json
from utils.generation import Generator

logger = logging.getLogger(__name__)

class PreferenceDecomposer(Generator):
    def __init__(
        self, 
        model_name="gpt-4o-2024-11-20",
        prompt_path="synthesis/preference_decomposer.yaml", 
        temperature=0, 
        max_tokens=4096, 
        verbose=False
    ):
        super().__init__(
            model_name, 
            prompt_path, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            verbose=verbose
        )
    
    def __call__(self, preferences):
        try:
            logger.info(f"Decomposing preferences into checklist")
            logger.debug(f"Preferences: {preferences}")
            
            output = super().__call__(
                preferences=preferences
            )
            checklist = parse_json(output)['checklist']
            
            logger.info(f"Successfully decomposed preferences into {len(checklist)} checklist items")
            return checklist
            
        except Exception as e:
            logger.error(f"Error decomposing preferences: {str(e)}")
            logger.exception("Full traceback for preference decomposition:")
            raise
    
    def decompose_for_sessions(self, sessions):
        try:
            logger.info(f"Decomposing preferences for {len(sessions)} sessions")
            preference_checklist_map = {}

            for i, session in enumerate(sessions):
                preference = session['preference']
                if preference in preference_checklist_map:
                    logger.debug(f"Using cached checklist for session {i+1}")
                    session['checklist'] = preference_checklist_map[preference]
                else:
                    logger.debug(f"Decomposing new preference for session {i+1}: {preference}")
                    checklist = self(preference)
                    preference_checklist_map[preference] = checklist
                    session['checklist'] = checklist
                    
            logger.info(f"Successfully decomposed preferences for all {len(sessions)} sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Error decomposing preferences for sessions: {str(e)}")
            logger.exception("Full traceback for session preference decomposition:")
            raise



