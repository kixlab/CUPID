import logging
from utils.parsing import parse_yaml
from utils.generation import Generator

logger = logging.getLogger(__name__)

class ContextFactorsGenerator(Generator):
    def __init__(
        self, 
        model_name="claude-3-5-sonnet-20241022", 
        prompt_path="synthesis/context_factors_generator.yaml", 
        temperature=1.0, 
        max_tokens=8192, 
        verbose=False
    ):
        super().__init__(
            model_name, 
            prompt_path, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            verbose=verbose
        )
    
    def __call__(self, user_profile, n_factors=10):
        try:
            logger.info(f"Generating {n_factors} context factors for user profile")
            logger.debug(f"User profile: {user_profile.get('description', 'N/A')}")
            
            output = super().__call__(
                user_persona=user_profile['description'],
                n_factors=n_factors
            )
            processed = parse_yaml(output)
            context_factors = processed['context_factors']
            
            logger.info(f"Successfully generated {len(context_factors)} context factors")
            return context_factors
            
        except Exception as e:
            logger.error(f"Error generating context factors: {str(e)}")
            logger.exception("Full traceback for context factors generation:")
            raise