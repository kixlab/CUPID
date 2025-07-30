import yaml, random
import logging

from utils.parsing import parse_yaml
from utils.generation import Generator

logger = logging.getLogger(__name__)

class SessionsGenerator(Generator):
    def __init__(
        self, 
        model_name="claude-3-5-sonnet-20241022", 
        prompt_path="synthesis/sessions_generator.yaml", 
        temperature=0.7, 
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
    
    def create_series_structure(self, n_sessions):
        try:
            n_consistent = 4
            n_contrastive = 2
            
            consistent_indices = random.sample(range(n_sessions - 1 - 1), n_consistent)
            consistent_indices.sort()
            consistent_indices.append(n_sessions - 1)

            remaining_indices = [i for i in range(n_sessions) if i not in consistent_indices]
            
            contrastive_indices = random.sample(remaining_indices, n_contrastive)

            structure_str = ""
            for i in range(n_sessions):
                if i in consistent_indices and i not in consistent_indices[:len(consistent_indices)//2]:
                    structure_str += f"    - Scenario ID {i+1}: Context factor A with changed preference A'.\n"
                elif i in consistent_indices:
                    structure_str += f"    - Scenario ID {i+1}: Context factor A with original preference A.\n"
                elif i in contrastive_indices:
                    structure_str += f"    - Scenario ID {i+1}: Context factor B with original preference B.\n"
            return structure_str[:-1]
            
        except Exception as e:
            logger.error(f"Error creating series structure for {n_sessions} sessions: {str(e)}")
            logger.exception("Full traceback for series structure creation:")
            raise
    
    def factors_to_yaml(self, factors):
        try:
            processed = {"factors": []}
            for factor in factors:
                processed["factors"].append({
                    "context_factor": factor['factor'],
                    "preference": factor['preference'],
                    "task_types": factor["task_types"]
                })
            return yaml.dump(processed, default_flow_style=False)
            
        except Exception as e:
            logger.error(f"Error converting factors to YAML: {str(e)}")
            logger.exception("Full traceback for factors to YAML conversion:")
            raise
        
    def __call__(self, user_profile, context_factors, n_sessions=5):
        try:
            logger.info(f"Generating {n_sessions} interaction sessions")
            logger.debug(f"User profile: {user_profile.get('description', 'N/A')}")
            logger.debug(f"Context factors count: {len(context_factors)}")
            
            series_structure = self.create_series_structure(n_sessions)
            contrastive_factors = list(filter(lambda factor: 'related_factor' in factor and factor['related_factor'] != 'N/A', context_factors))
            
            if len(contrastive_factors) < 2:
                logger.warning(f"Found only {len(contrastive_factors)} contrastive factors, need at least 2")
                # Handle case where we don't have enough contrastive factors
                if len(contrastive_factors) == 0:
                    logger.error("No contrastive factors found, cannot generate sessions")
                    raise ValueError("No contrastive factors found")
                # Use the same factor twice if we only have one
                contrastive_factors = [contrastive_factors[0], contrastive_factors[0]]

            output = super().__call__(
                user_persona=user_profile['description'],
                n_sessions=n_sessions,
                context_factors=self.factors_to_yaml(context_factors),
                series_structure=series_structure,
                first_factor=contrastive_factors[0]['factor'],
                second_factor=contrastive_factors[1]['factor']
            )
            processed = parse_yaml(output)
            sessions = processed['scenarios']
            
            logger.info(f"Successfully generated {len(sessions)} interaction sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Error generating interaction sessions: {str(e)}")
            logger.exception("Full traceback for interaction sessions generation:")
            raise