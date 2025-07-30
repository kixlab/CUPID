"""
Module for generating AI assistant responses during evaluation.
Loads prompt templates and formats interaction logs for model input.
"""
import yaml
from importlib import resources
from utils import format_interaction_log
import logging

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates AI assistant responses given the current request and previous interactions.
    """
    def __init__(self, model):
        prompt_file = resources.files('prompts') / "evaluation/response_generator.yaml"
        with prompt_file.open('r') as file:
            prompt = yaml.safe_load(file)
        self.system_template = prompt.get('system_prompt', None)
        self.prompt_template = prompt['user_prompt']
        self.model = model
    
    def __call__(self, curr_request, prev_interactions):
        """
        Generate a response for the current request, using the model and formatted interaction log.
        """
        interaction_log_str = format_interaction_log(prev_interactions)
        try:
            output = self.model(
                system_prompt=self.system_template,
                user_prompt=self.prompt_template.format(
                    curr_request=curr_request,
                    interaction_log=interaction_log_str
                )
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
        return curr_request, output