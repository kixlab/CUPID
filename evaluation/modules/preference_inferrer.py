"""
Module for inferring user preferences from interaction logs during evaluation.
Uses prompt templates and a preference decomposer for checklist extraction.
"""
import yaml
from importlib import resources
from utils import format_interaction_log
from synthesis.modules import PreferenceDecomposer
import logging

logger = logging.getLogger(__name__)

class PreferenceInferrer:
    """
    Infers user preferences and generates a checklist from interaction logs using a model.
    """
    def __init__(self, model):
        prompt_file = resources.files('prompts') / "evaluation/preference_inferrer.yaml"
        with prompt_file.open('r') as file:
            prompt = yaml.safe_load(file)
        self.system_template = prompt.get('system_prompt', None)
        self.prompt_template = prompt['user_prompt']
        self.decomposer = PreferenceDecomposer()
        self.model = model

    def process_output(self, output):
        """
        Extract the most likely preference from the model output.
        """
        try:
            processed = output.split("### Most Likely Preference")[1].strip()
            processed = processed.split("\n")[0].strip()
        except Exception as e:
            logger.error(f"Error processing output: {e}\nOutput: {output}")
            raise
        return processed
    
    def __call__(self, curr_request, prev_interactions):
        """
        Infer preference and checklist from the current request and previous interactions.
        """
        try:
            output = self.model(
                system_prompt=self.system_template,
                user_prompt=self.prompt_template.format(
                    curr_request=curr_request,
                    interaction_log=format_interaction_log(prev_interactions)
                )
            )
            preference = self.process_output(output)
        except Exception as e:
            logger.error(f"Error inferring preference: {e}")
            raise
        checklist = self.decomposer(preference)
        return preference, checklist