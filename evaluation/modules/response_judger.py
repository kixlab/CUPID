"""
Module for judging AI assistant responses during evaluation.
Provides ResponseJudger class for scoring and analysis extraction.
"""
from utils import Generator
import logging

logger = logging.getLogger(__name__)

class ResponseJudger(Generator):
    """
    Judges AI assistant responses for alignment with user preferences and checklists.
    """
    def __init__(
        self, 
        model_name="gpt-4o-2024-11-20", 
        prompt_file="evaluation/response_judger.yaml", 
        temperature=0, 
        max_tokens=8192, 
        verbose=False
    ):
        super().__init__(model_name, prompt_file, temperature=temperature, max_tokens=max_tokens, verbose=verbose)
    
    def process_output(self, output):
        """
        Extract the evaluation analysis and score from the model output.
        """
        try:
            # Extract score section by looking for exact header match
            score_headers = [
                "### **Evaluation Score**",
                "### Evaluation Score"
            ]
            for header in score_headers:
                if header in output:
                    score = output.split(header)[1].strip()
                    break
        except Exception as e:
            logger.error(f"Error processing response judger output: {e}\nOutput: {output}")
            raise
        return output, score

    def __call__(self, user_request, ai_response, preference, checklist):
        """
        Judge the AI response for alignment with the given preference and checklist.
        """
        try:
            output = super().__call__(
                user_request=user_request,
                ai_response=ai_response,
                preference=preference,
                checklist="\n".join(list(map(lambda x: f"- {x}", checklist)))
            )
            return self.process_output(output)
        except Exception as e:
            logger.error(f"Error judging response: {e}")
            raise