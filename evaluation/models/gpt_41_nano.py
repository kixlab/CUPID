"""
GPT-4.1 Nano model wrapper for evaluation pipeline.
Implements the Model interface and registers itself for use.
"""
import os
import openai
import logging
from evaluation.models.model import Model, register_model

openai_client = openai.Client(
    api_key=os.environ.get("OPENAI_API_KEY")
)
logger = logging.getLogger(__name__)

@register_model
class GPT41Nano(Model):
    """
    Model wrapper for OpenAI GPT-4.1 Nano.
    """
    model_name = "gpt-4.1-nano-2025-04-14"

    def __call__(self, system_prompt, user_prompt):
        """
        Generate a response from the model given system and user prompts.
        """
        try:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise