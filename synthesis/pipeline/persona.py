import os
import json
import logging
from synthesis.modules import PersonaTemplateSampler, PersonaGenerator
from utils.files import load_json, save_json, ensure_directory

logger = logging.getLogger(__name__)

def generate_personas(model_name: str, n_personas: int, output_dir: str) -> list:
    """
    Generate or load personas and save to output_dir/personas.json.
    
    This function implements an incremental persona generation process:
    1. First checks if personas already exist and loads them
    2. If we need more personas, generates diverse persona templates using sampling
    3. Uses the specified model to generate persona descriptions from templates
    4. Saves the updated personas list and returns it
    
    The process avoids regenerating existing personas and ensures diversity by
    tracking used seed data.

    Args:
        model_name (str): The model to use for persona generation.
        n_personas (int): Number of personas to generate.
        output_dir (str): Directory to save personas.

    Returns:
        list: List of persona dictionaries.
    """
    # Ensure output directory exists
    ensure_directory(output_dir)

    # Load existing personas if they exist
    personas_path = os.path.join(output_dir, "personas.json")
    personas = load_json(personas_path, default=[])

    # If we already have enough personas, return them
    if len(personas) >= n_personas:
        return personas

    # Initialize generators for creating new personas
    persona_sampler = PersonaTemplateSampler()  # Creates diverse persona templates
    persona_generator = PersonaGenerator(model_name=model_name)  # Generates full personas from templates
    
    # Track what seed data we've already used to ensure diversity
    used_seeds = [p.get('seed') for p in personas]
    n_remaining = n_personas - len(personas)

    # Generate diverse persona templates
    logger.info("Creating Diverse Persona Templates...")
    persona_templates = persona_sampler(n_remaining, used_seeds)
    
    # Generate full personas from templates using the specified model
    logger.info("Generating Personas...")
    new_personas = persona_generator(persona_templates)
    personas += new_personas
    
    # Log the generated personas for debugging (at debug level to avoid spam)
    logger.debug(json.dumps(personas, indent=2))

    # Save the updated personas list
    save_json(personas_path, personas, indent=4)
    return personas 