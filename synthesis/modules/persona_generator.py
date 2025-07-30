import logging
from utils.parsing import parse_json
from utils.generation import Generator

logger = logging.getLogger(__name__)

class PersonaGenerator(Generator):
    def __init__(
        self, 
        batch_size=4, 
        model_name="claude-3-5-sonnet-20241022", 
        prompt_path="synthesis/persona_generator.yaml", 
        temperature=0.7, 
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
        self.batch_size = batch_size
    
    def __call__(self, persona_templates):
        personas = []
        for i in range(0, len(persona_templates), self.batch_size):
            batch = persona_templates[i:i + self.batch_size]
            batch_input_str = "\n".join(map(lambda x: f"{x[0] + 1}. {x[1]}", enumerate(batch)))

            logger.info(f"Processing persona batch {i//self.batch_size + 1}/{(len(persona_templates)-1)//self.batch_size + 1}")
            logger.debug(f"Batch input: {batch_input_str}")

            try:
                output = super().__call__(
                    seed_descriptions=batch_input_str
                )

                processed_output = parse_json(output)["personas"]
                for persona, output in zip(batch, processed_output):
                    persona['description'] = output['description']
                    persona['occupation'] = output['occupation']
                    personas.append(persona)
                    
                logger.info(f"Successfully processed {len(processed_output)} personas in batch")
                
            except Exception as e:
                logger.error(f"Error processing persona batch {i//self.batch_size + 1}: {str(e)}")
                logger.exception(f"Full traceback for persona batch {i//self.batch_size + 1}:")
                raise
        
        for i, persona in enumerate(personas):
            persona['seed'] = persona_templates[i]['seed']

        logger.info(f"Generated {len(personas)} personas total")
        return personas