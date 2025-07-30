import json
import logging
from utils.generation import GeneratorChat

logger = logging.getLogger(__name__)

class InteractionSimulator:
    def __init__(
        self, 
        model_name="gpt-4o-mini",
        response_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 
        verbose=False
    ):
        self.verbose = verbose
        self.model_name = model_name
        self.response_model = response_model

    def __call__(self, user_profile, session, max_turns=20):
        request = session['request_with_factor']
        if 'resource' in session and session['resource'] is not None:
            request = request.replace(f"[resource]", session['resource'])
        
        try:
            user_simulator = UserSimulator(
                user_profile, 
                session['context_factor'],
                session['preference'],
                session['checklist'],
                request.strip(),
                model_name=self.model_name,
                verbose=self.verbose
            )
            assistant_simulator = GeneratorChat(
                self.response_model,
                "synthesis/assistant_simulator.yaml",
                temperature=0.7,
                max_tokens=2048
            )

            user_message = request
            ai_message = None
            is_complete = False
            for turn in range(max_turns):
                if turn != 0:
                    user_message, is_complete = user_simulator(ai_message)
                if is_complete:
                    break
                ai_message = assistant_simulator(user_message)
            
            chat_history = user_simulator.get_chat_history()
            if not is_complete:
                chat_history.append({
                    "role": "assistant",
                    "content": ai_message
                })
            
            return {
                "session_id": session['id'],
                "preference": session['preference'],
                "checklist": session['checklist'],
                "chat": chat_history,
                "is_satisfied": is_complete
            }
        except Exception as e:
            logger.error(f"Error in InteractionSimulator for session {session['id']}: {str(e)}")
            logger.exception(f"Full traceback for InteractionSimulator session {session['id']}:")
            raise

class UserSimulator(GeneratorChat):
    def __init__(
        self,
        user_profile,
        context_factor,
        preference,
        checklist,
        initial_message,
        model_name="gpt-4o-2024-11-20",
        prompt_path="synthesis/user_simulator.yaml", 
        temperature=0.3, 
        max_tokens=8192,
        verbose=False
    ):
        self.user_profile = user_profile
        self.preference = preference
        self.checklist = checklist
        self.N_RETRIES = 3

        super().__init__(
            model_name, 
            prompt_path,
            initial_message={
                "role": "assistant", 
                "content": f"#### Evaluation of AI Assistant's Response\n\nN/A\n\n#### Evaluation Score\n\nN/A\n\n#### Continue or End?\n\nCONTINUE\n\n#### Selected Checklist Items\n\nN/A\n\n#### Thinking\n\nN/A\n\n#### Your Message\n\n{initial_message}"
            },
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=False,
            user_profile=user_profile['description'],
            context_factor=context_factor,
            preference=preference,
            checklist="\n".join([f"{i+1}. {item}" for i, item in enumerate(checklist)]),
        )
        self.is_verbose = verbose
    
    def process_output(self, output):
        try:
            parsed = output.split("#### Continue or End?")[1].split("#### Selected Checklist Items")
            status = parsed[0].strip()
            response = parsed[1].split("#### Your Message")[1].strip()
            return response, True if status == "END" else False
        except Exception as e:
            logger.error(f"Error processing UserSimulator output: {str(e)}")
            logger.error(f"Output that failed to parse: {output}")
            raise e
    
    def __call__(self, message):
        n_tries = 0
        while True:
            output = super().__call__(message)
            try:
                processed, is_end = self.process_output(output)
                break
            except Exception as e:
                logger.warning(f"ERROR in UserSimulator: retrying {n_tries+1}/{self.N_RETRIES} - {str(e)}")
                n_tries += 1
                self.pop_last_turn()
                if n_tries >= self.N_RETRIES:
                    logger.error(f"UserSimulator failed after {self.N_RETRIES} retries")
                    logger.error(f"Final error: {e}")
                    logger.error(f"Final output: {output}")
                    raise Exception(f"{e}\n\nOutput: {output}")

        if self.is_verbose:
            text = f"\n\n---\n\n<<CHECKLIST>>\n{json.dumps(self.checklist, indent=2)}\n\n<<<AI>>>\n{message}\n\n<<<User>>>\n{output}\n\n---\n\n"
            logger.debug(text)
            with open("user_simulator.log", "a") as f:
                f.write(text)
            
        return processed, is_end

    def get_chat_history(self):
        # Flip the roles so that the user simulator's messages are the user's messages
        processed_history = []
        for message in self.chat_history:
            if message['role'] == "assistant":
                content, _ = self.process_output(message['content'])
                processed_history.append({
                    "role": "user",
                    "content": content
                })
            else:
                processed_history.append({
                    "role": "assistant",
                    "content": message['content']
                })
        return processed_history
