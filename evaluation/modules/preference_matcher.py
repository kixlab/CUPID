"""
Module for matching inferred preferences to ground-truth preferences during evaluation.
Provides PreferenceMatcher class for scoring and label assignment.
"""
import json
from utils import Generator, parse_json
from config import PREFMATCHER_MODEL_NAME
import logging

logger = logging.getLogger(__name__)

EXAMPLES = [
    {
        "preference": "Procedures must prioritize equipment security and include hourly weather check requirements before any sampling activity.",
        "checklist": {
            "Does the protocol incorporate detailed weather monitoring procedures?": 0.5,
            "Does the protocol include equipment protection measures?": 0.5,
            "Does the protocol outline systematic decision-making processes for adapting to changing field conditions during wet season sampling?": 0
        }
    },
    {
        "preference": "Test results should focus primarily on user engagement metrics and social interaction patterns between participants.",
        "checklist": {
            "Does the analysis protocol thoroughly explore player engagement?": 0.5,
            "Does the analysis protocol thoroughly explore collaboration?": 0.5,
            "Does the analysis protocol thoroughly explore competition?": 0,
            "Does the analysis protocol provide actionable insights to enhance the AR mini-game's social dynamics and overall player experience?": 0.5
        }
    },
    {
        "preference": "Solutions must strictly adhere to established military doctrine and proven tactical approaches, avoiding experimental or innovative methods.",
        "checklist": {
            "Does the training plan align with established military standards?": 1,
            "Is the training plan clear?": 0.5,
            "Is the training plan structured?": 1,
            "Does the training plan include objective, data-driven performance evaluation criteria?": 1
        }
    },
    {
        "preference": "Presentation content must use formal academic language and technical terminology with precise timing structures.",
        "checklist": {
            "Is the presentation structured formally?": 1,
            "Does the presentation incorporate discipline-specific terminology?": 0.5,
            "Does the presentation provide a clear time allocation for each section within the keynote speech format?": 0.5
        }
    },
    {
        "preference": "Authentication checks must be completed within 15 minutes and focus only on critical identifying features identified in standard guides.",
        "checklist": {
            "Does the procedure prioritize key distinctive features of military stamps?": 0,
            "Does the procedure allow for efficient verification within time constraints?": 1,
            "Does the procedure adhere to established philatelic standards for accuracy?": 0
        }
    },
    {
        "preference": "Statistical findings should be presented primarily through player impact stories and career trajectories.",
        "checklist": {
            "Does the presentation clearly communicate the analytics team's impact?": 0.5,
            "Does the presentation use engaging narratives?": 0.5,
            "Does the presentation include relatable examples?": 0.5
        }
    },
    {
        "preference": "All scientific analysis must prioritize identifying and correcting misconceptions, regardless of the documentary's artistic merit.",
        "checklist": {
            "Does the analysis critically assess each segment for oversimplification?": 0,
            "Does the analysis critically assess each segment for inaccuracies?": 1,
            "Does the analysis provide detailed scientific corrections for any identified issues?": 0.5,
            "Does the analysis provide detailed scientific clarifications for any identified issues?": 0.5
        }
    },
    {
        "preference": "Review processes must prioritize comprehensive stakeholder input over speed, with extended review periods.",
        "checklist": {
            "Do the procedures outline structured review stages?": 1,
            "Are stakeholder responsibilities clearly assigned in the procedures?": 1,
            "Do the procedures provide adequate timelines for feedback?": 0.5,
            "Do the procedures apply to all content types to ensure comprehensive input?": 0
        }
    }
]

def score_to_label(score):
    """
    Convert a numeric score to a coverage label.
    """
    if score == 1:
        return "Fully Covered"
    elif score == 0.5:
        return "Partially Covered"
    else:
        return "Not Covered"

class PreferenceMatcher(Generator):
    """
    Matches checklist entries to preferences and assigns coverage labels using a model.
    """
    def __init__(
        self,
        model_name="gpt-4o-2024-11-20",
        prompt_file="evaluation/preference_matcher.yaml",
        temperature=0,
        max_tokens=8192,
        verbose=False
    ):
        self.is_finetuned = model_name == PREFMATCHER_MODEL_NAME
        super().__init__(
            model_name,
            prompt_file if not self.is_finetuned else "evaluation/preference_matcher_model.yaml",
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose
        )

    def __call__(self, checklist, preference):
        """
        Match checklist entries to a preference and return results with labels.
        """
        examples_str = ""
        for example in EXAMPLES:
            examples_str += f"#### Preference\n\n{example['preference']}\n\n#### Output\n\n```json\n"
            json_obj = {
                "results": [
                    {
                        "index": i + 1,
                        "entry": entry,
                        "label": score_to_label(example['checklist'][entry]),
                    } for i, entry in enumerate(example['checklist'])
                ]
            }
            examples_str += json.dumps(json_obj, indent=2)
            examples_str += "\n```\n\n---\n\n"
        try:
            output = super().__call__(
                checklist="\n".join([f"{j + 1}. {entry}" for j, entry in enumerate(checklist)]),
                preference=preference,
                examples=examples_str.strip() if not self.is_finetuned else None
            )
            matches = parse_json(output)
        except Exception as e:
            logger.error(f"Error parsing matcher output: {e}\nOutput: {output if 'output' in locals() else ''}")
            raise
      
        return matches['results'] if not self.is_finetuned else matches, output