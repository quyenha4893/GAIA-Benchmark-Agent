from typing import Any
from smolagents.tools import Tool
from smolagents import LiteLLMModel

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False


def check_reasoning(final_answer, agent_memory):
    model_name = 'cogito:14b'
    multimodal_model = LiteLLMModel(model_id=f'ollama_chat/{model_name}')
    prompt = f"""
        Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the answer that was given: 
        {final_answer}
        Please check that the reasoning process and results are correct: do they correctly answer the given task?
        First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not.
        Don't be harsh: if the result mostly solves the task, it should pass.
        """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True