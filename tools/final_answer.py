from typing import Any
from smolagents.tools import Tool
from smolagents import LiteLLMModel

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Used to submit the final answer to the problem."
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
        Be reasonably strict.  You are being graded on your ability to provide the right answer.  You should have >90% confidence that the answer is correct.
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

def ensure_formatting(final_answer, agent_memory):
    # Ensure the final answer is formatted correctly
    model_name = 'granite3.3:8b'
    # Initialize the chat model
    model = LiteLLMModel(model_id=f'ollama_chat/{model_name}',
                             flatten_messages_as_text=True)
    prompt = f"""
        Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the FINAL ANSWER that was given: 
        {final_answer}
        Ensure the FINAL ANSWER is in the right format as asked for by the task.  Here are the instructions that you need to evaluate:
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
        If you are asked for a number, don't use commas to write your number.  Don't use units such as $ or percent sign unless specified otherwise. Write your number in Arabic numbers (such as 9 or 3 or 1093) unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        If you are asked for a comma separated list, ensure you only return the content of that list, and NOT the brackets '[]'
        First list reasons why it is/is not in the correct format and then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not.
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
    output = model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True