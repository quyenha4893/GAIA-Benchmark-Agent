from smolagents import CodeAgent, DuckDuckGoSearchTool, GradioUI, LiteLLMModel
from smolagents.utils import encode_image_base64, make_image_url
import yaml
#from Gradio_UI import GradioUI
from langchain_ollama import ChatOllama
from tools.final_answer import FinalAnswerTool
import os
from PIL import Image

from dotenv import load_dotenv

# Load prompts from YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Get the system prompt from the YAML file
system_prompt = prompt_templates["system_prompt"]

search_tool = DuckDuckGoSearchTool()

search_model_name = 'granite3.3:latest'
search_model = LiteLLMModel(model_id=f'ollama_chat/{search_model_name}')

web_agent = CodeAgent(
    model=search_model,
    tools=[search_tool],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="web_agent",
    description="Browses the web to find information",
    prompt_templates=prompt_templates
)

react_model_name = 'cogito:14b'
# Initialize the chat model
react_model = LiteLLMModel(model_id=f'ollama_chat/{react_model_name}')
final_answer = FinalAnswerTool()

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

manager_agent = CodeAgent(
    model=react_model,
    tools=[final_answer],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    max_steps=6,
    verbosity_level=1,
    planning_interval=None,
    name="Manager",
    description="The manager of the team, responsible for overseeing and guiding the team's work.",
    final_answer_checks=[check_reasoning],
    prompt_templates=prompt_templates
)

GradioUI(manager_agent).launch()
