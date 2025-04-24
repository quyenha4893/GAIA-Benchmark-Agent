from smolagents import CodeAgent, DuckDuckGoSearchTool, GradioUI, LiteLLMModel
import yaml
#from Gradio_UI import GradioUI
from langchain_ollama import ChatOllama
from tools.final_answer import FinalAnswerTool

from dotenv import load_dotenv

# Load prompts from YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Get the system prompt from the YAML file
system_prompt = prompt_templates["system_prompt"]

search_tool = DuckDuckGoSearchTool()
model_name = 'cogito:14b'

# Initialize the chat model
model = LiteLLMModel(model_id=f'ollama_chat/{model_name}')

final_answer = FinalAnswerTool()

agent = CodeAgent(
    model=model,
    tools=[search_tool, final_answer],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()