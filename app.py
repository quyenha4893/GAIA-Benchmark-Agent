from smolagents import CodeAgent, DuckDuckGoSearchTool, GradioUI, LiteLLMModel, PythonInterpreterTool
import yaml
from tools.final_answer import FinalAnswerTool, check_reasoning
from tools.tools import go_back, close_popups, search_item_ctrl_f, save_screenshot, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file, save_and_read_file

from dotenv import load_dotenv

# Load prompts from YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Get the system prompt from the YAML file
#system_prompt = prompt_templates["system_prompt"]

search_tool = DuckDuckGoSearchTool()
python_interpretor_tool = PythonInterpreterTool()

search_model_name = 'granite3.3:latest'
search_model = LiteLLMModel(model_id=f'ollama_chat/{search_model_name}',
                            flatten_messages_as_text=True)

web_agent = CodeAgent(
    model=search_model,
    tools=[search_tool, go_back, close_popups, search_item_ctrl_f],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name="web_agent",
    description="Browses the web to find information",
    additional_authorized_imports=[],
    prompt_templates=prompt_templates
)

image_model_name = 'gemma3:12b'
image_model = LiteLLMModel(model_id=f'ollama_chat/{image_model_name}',
                            flatten_messages_as_text=True)
image_agent = CodeAgent(
    model=image_model,
    tools=[],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    additional_authorized_imports=["PIL", "requests", "io"],
    name="image_agent",
    description="Review images and visual data for answers to questions based on visual data",
    prompt_templates=prompt_templates
)

react_model_name = 'cogito:14b'
# Initialize the chat model
react_model = LiteLLMModel(model_id=f'ollama_chat/{react_model_name}',
                            flatten_messages_as_text=True)
final_answer = FinalAnswerTool()

manager_agent = CodeAgent(
    model=react_model,
    tools=[python_interpretor_tool,download_file_from_url,extract_text_from_image, analyze_csv_file, analyze_excel_file, save_and_read_file,final_answer],
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

if __name__ == "__main__":
    GradioUI(manager_agent).launch()
