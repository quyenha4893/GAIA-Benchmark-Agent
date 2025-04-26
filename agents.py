import smolagents.models as sm_models

_orig_roles = sm_models.MessageRole.roles

@classmethod
def _roles_with_control(cls):
    return _orig_roles() + ["control"]

sm_models.MessageRole.roles = _roles_with_control



from smolagents import CodeAgent, DuckDuckGoSearchTool, GradioUI, LiteLLMModel, PythonInterpreterTool, OpenAIServerModel, ChatMessage, ToolCallingAgent, VisitWebpageTool
import yaml
from tools.final_answer import FinalAnswerTool, check_reasoning, ensure_formatting
#from tools.tools import go_back, close_popups, search_item_ctrl_f, save_screenshot, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file, save_and_read_file

from dotenv import load_dotenv

# Load prompts from YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


class ThinkingLiteLLMModel(LiteLLMModel):
    def __init__(self, *args, **kwargs):
        # ensure the Litellm client also maps "control" → "control"
        cr = kwargs.pop("custom_role_conversions", {})
        cr["control"] = "control"
        super().__init__(*args, custom_role_conversions=cr, **kwargs)

    def __call__(self, messages, **kwargs) -> ChatMessage:
        # NOTE: content must be a list of {type, text} dicts
        thinking_msg = {
            "role": "control",
            "content": [{"type": "text", "text": "thinking"}]
        }
        # prepend onto whatever messages the Agent built
        return super().__call__([thinking_msg] + messages, **kwargs)

# search_model_name = 'granite3.3:latest'
# search_model_name = 'cogito:14b'
search_model_name = 'qwen2:7b'
search_model = ThinkingLiteLLMModel(model_id=f'ollama_chat/{search_model_name}',
                             flatten_messages_as_text=True)

search_tool = DuckDuckGoSearchTool()
python_interpretor_tool = PythonInterpreterTool()
visit_webpage_tool = VisitWebpageTool()
final_answer = FinalAnswerTool()

web_agent = CodeAgent(
    model=search_model,
    tools=[search_tool, visit_webpage_tool, final_answer],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=6,
    name="web_agent",
    description="Browses the web to find information",
    additional_authorized_imports=['bs4', 'requests', 'io'],
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


manager_agent = CodeAgent(
    model=react_model,
    tools=[final_answer],
    managed_agents=[web_agent],
    additional_authorized_imports=[],
    max_steps=6,
    verbosity_level=1,
    planning_interval=6,
    name="Manager",
    description="The manager of the team, responsible for overseeing and guiding the team's work.",
    final_answer_checks=[check_reasoning, ensure_formatting],
    prompt_templates=prompt_templates
)



if __name__ == "__main__":
    GradioUI(manager_agent).launch()
