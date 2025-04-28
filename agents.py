import smolagents.models as sm_models

_orig_roles = sm_models.MessageRole.roles

@classmethod
def _roles_with_control(cls):
    return _orig_roles() + ["control"]

sm_models.MessageRole.roles = _roles_with_control



from smolagents import (CodeAgent, 
                        GradioUI, 
                        LiteLLMModel, 
                        OpenAIServerModel, 
                        ChatMessage, 
                        ToolCallingAgent)
from smolagents.default_tools import (DuckDuckGoSearchTool, 
                                      VisitWebpageTool, 
                                      WikipediaSearchTool, 
                                      SpeechToTextTool,
                                      PythonInterpreterTool)
import yaml
from tools.final_answer import FinalAnswerTool, check_reasoning, ensure_formatting
from tools.tools import youtube_frames_to_images, use_vision_model, search_item_ctrl_f, go_back, close_popups, save_and_read_file, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file
import os
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
search_model_name = 'cogito:14b'
# search_model_name = 'qwen2:7b'
search_model = ThinkingLiteLLMModel(model_id=f'ollama_chat/{search_model_name}',
                             flatten_messages_as_text=True)

web_agent = CodeAgent(
    model=search_model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), FinalAnswerTool()],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=6,
    name="web_agent",
    description="Searches the web using the and reviews web pages to find information.",
    additional_authorized_imports=['bs4', 'requests', 'io', 'wiki'],
    prompt_templates=prompt_templates
)

image_model_name = 'llama3.2-vision'
image_model = OpenAIServerModel(model_id=image_model_name,
                                api_base='http://localhost:11434/v1/',
                                api_key='ollama',
                            flatten_messages_as_text=False)
image_agent = ToolCallingAgent(
    model=image_model,
    tools=[FinalAnswerTool()],
    max_steps=4,
    verbosity_level=2,
    grammar=None,
    planning_interval=6,
    #additional_authorized_imports=["PIL", "requests", "io", "numpy"],
    name="image_agent",
    description="Review images and videos for answers to questions based on visual data",
    prompt_templates=prompt_templates
)

# react_model_name = 'mistral-small3.1:24b'
# # Initialize the chat model
# react_model = OpenAIServerModel(model_id=react_model_name,
#                                 api_base='http://localhost:11434/v1/',
#                                 api_key='ollama',
#                             flatten_messages_as_text=False)

react_model_name = "gemini/gemini-2.0-flash-exp"
react_model = LiteLLMModel(model_id=react_model_name, 
                           api_key=os.getenv("GEMINI_KEY"),
                           temperature=0.2
                           )


manager_agent = CodeAgent(
    model=react_model,
    tools=[FinalAnswerTool(), 
           DuckDuckGoSearchTool(), 
           VisitWebpageTool(), 
           WikipediaSearchTool(),
           SpeechToTextTool(),
           use_vision_model,
           youtube_frames_to_images,
           search_item_ctrl_f, go_back, close_popups, 
           save_and_read_file, download_file_from_url, 
           extract_text_from_image, 
           analyze_csv_file, analyze_excel_file
           ],
    managed_agents=[],
    additional_authorized_imports=['os'],
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
