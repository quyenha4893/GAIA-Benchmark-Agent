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
from tools.tools import (youtube_frames_to_images, use_vision_model, 
                         read_file, download_file_from_url, 
                         extract_text_from_image, analyze_csv_file, 
                         analyze_excel_file, youtube_transcribe,
                         transcribe_audio, review_youtube_video)
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Load prompts from YAML file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
    
class SlowLiteLLMModel(LiteLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, messages, **kwargs) -> ChatMessage:
        time.sleep(15)
        # prepend onto whatever messages the Agent built
        return super().__call__(messages, **kwargs)


react_model_name = "gemini/gemini-2.0-flash-exp"
react_model = SlowLiteLLMModel(model_id=react_model_name, 
                           api_key=os.getenv("GEMINI_KEY"),
                           temperature=0.2
                           )


manager_agent = CodeAgent(
    model=react_model,
    tools=[FinalAnswerTool(), 
           DuckDuckGoSearchTool(), 
           VisitWebpageTool(max_output_length=500000), 
           WikipediaSearchTool(extract_format='HTML'),
           SpeechToTextTool(),
           youtube_frames_to_images,
           youtube_transcribe,
           use_vision_model,
           read_file, download_file_from_url, 
           extract_text_from_image, 
           analyze_csv_file, analyze_excel_file,
           transcribe_audio,
           review_youtube_video
           ],
    managed_agents=[],
    additional_authorized_imports=['os', 'pandas', 'numpy', 'PIL', 'tempfile', 'PIL.Image'],
    max_steps=20,
    verbosity_level=1,
    planning_interval=6,
    name="Manager",
    description="The manager of the team, responsible for overseeing and guiding the team's work.",
    final_answer_checks=[check_reasoning, ensure_formatting],
    prompt_templates=prompt_templates
)


if __name__ == "__main__":
    GradioUI(manager_agent).launch()
