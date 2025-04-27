import os
import gradio as gr
import requests
import pandas as pd
import os
from agents import manager_agent
from datetime import datetime
from typing import Optional
import time

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = manager_agent
        self.verbose = True

    def __call__(self, question: str, files: list[str] = None) -> str:
        print(f"Agent received question: {question[:50]}... with files: {files}")
        result = self.answer_question(question, files)
        print(f"Agent returning answer: {result}")
        time.sleep(60)
        return result
    def answer_question(self, question: str, task_file_path: Optional[str] = None) -> str:
        """
        Process a GAIA benchmark question and return the answer
        
        Args:
            question: The question to answer
            task_file_path: Optional path to a file associated with the question
            
        Returns:
            The answer to the question
        """
        try:
            if self.verbose:
                print(f"Processing question: {question}")
                if task_file_path:
                    print(f"With associated file: {task_file_path}")
            
            # Create a context with file information if available
            context = question
            file_content = None
            
            # If there's a file, read it and include its content in the context
            if task_file_path:
                try:
                    with open(task_file_path, 'r') as f:
                        file_content = f.read()
                    
                    # Determine file type from extension
                    import os
                    file_ext = os.path.splitext(task_file_path)[1].lower()
                    
                    context = f"""
Question: {question}

This question has an associated file. Here is the file content:

```{file_ext}
{file_content}
```

Analyze the file content above to answer the question.
"""
                except Exception as file_e:
                    context = f"""
Question: {question}

This question has an associated file at path: {task_file_path}
However, there was an error reading the file: {file_e}
You can still try to answer the question based on the information provided.
"""
            
            # Check for special cases that need specific formatting
            # Reversed text questions
            if question.startswith(".") or ".rewsna eht sa" in question:
                context = f"""
This question appears to be in reversed text. Here's the reversed version:
{question[::-1]}

Now answer the question above. Remember to format your answer exactly as requested.
"""
            
            # Add a prompt to ensure precise answers
            full_prompt = f"""{context}

When answering, provide ONLY the precise answer requested. 
Do not include explanations, steps, reasoning, or additional text.
Be direct and specific. GAIA benchmark requires exact matching answers.
For example, if asked "What is the capital of France?", respond simply with "Paris".
"""
            
            # Run the agent with the question
            answer = self.agent.run(full_prompt)
            
            # Clean up the answer to ensure it's in the expected format
            # Remove common prefixes that models often add
            answer = self._clean_answer(answer)
            
            if self.verbose:
                print(f"Generated answer: {answer}")
                
            return answer
        except Exception as e:
            error_msg = f"Error answering question: {e}"
            if self.verbose:
                print(error_msg)
            return error_msg
    
    def _clean_answer(self, answer: any) -> str:
        """
        Clean up the answer to remove common prefixes and formatting
        that models often add but that can cause exact match failures.
        
        Args:
            answer: The raw answer from the model
            
        Returns:
            The cleaned answer as a string
        """
        # Convert non-string types to strings
        if not isinstance(answer, str):
            # Handle numeric types (float, int)
            if isinstance(answer, float):
                # Format floating point numbers properly
                # Check if it's an integer value in float form (e.g., 12.0)
                if answer.is_integer():
                    formatted_answer = str(int(answer))
                else:
                    # For currency values that might need formatting
                    if abs(answer) >= 1000:
                        formatted_answer = f"${answer:,.2f}"
                    else:
                        formatted_answer = str(answer)
                return formatted_answer
            elif isinstance(answer, int):
                return str(answer)
            else:
                # For any other type
                return str(answer)
        
        # Now we know answer is a string, so we can safely use string methods
        # Normalize whitespace
        answer = answer.strip()
        
        # Remove common prefixes and formatting that models add
        prefixes_to_remove = [
            "The answer is ", 
            "Answer: ",
            "Final answer: ",
            "The result is ",
            "To answer this question: ",
            "Based on the information provided, ",
            "According to the information: ",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove quotes if they wrap the entire answer
        if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        return answer


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://github.com/ssgrummons/huggingface_final_assignment"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        files = [item.get("file_name")]
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text, files)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

import sys
from pathlib import Path

class Tee:
    def __init__(self, file_path):
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal_stdout = sys.__stdout__
        self.terminal_stderr = sys.__stderr__
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal_stdout.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal_stdout.isatty()


if __name__ == "__main__":
    
    # Redirect stdout and stderr
    log_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = f"./logs/output_{log_timestamp}.log"
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee

    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)