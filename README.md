# Hugging Face AI Agents Course Final Assignment

This is my final assignment for the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction).

## Dependencies

This does require some hardware to run on.  I am running it on an Apple Silicon M1 MAX.

The following needs to be installed on your system
- `ollama` (for running models)
- `ffmpeg` (for saving YouTube videos for the agent to review)
- `poetry` (for managing dependencies)

The following models are used
- `gemini/gemini-2.0-flash-exp`  Available for free, but requires API key from [Google AI Studio](https://aistudio.google.com/)
- `granite3.3:8b` Lightweight model to ensure the final answer is formatted correctly.  
- `cogito:14b` Used to evaluate the final answer provided by the agent.


Download models from Ollama:
```bash
ollama pull granite3.3:8b
ollama pull cogito:14b
```

## Running the solution

To run the agent without submitting solutions.  
```bash
poetry install
poetry run python agents.py
```

## Lessons Learned

1. Start with big models, make the solution work, then optimize from there.
2. I initially tried running this entirely on Ollama with the largest Multi-Model Tool-using model my hardware could support: [mistral-small3.1](https://ollama.com/library/mistral-small3.1).  However, it was not able to handle the complexity of GAIA level 1 questions, so I since upgraded to use Google's `gemini-2.0-flash-exp` as it produced more accurate answers at a cost my wallet could handle (i.e. Free)
3. I had to modify the `LiteLLMModel` class with a 5 second sleep timer to prevent Rate Limiting errors.
4. Sometimes answers were hit and miss.  AI Agents are non-deterministic, with is both a strength and a weakness.  Having another lighter model evaluate the answers provided helped guide the agent to rethink the problem.
5. After initially trying a multi-agent solution, I opted for simplicity and built one agent.  However, I could still use AI models in tool calls for smaller tasks, such as transcription or object recognition.  
6. `smolagents` does not support the `control` role needed to enable thinking for `granite3.3`.  I was able to customize a `ThinkingLiteLLMModel` class to toggle on thinking, but opted against leveraging granite for those purposes in the final submission.