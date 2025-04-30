# Hugging Face AI Agents Course Final Assignment

This is my final assignment for the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction).

## Dependencies

This does require some hardware to run on.  I am running it on an Apple Silicon M1 MAX.

The following needs to be installed on your system:
- `ollama` (for running models)
- `ffmpeg` (for saving YouTube videos for the agent to review)
- `poetry` (for managing dependencies)

I have been experimenting with multiple models as the core agent.  So far, I have only been able to pass the GAIA benchmark using large models from Google.  These models require an API key from [Google AI Studio](https://aistudio.google.com/).
- `gemini/gemini-2.5-pro-preview-03-25` Most expensive model from Google, but returned 14/20 score
- `gemini/gemini-2.0-flash-exp`  Available for free, but rate limiting applies.


I have tried to offload some AI processing to smaller models from Ollama.  I use `cogito` to review the answer for correctness, and `granite3.3` to evaluate whether the answer meets the formatting requirements.  I've used `gemma3` to evaaluate images.
```bash
ollama pull granite3.3:8b
ollama pull cogito:14b
ollama pull gemma3:12b
```

## Running the solution

To run the agent without submitting solutions.  
```bash
poetry install
poetry run python agents.py
```

## Lessons Learned

1. After attempting this all using small models running on Ollama, I had to pivot to using a large model from a Cloud Provider.  The GAIA tasks are too complex for a small model.  
2. I initially tried running this entirely on Ollama with the largest Multi-Model Tool-using model my hardware could support: [mistral-small3.1](https://ollama.com/library/mistral-small3.1).  However, it was not able to handle the complexity of GAIA level 1 questions, it wasn't until I used Google's `gemini-2.0-flash-exp` as it produced more accurate answers at a cost my wallet could handle (i.e. Free).
4. I have offloaded smaller tasks to smaller models.  So while the core agent handles most of the heavy lifting, audio transcription, image recognition, and answer validation use smaller models at a lower cost.
5. I had to modify the `LiteLLMModel` class with a 5 second sleep timer to prevent Rate Limiting errors.
6. Sometimes answers were hit and miss.  AI Agents are non-deterministic, with is both a strength and a weakness.  Having another lighter model evaluate the answers provided helped guide the agent to rethink the problem.
7. After initially trying a multi-agent solution, I opted for simplicity and built one agent.  However, I could still use AI models in tool calls for smaller tasks, such as transcription or object recognition.  
8. `smolagents` does not support the `control` role needed to enable thinking for `granite3.3`.  I was able to customize a `ThinkingLiteLLMModel` class to toggle on thinking, but opted against leveraging granite for those purposes in the final submission.