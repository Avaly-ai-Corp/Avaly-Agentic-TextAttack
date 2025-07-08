from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .textattack_tool import run_attack, recipes
from google.genai import types

model = LiteLlm(
    model="ollama_chat/magistral:24b-small-2506-q8_0",
    api_base="http://host.docker.internal:11434",
)

system_prompt = """
You are an assistant specialized in text-based adversarial attacks.
When the user asks you to run an attack, collect exactly these four parameters:
  • model_name  
  • dataset_name  
  • recipe_name  
  • n  (number of examples)

If any are missing, ask only for the missing ones (e.g. “Please provide the missing parameter(s): recipe_name, n.”).  
When the user explicitly requests available recipes (or omits recipe_name), invoke the recipes tool.  
When all parameters are present, invoke the run_attack tool.  
For any other questions, reply normally and conversationally.
"""

root_agent = LlmAgent(
    name="attack_agent",
    model=model,
    tools=[run_attack, recipes],
    instruction=system_prompt,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.15
    ),
)
