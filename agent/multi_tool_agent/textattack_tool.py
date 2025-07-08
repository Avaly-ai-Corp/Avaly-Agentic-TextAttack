from .app.textattack_wrapper import Finding
from pydantic import BaseModel
import pandas as pd


class TAParams(BaseModel):
    model_name: str
    dataset_name: str
    recipe_name: str
    n: int


ALLOWED_RECIPES = [
    "textfooler",
    "bae",
    "pwws",
    "deepwordbug",
    "pruthi",
    "alzantot",
    "faster-alzantot",
    "iga",
    "pso",
    "textbugger",
    "checklist",
    "clare",
    "kuleshov",
    "input-reduction",
]


def run_attack(
    model_name: str,
    dataset_name: str,
    recipe_name: str,
    n: int,
) -> dict:
    """
    Runs an attack.
    Required arguments:
        model_name: str,
        dataset_name: str,
        recipe_name: str,
        n: int,
    """
    if recipe_name not in ALLOWED_RECIPES:
        return {
            "error": f"Recipe {recipe_name} is not allowed. Please use one of the following: {ALLOWED_RECIPES}"
        }

    runner = Finding(
        model_from_hf=model_name,
        dataset_from_hf=dataset_name,
        recipe=recipe_name,
        n=n,
        model_from_local=None,
        dataset_from_local=None,
    )
    result = runner.run()
    print("\n\nresult from textattack ->>>>\n\n", result)
    print("type of result ->>>>\n\n", type(result))
    return result

def recipes() -> dict:
    """
    Returns a JSON-serializable dict with a 'recipes' key listing
    all allowed TextAttack recipes. No arguments required.

     WHEN TO CALL THIS TOOL:
      - If the user asks “What recipes are available?”
      - If the user asks “Which recipes can I use?”
      - If the user asks a question similar to the above
      - If the user tries to run an attack without specifying a recipe_name.

    The LLM should not answer recipe-related queries itself but must always call this function

    Example return: {"recipes": ["textfooler", "bae", ...]}
    """
    return {"recipes": ALLOWED_RECIPES}
