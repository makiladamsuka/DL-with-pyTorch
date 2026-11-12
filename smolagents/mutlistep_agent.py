import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from smolagents import ActionStep, CodeAgent, OpenAIModel, PlanningStep, tool


@tool
def get_opponent_profile() -> dict[str, Any]:
    """Return scouting information for the opposing point guard.

    Returns:
        A profile containing the player's scoring average and tendencies.
    """
    return {
        "position": "point guard",
        "points_per_game": 25,
        "strengths": ["pull-up shooting", "high pick-and-roll"],
        "tendencies": ["drives right", "looks for the corner pass when doubled"],
    }


def action_callback(agent_step: ActionStep, agent: CodeAgent) -> None:
    """Print progress after each action step."""
    print(f"\nAction step {agent_step.step_number}")
    if agent_step.is_final_answer:
        token_usage = agent_step.token_usage
        if token_usage is not None:
            print(f"Total tokens used: {token_usage.total_tokens}")


def planning_callback(agent_step: PlanningStep, agent: CodeAgent) -> None:
    """Print the current plan whenever the agent replans."""
    print("\nAGENT PLAN")
    print("=" * 50)
    plan = agent_step.plan
    print(plan[:500] + ("..." if len(plan) > 500 else ""))


def create_agent() -> CodeAgent:
    """Build a CodeAgent configured to replan throughout the task."""
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env or set it in "
            "your PowerShell session before running this script."
        )

    model = OpenAIModel(
        model_id="openrouter/free",
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return CodeAgent(
        tools=[get_opponent_profile],
        model=model,
        planning_interval=2,
        step_callbacks={
            PlanningStep: planning_callback,
            ActionStep: action_callback,
        },
    )


if __name__ == "__main__":
    agent = create_agent()
    task = (
        "Create a practical defensive game plan for stopping the opposing point guard. "
        "First inspect the scouting profile with the available tool. Then reason through "
        "the primary coverage, pick-and-roll adjustments, help rotations, and late-game "
        "fallback. Return a concise plan with roles for the on-ball defender and help "
        "defenders."
    )
    result = agent.run(task)
    print("\nFINAL ANSWER\n" + str(result))

