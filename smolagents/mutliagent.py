import os

from dotenv import load_dotenv
from smolagents import OpenAIModel, ToolCallingAgent, tool


@tool
def translate_skills(current_skills: str) -> str:
    """Translate existing skills into data-science resume language.

    Args:
        current_skills: The user's existing skills.
    """
    return (
        f"Translate these skills for a data-science resume: {current_skills}. "
        "Highlight analytics, experimentation, reporting, stakeholder communication, "
        "and measurable results."
    )


@tool
def improve_resume_layout(resume_text: str) -> str:
    """Suggest a simple, readable resume structure.

    Args:
        resume_text: The resume content to organize.
    """
    return (
        "Use this order: Summary, Skills, Experience, Projects, Education. "
        "Keep bullets concise and start each bullet with an action verb. "
        f"Resume content to organize: {resume_text}"
    )


@tool
def check_company_fit(company_name: str, target_role: str) -> str:
    """Give a simple starting point for evaluating a company and role.

    Args:
        company_name: The company being considered.
        target_role: The job role being considered.
    """
    return (
        f"Evaluate {company_name} for a {target_role} role by checking the job "
        "requirements, analytics stack, mentorship, team structure, and growth path."
    )


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

token = os.getenv("OPENROUTER_API_KEY")
if not token:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to smolagents/.env or set it "
        "in your PowerShell session before running this script."
    )

model = OpenAIModel(
    model_id="openrouter/free",
    api_base="https://openrouter.ai/api/v1",
    api_key=token,
)

resume_agent = ToolCallingAgent(
    tools=[translate_skills, improve_resume_layout],
    model=model,
    instructions="You help users improve resumes for career changes.",
    name="resume_agent",
    description="Improves resume skills and structure.",
)

company_agent = ToolCallingAgent(
    tools=[check_company_fit],
    model=model,
    instructions="You help users evaluate companies and job opportunities.",
    name="company_agent",
    description="Evaluates company fit for a target role.",
)

career_manager = ToolCallingAgent(
    tools=[],
    model=model,
    instructions=(
        "You coordinate resume and company advice. Ask the specialist agents for "
        "help, then combine their answers into a short practical action plan."
    ),
    managed_agents=[resume_agent, company_agent],
)


if __name__ == "__main__":
    result = career_manager.run(
        "I want to switch from marketing to data science. "
        "Help me improve my resume and evaluate whether a data-science role "
        "at a company called Acme would be a good fit."
    )
    print(result)
