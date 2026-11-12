import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel, tool


ORDERS: dict[str, list[str]] = {
    "T5": ["Latte", "Iced tea"],
    "T6": ["Cappuccino"],
}


@tool
def lookup_orders(table_id: str) -> list[str]:
    """Look up the current drink orders for a café table.

    Args:
        table_id: The table's identifier, such as "T5".

    Returns:
        The drink names currently ordered for that table.
    """
    normalized_table_id = table_id.strip().upper()
    if normalized_table_id not in ORDERS:
        raise ValueError(f"No orders found for table {normalized_table_id}.")
    return ORDERS[normalized_table_id]


@tool
def generate_order_id(table_id: str, drink_name: str) -> str:
    """Generate a unique order ID for a café drink order.

    Args:
        table_id: The table's identifier, such as "T5".
        drink_name: The name of the drink, such as "Latte".

    Returns:
        An ID containing the table, drink, and current timestamp.
    """
    normalized_table_id = table_id.strip().upper()
    normalized_drink_name = drink_name.strip().lower().replace(" ", "-")
    if not normalized_table_id or not normalized_drink_name:
        raise ValueError("Both table_id and drink_name are required.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{normalized_table_id}_{normalized_drink_name}_{timestamp}"


def create_agent() -> CodeAgent:
    """Build the tool-calling agent from the repository's local .env file."""
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
        tools=[lookup_orders, generate_order_id],
        model=model,
    )


def main() -> None:
    agent = create_agent()
    task = (
        "For table T5, look up the current drink orders. Generate a unique order ID "
        "for each drink, then return a clear list pairing each drink with its order ID."
    )
    print(agent.run(task))


if __name__ == "__main__":
    main()
