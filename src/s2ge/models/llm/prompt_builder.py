"""Legacy plain-text instruction assembly helper."""


def build_instruction_prompt(question: str, desc: str = "") -> str:
    """Prepend an optional graph description to a question."""
    return f"{desc}\n{question}" if desc else question
