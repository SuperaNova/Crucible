from pydantic import BaseModel

class AppraisalResult(BaseModel):
    """Structured output from the Appraiser agent."""
    item_a: str
    item_b: str
    item_a_tags: list[str]
    item_b_tags: list[str]

class SmithingResult(BaseModel):
    """Structured output from the Master Smith agent."""
    fused_name: str
    reasoning: str
    image_prompt: str
