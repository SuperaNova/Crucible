from pydantic import BaseModel

class AppraisalResult(BaseModel):
    """
    Structured output from the BLIP Vision-Encoder-Decoder Appraiser.
    Fields match the dict returned by BLIPAppraiser.appraise() and are
    injected into ADK session state before the Master Smith runs.
    """
    item_a: str
    item_b: str
    item_a_tags: list[str]
    item_b_tags: list[str]

class SmithingResult(BaseModel):
    """Structured output from the Master Smith agent."""
    fused_name: str
    reasoning: str
    image_prompt: str
