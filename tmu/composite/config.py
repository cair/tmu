from pydantic import BaseModel, Field
from typing import Union

class TMBaseConfig(BaseModel):
    pass

class TMClassifierConfig(TMBaseConfig):
    number_of_clauses: int = Field(10, description="Number of Clauses")
    T: int = Field(500, description="T value")
    s: float = Field(10.0, description="s value")
    max_included_literals: int = Field(32, description="Max Included Literals")
    platform: str = Field("GPU", description="Device to use")
    weighted_clauses: bool = Field(True, description="Use Weighted Clauses or not")
    patch_dim: Union[tuple, None] = Field(None, description="Size of Patch")
