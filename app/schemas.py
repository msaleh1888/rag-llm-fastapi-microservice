# app/schemas.py

from typing import List
from pydantic import BaseModel

class AskRequest(BaseModel):
    """
    Request body for the /ask endpoint.
    """
    query: str

class AskResponse(BaseModel):
    """
    Response body for the /ask endpoint.
    """
    answer: str
    contexts: List[str]