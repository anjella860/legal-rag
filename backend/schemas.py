from pydantic import BaseModel
from typing import Optional, List

class AskRequest(BaseModel):
    question:     str
    document_ids: Optional[List[int]] = None

class QAResponse(BaseModel):
    answer:  str
    sources: List[dict]
