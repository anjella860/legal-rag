from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel

class ApiResponse(BaseModel):
    success:   bool
    message:   Optional[str] = None
    data:      Optional[Any] = None
    error:     Optional[str] = None
    timestamp: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @classmethod
    def ok(cls, data: Any = None, message: str = "요청이 성공적으로 처리되었습니다."):
        return cls(success=True, message=message, data=data)

    @classmethod
    def fail(cls, error: str):
        return cls(success=False, error=error)
