from pydantic import BaseModel

class ClipTxtRequest(BaseModel):
    text: str

