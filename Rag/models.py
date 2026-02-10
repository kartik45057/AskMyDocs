from pydantic import BaseModel, Field, field_validator
from typing import Annotated

class Query(BaseModel):
    query: Annotated[str, Field(description="search query from the user")]
    k: Annotated[int, Field(default=4, gt=-1, description="Numbers of similar documents from the vector store")]

class FileInfo(BaseModel):
    file_path: Annotated[str, Field(description="path to the file")]
    file_type: Annotated[str, Field(description="type of the file(pdf, doc, text)")]

    @field_validator("file_type")
    @classmethod
    def convert_to_lowercase(cls, value):
        return value.lower()