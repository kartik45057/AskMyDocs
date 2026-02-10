from Rag.Workflow import Execute_Workflow
from Rag.models import FileInfo


#-----------------------------------------------------------------------------------------------
file_info = FileInfo(
        file_path="C:\\Users\\karti\\Downloads\\python-basics-a-practical-introduction-to-python-3.pdf",
        file_type="pdf"
)
result = Execute_Workflow("explain numpy", file_info)
print("result ", result["page_content"])



