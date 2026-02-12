from Rag.Workflow import Execute_Workflow
from Rag.models import FileInfo


#-----------------------------------------------------------------------------------------------
file_info = FileInfo(
        file_path="C:\\Users\\karti\\Downloads\\python-basics-a-practical-introduction-to-python-3.pdf",
        file_type="pdf"
)
Execute_Workflow(file_info)




