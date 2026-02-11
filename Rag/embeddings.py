from langchain_huggingface import HuggingFaceEmbeddings


#initialize embedding model
"""
connects to huggingface.io | Downloads model(first time only) | caches locally into ~/.cache/huggingface/
loads model into memory
configures where the model runs | device = cpu(for running on cpu), device = cuda(for runnning on gpu)
'normalize_embeddings': True | to make sure that all vectors have equal length for better similarity comparision
"""
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
