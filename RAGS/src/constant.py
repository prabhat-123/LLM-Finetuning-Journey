hostname = "localhost"
portno = 19530
receipe_collection_name = "indian_receipe_collections"
emb_colname = "receipe_instructions_embeddings"
response_field = ["TranslatedInstructions"]
search_params = {
    "metric_type": "L2",
    "offset": 0, 
    "ignore_growing": False,
    "params": {"nprobe": 5}   
}
emb_model_name = 'sentence-transformers/all-MiniLM-L6-v2'