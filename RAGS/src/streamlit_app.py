import streamlit as st
from constant import (
    hostname,
    portno,
    receipe_collection_name,
    search_params,
    emb_colname, 
    response_field,
    emb_model_name
)
from milvus_db import (
    connect_milvus_database,
    search_results,
)
from receipe_generator import ReceipeGenerator
from sentence_transformers import SentenceTransformer


def generate_query_vector(model_name, query):
    model = SentenceTransformer(model_name)
    query_vector = model.encode(query)
    return query_vector


def get_retreived_context_response(query, retrieved_results, 
                                   cosine_threshold):
    top3_context = []
    for result in retrieved_results[0]:
        if result.distance < cosine_threshold:
            context = result.entity.get('TranslatedInstructions')
            context.replace("\\xa0", " ")
            context.replace("\xa0", " ")
            top3_context.append(context)
    if len(top3_context) == 0:
        return "Results Not Found"
    else:
        top3_receipes = "\n\n".join(top3_context)
        receipe_generator = ReceipeGenerator()
        response = receipe_generator.generate_receipe(query, top3_receipes)
        return response


def get_receipe(query, cosine_threshold):
    query_vector = generate_query_vector(model_name=emb_model_name, 
                                        query = query)
    connect_milvus_database(hostname, portno)
    retrieved_results = search_results(query_vector, 
                                       receipe_collection_name,
                                       emb_colname, search_params,
                                       response_field)
    receipe_response = get_retreived_context_response(query, retrieved_results,
                                                      cosine_threshold)
    return receipe_response
    

if __name__ == "__main__":
    st.title("Indian Cuisine Receipe Generator")
    query = st.text_area("Ask For Indian Cuisine Receipe")
    # Button to send text to the server
    if st.button('Generate'):
        receipe = get_receipe(query, 1.0)
        st.write('Processed Result:', receipe)
