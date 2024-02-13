from pymilvus import (
    connections,
    Collection,
    utility,
)


def connect_milvus_database(hostname: str = "localhost", port: int = 19530):
    """
    Establishes a connection to the Milvus database.

    Args:
        hostname (str, optional): The hostname of the Milvus server. Defaults to "localhost".
        port (int, optional): The port number of the Milvus server. Defaults to 19530.
    """
    connections.connect(alias="default", hostname=hostname, port=port)


def create_index(
    collection: Collection,
    vector_colname: str,
    index_type: str = "IVF_FLAT",
    metric_type: str = "L2",
    nlist: int = 128,
):
    """
    Creates an index on a specified vector column in the collection.

    Args:
        collection (Collection): The collection object.
        vector_colname (str): The name of the vector column to create the index on.
        index_type (str, optional): The type of index to create. Defaults to 'IVF_FLAT'.
        metric_type (str, optional): The metric type of the index. Defaults to 'L2'.
        nlist (int, optional): The number of clusters for the IVF index. Defaults to 128.
    """
    index_params = {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": {"nlist": nlist},
    }
    collection.create_index(vector_colname, index_params)


def insert_data(collection: Collection, data: list):
    """
    Inserts data into the specified collection.

    Args:
        collection (Collection): The collection object.
        data (list): The data to be inserted into the collection.
    """
    collection.insert(data)
    collection.flush()


def delete_collection(collection_name: str):
    """
    Deletes a collection from the Milvus database.

    Args:
        collection_name (str): The name of the collection to be deleted.
    """
    utility.drop_collection(collection_name)


def search_results(query_vector: list, collection_name: str,
                   emb_colname, search_params: dict,
                   output_fields: list):
    collection = Collection(collection_name)
    collection.load()
    results = collection.search(
    data = [query_vector],
    anns_field = emb_colname,
    param = search_params,
    limit=3,
    expr = None,
    output_fields = output_fields,
    consistency_level = "Strong"
    )
    return results
