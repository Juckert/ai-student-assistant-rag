from dotenv import load_dotenv

def get_agent(mode: str = "default"):
    """
    Factory function to get an instance of the RAG agent based on the specified mode.
    This uses local imports to prevent circular dependencies.
    """
    from rag_system.agent.loop import Agent
    from rag_system.utils.ollama_client import OllamaClient
    from rag_system.main import PIPELINE_CONFIGS, OLLAMA_CONFIG, LLM_BACKEND, WATSONX_CONFIG

    load_dotenv()
    
    # Initialize the appropriate LLM client based on backend configuration
    if LLM_BACKEND.lower() == "watsonx":
        from rag_system.utils.watsonx_client import WatsonXClient
        
        if not WATSONX_CONFIG["api_key"] or not WATSONX_CONFIG["project_id"]:
            raise ValueError(
                "Watson X configuration incomplete. Please set WATSONX_API_KEY and WATSONX_PROJECT_ID "
                "environment variables."
            )
        
        llm_client = WatsonXClient(
            api_key=WATSONX_CONFIG["api_key"],
            project_id=WATSONX_CONFIG["project_id"],
            url=WATSONX_CONFIG["url"]
        )
        llm_config = WATSONX_CONFIG
    else:
        llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
        llm_config = OLLAMA_CONFIG
    
    config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['default'])
    
    if 'storage' not in config:
        config['storage'] = {
            'db_path': 'lancedb',
            'text_table_name': 'text_pages_default',
            'image_table_name': 'image_pages'
        }
    
    agent = Agent(
        pipeline_configs=config, 
        llm_client=llm_client, 
        ollama_config=llm_config
    )
    return agent

def index_from_postgres(mode: str = "default") -> int:
    """
    Reads all chunks from PostgreSQL (qa_chunks + document_chunks),
    embeds them with Qwen, and writes the result into LanceDB for retrieval.
    Overwrites the existing table on every call (full re-sync).
    Returns the number of indexed chunks.
    """
    import numpy as np
    from rag_system.ingestion.postgres_source import PostgresChunkSource
    from rag_system.indexing.representations import QwenEmbedder, EmbeddingGenerator
    from rag_system.indexing.embedders import LanceDBManager, VectorIndexer
    from rag_system.main import PIPELINE_CONFIGS, EXTERNAL_MODELS

    config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS["default"])
    storage = config["storage"]
    table_name = storage["text_table_name"]

    chunks = PostgresChunkSource().load()
    if not chunks:
        raise ValueError("No chunks found in PostgreSQL. Run ingestion first.")

    embedder = QwenEmbedder(model_name=EXTERNAL_MODELS["embedding_model"])
    generator = EmbeddingGenerator(embedding_model=embedder)
    embeddings_list = generator.generate(chunks)
    embeddings = np.array(embeddings_list, dtype=np.float32)

    manager = LanceDBManager(storage["lancedb_uri"])
    db = manager.db
    if table_name in db.table_names():
        db.drop_table(table_name)

    VectorIndexer(manager).index(table_name, chunks, embeddings)

    tbl = manager.get_table(table_name)
    tbl.create_fts_index("text", replace=True)

    print(f"Indexed {len(chunks)} chunks from PostgreSQL into LanceDB table '{table_name}'.")
    return len(chunks)


def get_indexing_pipeline(mode: str = "default"):
    """
    Factory function to get an instance of the Indexing Pipeline.
    """
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.main import PIPELINE_CONFIGS, OLLAMA_CONFIG, LLM_BACKEND, WATSONX_CONFIG
    from rag_system.utils.ollama_client import OllamaClient

    load_dotenv()
    
    # Initialize the appropriate LLM client based on backend configuration
    if LLM_BACKEND.lower() == "watsonx":
        from rag_system.utils.watsonx_client import WatsonXClient
        
        if not WATSONX_CONFIG["api_key"] or not WATSONX_CONFIG["project_id"]:
            raise ValueError(
                "Watson X configuration incomplete. Please set WATSONX_API_KEY and WATSONX_PROJECT_ID "
                "environment variables."
            )
        
        llm_client = WatsonXClient(
            api_key=WATSONX_CONFIG["api_key"],
            project_id=WATSONX_CONFIG["project_id"],
            url=WATSONX_CONFIG["url"]
        )
        llm_config = WATSONX_CONFIG
    else:
        llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
        llm_config = OLLAMA_CONFIG
    
    config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['default'])
    
    return IndexingPipeline(config, llm_client, llm_config)     