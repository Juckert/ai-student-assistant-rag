from typing import List, Dict, Any


class PostgresChunkSource:
    """Loads pre-chunked data from PostgreSQL qa_chunks and document_chunks tables."""

    def load(self) -> List[Dict[str, Any]]:
        from app.database.database import get_db_session, QAChunk, DocumentChunk

        chunks = []

        with get_db_session() as session:
            qa_rows = session.query(QAChunk).all()
            for i, row in enumerate(qa_rows):
                text = f"Вопрос: {row.question}\nОтвет: {row.answer}"
                chunks.append({
                    "chunk_id": f"qa_{row.id}",
                    "text": text,
                    "metadata": {
                        "document_id": "qa_chunks",
                        "chunk_index": i,
                        "original_text": text,
                        "source": "qa_chunks",
                        "chunk_type": "qa",
                    },
                })

            doc_rows = session.query(DocumentChunk).all()
            for i, row in enumerate(doc_rows):
                chunks.append({
                    "chunk_id": f"doc_{row.id}",
                    "text": row.text,
                    "metadata": {
                        "document_id": row.filename,
                        "chunk_index": i,
                        "original_text": row.text,
                        "source": row.filename,
                        "chunk_type": "document",
                    },
                })

        return chunks
