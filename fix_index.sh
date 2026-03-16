#!/bin/bash
echo "Rebuilding vector index with HNSW..."
docker exec -it medai-postgres psql -U medai -d medical_insights << 'SQLEOF'
-- Drop old ivfflat index
DROP INDEX IF EXISTS summary_vectors_embedding_idx;

-- Create HNSW index — better accuracy at small scale (499 vectors)
CREATE INDEX summary_vectors_embedding_idx
ON summary_vectors
USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- Verify
SELECT COUNT(*) as total_vectors FROM summary_vectors;
SQLEOF
echo "Done!"
