/**
 * SIFT — Centralized API Wire Models & Schemas
 * 
 * Typed TypeScript definitions reflecting the exact FastAPI backend Pydantic
 * schemas from app/schemas/ and app/models/.
 * Includes both canonical snake_case backend fields and compatibility aliases.
 */

// ============================================================================
// 1. Common Enums & Primitives
// ============================================================================

export type DocumentType = 'pdf' | 'txt' | 'md' | 'docx';

export type ProcessingStatus = 'processing' | 'indexed' | 'failed' | 'pending';

export type DocumentTag =
  | 'Technical'
  | 'Financial'
  | 'Security'
  | 'Legal'
  | 'Research'
  | 'Compliance';

export type RetrievalMode = 'dense' | 'hybrid' | 'bm25';

// ============================================================================
// 2. Document Models
// ============================================================================

export interface DocumentResponse {
  id: string;
  original_filename: string;
  stored_filename: string;
  name: string;
  file_type: DocumentType;
  type: DocumentType;
  file_size: number;
  size_bytes: number;
  size: string;
  upload_timestamp: string;
  uploaded_at: string;
  processing_status: ProcessingStatus;
  status: ProcessingStatus;
  page_count: number;
  error_message?: string | null;
  hash: string;
  tag: DocumentTag;
  author?: string | null;
  description: string;
  raw_text_preview?: string | null;
  extracted_text?: string | null;
  chunks_count: number;
  vector_count: number;
  token_count: number;
}

export interface DocumentDetailResponse extends DocumentResponse {
  extracted_text: string;
}

export interface DocumentListResponse {
  total: number;
  documents: DocumentResponse[];
}

export interface DocumentListParams {
  tag?: string;
  status?: string;
  search?: string;
}

export interface DocumentUploadParams {
  file: File | Blob;
  tag?: DocumentTag | string;
  description?: string;
}

export interface DocumentRenameRequest {
  name: string;
}

// ============================================================================
// 3. Chunking & Embeddings Models
// ============================================================================

export interface ChunkResponse {
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  character_count: number;
  token_count: number;
  page_start?: number | null;
  page_end?: number | null;
  metadata: Record<string, unknown>;
}

export interface DocumentChunksResponse {
  document_id: string;
  total_chunks: number;
  chunk_size: number;
  chunk_overlap: number;
  chunks: ChunkResponse[];
}

export interface ChunkQueryParams {
  chunk_size?: number;
  chunk_overlap?: number;
  minimum_chunk_size?: number;
}

export interface EmbeddingGenerationResponse {
  document_id: string;
  model_name: string;
  embedding_dimension: number;
  chunks_processed: number;
  embeddings_created: number;
  status: string;
}

export interface EmbeddingSummaryResponse {
  document_id: string;
  model_name: string;
  embedding_dimension: number;
  total_embeddings: number;
  status: string;
  normalized: boolean;
}

// ============================================================================
// 4. Indexing Models
// ============================================================================

export interface IndexDocumentResponse {
  document_id: string;
  status: string;
  indexed_chunks: number;
  total_chunks: number;
  embedding_model: string;
  embedding_dimension: number;
  collection_name: string;
  message: string;
}

export interface IndexStatusResponse {
  document_id: string;
  status: string;
  is_indexed: boolean;
  indexed_chunks: number;
  total_chunks: number;
  embedding_model?: string | null;
  embedding_dimension?: number | null;
  collection_name: string;
}

// ============================================================================
// 5. Semantic Search Models
// ============================================================================

export interface SearchRequest {
  query: string;
  document_ids?: string[];
  top_k?: number;
  limit?: number;
  threshold?: number;
  mode?: RetrievalMode | string;
}

export interface SearchResultItem {
  // Canonical Phase 5 backend fields
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  content: string;
  source_filename: string;
  page_start?: number | null;
  page_end?: number | null;
  distance: number;
  similarity_score: number;

  // Frontend compatibility & UI alignment aliases
  id: string;
  docId: string;
  docName: string;
  docType: DocumentType;
  pageNumber: number;
  chunkId: string;
  passage: string;
  score: number;
  cosineDistance: number;
  tokenCount: number;
  matchHighlights: string[];
  sectionTitle: string;
  tag: DocumentTag;
}

export interface SearchResponse {
  query: string;
  total_matches: number;
  execution_time_ms: number;
  mode: string;
  metric: string;
  results: SearchResultItem[];
}

// ============================================================================
// 6. RAG Chat & Conversational Models
// ============================================================================

export interface CitationSchema {
  // Canonical Phase 6 backend fields
  id: string;
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  source_filename: string;
  page_start?: number | null;
  page_end?: number | null;
  passage: string;
  distance: number;
  similarity_score: number;

  // Frontend compatibility & UI alignment aliases
  docId: string;
  docName: string;
  pageNumber?: number | null;
  chunkId: string;
  similarityScore: number;
  sectionTitle: string;
  highlightCoordinates?: Record<string, number> | null;
}

export interface ChatMessageSchema {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string | null;
}

export interface ChatMessageResponse {
  message_id: string;
  session_id: string;
  role: string;
  content: string;
  citations: CitationSchema[];
  created_at: string;
}

export interface ChatRequest {
  message?: string;
  query?: string;
  session_id?: string;
  document_ids?: string[];
  top_k?: number;
  retrieval_mode?: RetrievalMode | string;
  history?: ChatMessageSchema[];
}

export interface ChatResponse {
  id: string;
  session_id: string;
  role: string;
  answer: string;
  content: string;
  timestamp: string;
  citations: CitationSchema[];
  retrieved_chunks: Record<string, unknown>[];
  provider: string;
  model: string;
  retrieval_metadata: Record<string, unknown>;
  generation_metadata: Record<string, unknown>;
  confidence_score: number;
  latency_ms: number;
  tokens_processed: number;
  retrieval_mode: string;
}

export interface ChatSessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatSessionDetail {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessageResponse[];
}

// ============================================================================
// 7. System Health & Root Status
// ============================================================================

export interface StorageHealth {
  storage_dir: string;
  free_space_mb: number;
  total_space_gb: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  documents_indexed: number;
  storage: StorageHealth;
}

export interface RootStatusResponse {
  service: string;
  version: string;
  status: string;
  docs: string;
  health: string;
}

// ============================================================================
// 8. API Error Structures
// ============================================================================

export interface ValidationErrorItem {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

export interface ApiErrorResponse {
  error?: boolean;
  message?: string;
  details?: Record<string, unknown>;
  path?: string;
  detail?: string | ValidationErrorItem[];
}
