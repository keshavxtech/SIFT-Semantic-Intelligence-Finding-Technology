export type DocumentType = 'pdf' | 'docx' | 'txt' | 'md';
export type ProcessingStatus = 'indexed' | 'processing' | 'queued' | 'error';
export type DocumentTag = 'Financial' | 'Technical' | 'Security' | 'Legal' | 'Research' | 'Compliance';

export interface DocumentItem {
  id: string;
  name: string;
  size: string;
  sizeBytes?: number;
  type: DocumentType;
  uploadedAt: string;
  chunksCount: number;
  vectorCount: number;
  tokenCount: number;
  status: ProcessingStatus;
  tag: DocumentTag;
  hash: string;
  author?: string;
  description: string;
  pageCount: number;
  rawTextPreview?: string;
}

export interface ProcessingActivity {
  id: string;
  docId: string;
  docName: string;
  stage: 'OCR & Layout' | 'Chunking' | 'Embedding' | 'Indexing';
  progress: number;
  timestamp: string;
  status: 'completed' | 'in_progress' | 'queued';
  eta?: string;
}

export interface Citation {
  id: string;
  docId: string;
  docName: string;
  pageNumber: number;
  pageStart?: number | null;
  pageEnd?: number | null;
  distance?: number;
  chunkIndex?: number;
  chunkId: string;
  passage: string;
  similarityScore: number;
  sectionTitle: string;
  highlightCoordinates?: {
    top: number;
    left: number;
    width: number;
    height: number;
  };
}

export interface ChatSession {
  id: string;
  sessionId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  confidenceScore?: number;
  retrievalMode?: 'hybrid' | 'dense' | 'bm25';
  latencyMs?: number;
  tokensProcessed?: number;
}

export interface SearchResult {
  id: string;
  docId: string;
  docName: string;
  docType: DocumentType;
  uploadedAt: string;
  pageNumber: number;
  pageStart?: number | null;
  pageEnd?: number | null;
  chunkId: string;
  chunkIndex?: number;
  passage: string;
  score: number;
  similarityScore?: number;
  matchHighlights: string[];
  sectionTitle: string;
  tag: DocumentTag;
  tokenCount: number;
  cosineDistance?: number;
  distance?: number;
}

export interface SystemMetric {
  label: string;
  value: string;
  subValue: string;
  change?: string;
  isPositive?: boolean;
}

export type ActiveTab = 'dashboard' | 'library' | 'chat' | 'search';
