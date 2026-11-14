/**
 * SIFT — API Model to UI Model Adapters
 * 
 * Provides explicit, deterministic mappers that convert backend wire payloads
 * into the camelCase data structures expected by the approved frontend UI.
 */

import {
  DocumentItem,
  SearchResult,
  Citation,
  ChatMessage,
  ChatSession,
  DocumentType as UiDocumentType,
  ProcessingStatus as UiProcessingStatus,
} from '@/types';
import {
  DocumentResponse,
  SearchResultItem,
  CitationSchema,
  ChatResponse,
  ChatMessageResponse,
  ChatSessionSummary,
} from './types';

/**
 * Formats an ISO 8601 timestamp string into a friendly relative or short display string.
 */
export function formatFriendlyDate(isoString?: string): string {
  if (!isoString) return 'Just now';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;

    const diffSeconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diffSeconds < 60) return 'Just now';
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    if (diffSeconds < 604800) return `${Math.floor(diffSeconds / 86400)}d ago`;

    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch {
    return isoString;
  }
}

/**
 * Maps a backend DocumentResponse to the frontend DocumentItem.
 */
export function toUiDocumentItem(doc: DocumentResponse): DocumentItem {
  // Normalize status to frontend ProcessingStatus enum ('indexed' | 'processing' | 'queued' | 'error')
  let uiStatus: UiProcessingStatus = 'processing';
  const rawStatus = (doc.status || doc.processing_status || '').toLowerCase();
  if (rawStatus === 'indexed') {
    uiStatus = 'indexed';
  } else if (rawStatus === 'failed') {
    uiStatus = 'error';
  } else if (rawStatus === 'pending') {
    uiStatus = 'queued';
  } else {
    uiStatus = 'processing';
  }

  return {
    id: doc.id,
    name: doc.name || doc.original_filename,
    size: doc.size,
    sizeBytes: doc.size_bytes ?? doc.file_size,
    type: doc.type || doc.file_type,
    uploadedAt: formatFriendlyDate(doc.uploaded_at || doc.upload_timestamp),
    chunksCount: doc.chunks_count ?? 0,
    vectorCount: doc.vector_count ?? 0,
    tokenCount: doc.token_count ?? 0,
    status: uiStatus,
    tag: doc.tag,
    hash: doc.hash,
    author: doc.author || 'Workspace User',
    description: doc.description || '',
    pageCount: doc.page_count ?? 1,
    rawTextPreview: doc.raw_text_preview || undefined,
  };
}

/**
 * Maps a backend SearchResultItem to the frontend SearchResult.
 */
export function toUiSearchResult(item: SearchResultItem): SearchResult {
  const pageStart = item.page_start ?? null;
  const pageEnd = item.page_end ?? null;
  let sectionTitle = item.sectionTitle || 'Passage';
  if (pageStart !== null) {
    sectionTitle = pageEnd !== null && pageEnd !== pageStart
      ? `Pages ${pageStart}–${pageEnd}`
      : `Page ${pageStart}`;
  }

  const filename = item.source_filename || item.docName || '';
  const ext = filename.split('.').pop()?.toLowerCase();
  const docType: UiDocumentType =
    item.docType === 'pdf' || ext === 'pdf'
      ? 'pdf'
      : item.docType === 'docx' || ext === 'docx'
      ? 'docx'
      : item.docType === 'md' || ext === 'md'
      ? 'md'
      : 'txt';
  const distance = item.distance ?? item.cosineDistance ?? 0;
  const score = item.similarity_score ?? item.score ?? Math.max(0, 1 - distance);

  return {
    id: item.id || `sr-${item.chunk_id}`,
    docId: item.docId || item.document_id,
    docName: item.docName || item.source_filename,
    docType,
    uploadedAt: 'Active Index',
    pageNumber: item.pageNumber ?? item.page_start ?? 1,
    pageStart,
    pageEnd,
    chunkId: item.chunkId || item.chunk_id,
    chunkIndex: item.chunk_index,
    passage: item.passage || item.content,
    score,
    similarityScore: score,
    matchHighlights: item.matchHighlights || [],
    sectionTitle,
    tag: item.tag || 'Technical',
    tokenCount: item.tokenCount ?? Math.max(1, Math.round((item.content || item.passage || '').length / 4)),
    cosineDistance: distance,
    distance,
  };
}

/**
 * Maps a backend CitationSchema to the frontend Citation model.
 */
export function toUiCitation(cit: CitationSchema): Citation {
  const pageStart = cit.page_start ?? (cit.pageNumber !== undefined && cit.pageNumber !== null ? cit.pageNumber : null);
  const pageEnd = cit.page_end ?? null;
  let sectionTitle = cit.sectionTitle || 'Source Reference';
  if (pageStart !== null) {
    sectionTitle = pageEnd !== null && pageEnd !== pageStart
      ? `Pages ${pageStart}–${pageEnd}`
      : `Page ${pageStart}`;
  }

  return {
    id: cit.id,
    docId: cit.docId || cit.document_id,
    docName: cit.docName || cit.source_filename,
    pageNumber: cit.pageNumber ?? cit.page_start ?? 1,
    pageStart,
    pageEnd,
    distance: cit.distance,
    chunkId: cit.chunkId || cit.chunk_id,
    chunkIndex: cit.chunk_index,
    passage: cit.passage,
    similarityScore: cit.similarityScore ?? cit.similarity_score,
    sectionTitle,
    highlightCoordinates: cit.highlightCoordinates
      ? {
          top: cit.highlightCoordinates.top ?? 0,
          left: cit.highlightCoordinates.left ?? 0,
          width: cit.highlightCoordinates.width ?? 0,
          height: cit.highlightCoordinates.height ?? 0,
        }
      : undefined,
  };
}

/**
 * Maps a backend ChatSessionSummary to the frontend ChatSession model.
 */
export function toUiChatSession(s: ChatSessionSummary): ChatSession {
  return {
    id: s.session_id,
    sessionId: s.session_id,
    title: s.title || 'Conversation',
    createdAt: formatFriendlyDate(s.created_at),
    updatedAt: formatFriendlyDate(s.updated_at),
    messageCount: s.message_count,
  };
}

/**
 * Maps a backend ChatResponse to the frontend ChatMessage.
 */
export function toUiChatMessageFromResponse(res: ChatResponse): ChatMessage {
  const citations = (res.citations || []).map(toUiCitation);
  return {
    id: res.id,
    role: 'assistant',
    content: res.content || res.answer,
    timestamp: res.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    citations,
    confidenceScore: res.confidence_score,
    retrievalMode: (res.retrieval_mode as 'hybrid' | 'dense' | 'bm25') || 'dense',
    latencyMs: Math.round(res.latency_ms),
    tokensProcessed: res.tokens_processed,
  };
}

/**
 * Maps a backend ChatMessageResponse (from session history) to the frontend ChatMessage.
 */
export function toUiChatMessageFromHistory(msg: ChatMessageResponse): ChatMessage {
  const citations = (msg.citations || []).map(toUiCitation);
  return {
    id: msg.message_id,
    role: (msg.role as 'user' | 'assistant') || 'assistant',
    content: msg.content,
    timestamp: formatFriendlyDate(msg.created_at),
    citations,
  };
}
