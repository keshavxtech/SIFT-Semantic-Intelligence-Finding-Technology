/**
 * SIFT — Document API Service
 * 
 * Provides methods for document ingestion, metadata retrieval, renaming, deletion,
 * semantic chunk generation, dense embedding computation, and ChromaDB vector indexing.
 */

import { ApiClient, apiClient, RequestOptions } from './client';
import {
  ChunkQueryParams,
  DocumentChunksResponse,
  DocumentDetailResponse,
  DocumentListParams,
  DocumentListResponse,
  DocumentRenameRequest,
  DocumentResponse,
  DocumentUploadParams,
  EmbeddingGenerationResponse,
  EmbeddingSummaryResponse,
  IndexDocumentResponse,
  IndexStatusResponse,
} from './types';

export class DocumentApiService {
  private client: ApiClient;

  constructor(client: ApiClient = apiClient) {
    this.client = client;
  }

  /**
   * Lists all ingested documents with optional tag, status, or keyword filtering.
   */
  public list(params?: DocumentListParams, options?: RequestOptions): Promise<DocumentListResponse> {
    return this.client.get<DocumentListResponse>('documents', {
      ...options,
      params: {
        tag: params?.tag,
        status: params?.status,
        search: params?.search,
      },
    });
  }

  /**
   * Retrieves full document details including parsed full extracted text.
   */
  public get(documentId: string, options?: RequestOptions): Promise<DocumentDetailResponse> {
    return this.client.get<DocumentDetailResponse>(`documents/${encodeURIComponent(documentId)}`, options);
  }

  /**
   * Uploads a document via multipart FormData (.pdf or .txt, max 10MB).
   */
  public upload(params: DocumentUploadParams, options?: RequestOptions): Promise<DocumentResponse> {
    const formData = new FormData();
    formData.append('file', params.file);
    if (params.tag) {
      formData.append('tag', params.tag);
    }
    if (params.description) {
      formData.append('description', params.description);
    }
    return this.client.upload<DocumentResponse>('documents/upload', formData, options);
  }

  /**
   * Renames a document's title.
   */
  public rename(documentId: string, name: string, options?: RequestOptions): Promise<DocumentResponse> {
    const body: DocumentRenameRequest = { name };
    return this.client.patch<DocumentResponse>(
      `documents/${encodeURIComponent(documentId)}/rename`,
      body,
      options
    );
  }

  /**
   * Permanently deletes a document and cascades deletion through SQLite, storage, and ChromaDB.
   */
  public delete(documentId: string, options?: RequestOptions): Promise<void> {
    return this.client.delete<void>(`documents/${encodeURIComponent(documentId)}`, options);
  }

  /**
   * Generates and persists structure-aware chunks with sentence boundary preservation.
   */
  public generateChunks(
    documentId: string,
    params?: ChunkQueryParams,
    options?: RequestOptions
  ): Promise<DocumentChunksResponse> {
    return this.client.post<DocumentChunksResponse>(
      `documents/${encodeURIComponent(documentId)}/chunks`,
      undefined,
      {
        ...options,
        params: {
          chunk_size: params?.chunk_size,
          chunk_overlap: params?.chunk_overlap,
          minimum_chunk_size: params?.minimum_chunk_size,
        },
      }
    );
  }

  /**
   * Retrieves existing ordered chunks for a document.
   */
  public getChunks(documentId: string, options?: RequestOptions): Promise<DocumentChunksResponse> {
    return this.client.get<DocumentChunksResponse>(
      `documents/${encodeURIComponent(documentId)}/chunks`,
      options
    );
  }

  /**
   * Generates and persists 384-dimensional dense vectors using local all-MiniLM-L6-v2.
   */
  public generateEmbeddings(
    documentId: string,
    forceRecompute: boolean = false,
    options?: RequestOptions
  ): Promise<EmbeddingGenerationResponse> {
    return this.client.post<EmbeddingGenerationResponse>(
      `documents/${encodeURIComponent(documentId)}/embeddings`,
      undefined,
      {
        ...options,
        params: { force_recompute: forceRecompute },
      }
    );
  }

  /**
   * Retrieves summary metadata for existing document embeddings without transmitting raw vectors.
   */
  public getEmbeddingsSummary(
    documentId: string,
    options?: RequestOptions
  ): Promise<EmbeddingSummaryResponse> {
    return this.client.get<EmbeddingSummaryResponse>(
      `documents/${encodeURIComponent(documentId)}/embeddings`,
      options
    );
  }

  /**
   * Transfers persisted chunks and dense vectors into the persistent ChromaDB collection.
   */
  public indexDocument(
    documentId: string,
    forceReindex: boolean = false,
    options?: RequestOptions
  ): Promise<IndexDocumentResponse> {
    return this.client.post<IndexDocumentResponse>(
      `documents/${encodeURIComponent(documentId)}/index`,
      undefined,
      {
        ...options,
        params: { force_reindex: forceReindex },
      }
    );
  }

  /**
   * Checks whether a document's chunks are currently indexed in ChromaDB.
   */
  public getIndexStatus(documentId: string, options?: RequestOptions): Promise<IndexStatusResponse> {
    return this.client.get<IndexStatusResponse>(
      `documents/${encodeURIComponent(documentId)}/index-status`,
      options
    );
  }
}

export const documentApi = new DocumentApiService();
