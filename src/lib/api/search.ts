/**
 * SIFT — Semantic Search API Service
 * 
 * Provides methods for neural semantic vector retrieval across indexed documents
 * using dense query embeddings and ChromaDB cosine distance.
 */

import { ApiClient, apiClient, RequestOptions } from './client';
import { SearchRequest, SearchResponse } from './types';

export class SearchApiService {
  private client: ApiClient;

  constructor(client: ApiClient = apiClient) {
    this.client = client;
  }

  /**
   * Executes dense vector similarity search against the active ChromaDB collection.
   */
  public search(request: SearchRequest, options?: RequestOptions): Promise<SearchResponse> {
    return this.client.post<SearchResponse>('search', request, options);
  }
}

export const searchApi = new SearchApiService();
