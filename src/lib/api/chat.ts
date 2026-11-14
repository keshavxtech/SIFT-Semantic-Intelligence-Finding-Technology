/**
 * SIFT — RAG Chat & Conversational API Service
 * 
 * Provides methods for grounded multi-document question answering,
 * session management, conversation history persistence, and citation traversal.
 */

import { ApiClient, apiClient, RequestOptions } from './client';
import {
  ChatRequest,
  ChatResponse,
  ChatSessionDetail,
  ChatSessionSummary,
} from './types';

export class ChatApiService {
  private client: ApiClient;

  constructor(client: ApiClient = apiClient) {
    this.client = client;
  }

  /**
   * Generates a context-grounded RAG response with verified citation anchors.
   */
  public sendMessage(request: ChatRequest, options?: RequestOptions): Promise<ChatResponse> {
    return this.client.post<ChatResponse>('chat', request, options);
  }

  /**
   * Lists stored chat sessions with message counts and timestamps.
   */
  public listSessions(limit: number = 50, options?: RequestOptions): Promise<ChatSessionSummary[]> {
    return this.client.get<ChatSessionSummary[]>('chat/sessions', {
      ...options,
      params: { limit },
    });
  }

  /**
   * Retrieves full conversation turn history and citations for a specific chat session.
   */
  public getSession(sessionId: string, options?: RequestOptions): Promise<ChatSessionDetail> {
    return this.client.get<ChatSessionDetail>(`chat/sessions/${encodeURIComponent(sessionId)}`, options);
  }

  /**
   * Permanently deletes a chat session and all its associated conversation turns.
   */
  public deleteSession(sessionId: string, options?: RequestOptions): Promise<void> {
    return this.client.delete<void>(`chat/sessions/${encodeURIComponent(sessionId)}`, options);
  }
}

export const chatApi = new ChatApiService();
