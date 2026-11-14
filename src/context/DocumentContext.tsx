'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  DocumentItem,
  ProcessingActivity,
  ChatMessage,
  Citation,
  ChatSession,
  SearchResult,
  ActiveTab,
  DocumentTag,
} from '@/types';
import { INITIAL_ACTIVITIES } from '@/lib/mockData';
import { documentApi, searchApi, chatApi } from '@/lib/api';
import {
  toUiDocumentItem,
  toUiSearchResult,
  toUiChatMessageFromResponse,
  toUiChatMessageFromHistory,
  toUiChatSession,
} from '@/lib/api/adapters';
import {
  ChunkQueryParams,
  DocumentChunksResponse,
  EmbeddingGenerationResponse,
  IndexDocumentResponse,
  IndexStatusResponse,
} from '@/lib/api/types';

export type ProcessingStage = 'chunking' | 'embedding' | 'indexing';

export interface SearchOptions {
  documentIds?: string[];
  topK?: number;
  threshold?: number;
  mode?: string;
}

interface DocumentContextType {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  documents: DocumentItem[];
  isLoadingDocuments: boolean;
  documentsError: string | null;
  refreshDocuments: () => Promise<void>;
  activities: ProcessingActivity[];
  selectedDocIds: string[];
  toggleSelectDoc: (id: string) => void;
  selectAllDocs: () => void;
  deselectAllDocs: () => void;
  activeCitation: Citation | null;
  setActiveCitation: (citation: Citation | null) => void;
  chatMessages: ChatMessage[];
  isGeneratingResponse: boolean;
  chatError: string | null;
  clearChatError: () => void;
  sendChatMessage: (content: string, retrievalMode?: 'hybrid' | 'dense' | 'bm25') => Promise<void>;
  clearChat: () => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  isLoadingSessions: boolean;
  isLoadingChatHistory: boolean;
  refreshSessions: () => Promise<void>;
  loadSession: (sessionId: string) => Promise<void>;
  createNewChat: () => void;
  deleteChatSession: (sessionId: string) => Promise<void>;
  searchResults: SearchResult[];
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  similarityThreshold: number;
  setSimilarityThreshold: (thresh: number) => void;
  isSearching: boolean;
  searchError: string | null;
  searchExecutionTimeMs: number | null;
  performSearch: (query: string, options?: SearchOptions) => Promise<SearchResult[]>;
  isUploadModalOpen: boolean;
  setIsUploadModalOpen: (open: boolean) => void;
  isCommandPaletteOpen: boolean;
  setIsCommandPaletteOpen: (open: boolean) => void;
  uploadDocument: (params: {
    file: File;
    tag?: DocumentTag;
    description?: string;
    onProgress?: (stage: 'OCR & Layout' | 'Chunking' | 'Embedding' | 'Indexing', progress: number) => void;
  }) => Promise<DocumentItem>;
  deleteDocument: (id: string) => Promise<void>;
  renameDocument: (id: string, newName: string) => Promise<void>;
  activePreviewDoc: DocumentItem | null;
  setActivePreviewDoc: (doc: DocumentItem | null) => void;
  quickAskDocument: (doc: DocumentItem) => void;
  processingDocIds: Record<string, ProcessingStage | undefined>;
  generateChunks: (documentId: string, params?: ChunkQueryParams) => Promise<DocumentChunksResponse>;
  generateEmbeddings: (documentId: string, forceRecompute?: boolean) => Promise<EmbeddingGenerationResponse>;
  indexDocument: (documentId: string, forceReindex?: boolean) => Promise<IndexDocumentResponse>;
  checkIndexStatus: (documentId: string) => Promise<IndexStatusResponse>;
  processDocumentPipeline: (documentId: string, forceReindex?: boolean) => Promise<DocumentItem>;
}

const DocumentContext = createContext<DocumentContextType | undefined>(undefined);

export function DocumentProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState<ActiveTab>('dashboard');
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState<boolean>(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);
  const [activities, setActivities] = useState<ProcessingActivity[]>(INITIAL_ACTIVITIES);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState<boolean>(false);
  const [isLoadingChatHistory, setIsLoadingChatHistory] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchExecutionTimeMs, setSearchExecutionTimeMs] = useState<number | null>(null);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.50);
  const latestSearchIdRef = React.useRef(0);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [activePreviewDoc, setActivePreviewDoc] = useState<DocumentItem | null>(null);
  const [processingDocIds, setProcessingDocIds] = useState<Record<string, ProcessingStage | undefined>>({});

  // Global Ctrl+K command palette shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Fetch real documents on mount
  const refreshDocuments = useCallback(async (): Promise<void> => {
    try {
      setIsLoadingDocuments(true);
      setDocumentsError(null);
      const res = await documentApi.list();
      const uiDocs = res.documents.map(toUiDocumentItem);
      setDocuments(uiDocs);
      setSelectedDocIds(uiDocs.map((d) => d.id));
      if (uiDocs.length > 0 && !activePreviewDoc) {
        setActivePreviewDoc(uiDocs[0]);
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unable to communicate with the SIFT backend server. Please verify the service is running at http://127.0.0.1:8000.';
      setDocumentsError(message);
    } finally {
      setIsLoadingDocuments(false);
    }
  }, [activePreviewDoc]);

  useEffect(() => {
    let isMounted = true;
    void (async () => {
      try {
        const res = await documentApi.list();
        if (!isMounted) return;
        const uiDocs = res.documents.map(toUiDocumentItem);
        setDocuments(uiDocs);
        setSelectedDocIds(uiDocs.map((d) => d.id));
        if (uiDocs.length > 0) {
          setActivePreviewDoc(uiDocs[0]);
        }
      } catch (err: unknown) {
        if (!isMounted) return;
        const message =
          err instanceof Error
            ? err.message
            : 'Unable to communicate with the SIFT backend server. Please verify the service is running at http://127.0.0.1:8000.';
        setDocumentsError(message);
      } finally {
        if (isMounted) {
          setIsLoadingDocuments(false);
        }
      }
    })();

    return () => {
      isMounted = false;
    };
  }, []);

  const toggleSelectDoc = (id: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAllDocs = () => {
    setSelectedDocIds(documents.map((d) => d.id));
  };

  const deselectAllDocs = () => {
    setSelectedDocIds([]);
  };

  const clearChatError = useCallback(() => {
    setChatError(null);
  }, []);

  const refreshSessions = useCallback(async (): Promise<void> => {
    try {
      setIsLoadingSessions(true);
      const list = await chatApi.listSessions(50);
      setSessions(list.map(toUiChatSession));
    } catch {
      // background refresh failure should not disrupt UI
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  const createNewChat = useCallback(() => {
    setCurrentSessionId(null);
    setChatMessages([]);
    setActiveCitation(null);
    setChatError(null);
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.removeItem('sift_active_session_id');
      }
    } catch {
      // ignore
    }
  }, []);

  const clearChat = useCallback(() => {
    createNewChat();
  }, [createNewChat]);

  const loadSession = useCallback(async (sessionId: string): Promise<void> => {
    try {
      setIsLoadingChatHistory(true);
      setChatError(null);
      const detail = await chatApi.getSession(sessionId);
      setCurrentSessionId(detail.session_id);
      try {
        if (typeof window !== 'undefined') {
          window.localStorage.setItem('sift_active_session_id', detail.session_id);
        }
      } catch {
        // ignore
      }
      const uiMessages = detail.messages.map(toUiChatMessageFromHistory);
      setChatMessages(uiMessages);

      const lastAssistantWithCitations = [...uiMessages]
        .reverse()
        .find((m) => m.role === 'assistant' && m.citations && m.citations.length > 0);
      if (lastAssistantWithCitations && lastAssistantWithCitations.citations) {
        setActiveCitation(lastAssistantWithCitations.citations[0]);
      } else {
        setActiveCitation(null);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load conversation history.';
      setChatError(msg);
    } finally {
      setIsLoadingChatHistory(false);
    }
  }, []);

  const deleteChatSession = useCallback(
    async (sessionId: string): Promise<void> => {
      try {
        await chatApi.deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.sessionId !== sessionId));
        if (currentSessionId === sessionId) {
          createNewChat();
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Failed to delete chat session.';
        setChatError(msg);
        throw err;
      }
    },
    [currentSessionId, createNewChat]
  );

  const sendChatMessage = useCallback(
    async (content: string, retrievalMode: 'hybrid' | 'dense' | 'bm25' = 'dense'): Promise<void> => {
      const trimmed = content.trim();
      if (!trimmed || isGeneratingResponse) return;

      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: trimmed,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setChatMessages((prev) => [...prev, userMsg]);
      setIsGeneratingResponse(true);
      setChatError(null);

      try {
        const res = await chatApi.sendMessage({
          message: trimmed,
          session_id: currentSessionId || undefined,
          document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
          top_k: 5,
          retrieval_mode: retrievalMode,
        });

        if (res.session_id) {
          setCurrentSessionId(res.session_id);
          try {
            if (typeof window !== 'undefined') {
              window.localStorage.setItem('sift_active_session_id', res.session_id);
            }
          } catch {
            // ignore
          }
          void refreshSessions();
        }

        const botMsg = toUiChatMessageFromResponse(res);
        setChatMessages((prev) => [...prev, botMsg]);

        if (botMsg.citations && botMsg.citations.length > 0) {
          setActiveCitation(botMsg.citations[0]);
        }
      } catch (err: unknown) {
        let errorMsg = 'An error occurred while communicating with the SIFT RAG backend.';
        if (err instanceof Error) {
          errorMsg = err.message;
        }
        setChatError(errorMsg);
      } finally {
        setIsGeneratingResponse(false);
      }
    },
    [currentSessionId, selectedDocIds, isGeneratingResponse, refreshSessions]
  );

  // Initial load of sessions from backend and auto-restore active session from localStorage
  useEffect(() => {
    let isMounted = true;
    void (async () => {
      try {
        const list = await chatApi.listSessions(50);
        if (!isMounted) return;
        const uiSessions = list.map(toUiChatSession);
        setSessions(uiSessions);

        let savedSessionId: string | null = null;
        try {
          if (typeof window !== 'undefined') {
            savedSessionId = window.localStorage.getItem('sift_active_session_id');
          }
        } catch {
          // ignore
        }

        if (savedSessionId && uiSessions.some((s) => s.sessionId === savedSessionId)) {
          await loadSession(savedSessionId);
        }
      } catch {
        // ignore initial session load error
      }
    })();

    return () => {
      isMounted = false;
    };
  }, [loadSession]);

  const quickAskDocument = (doc: DocumentItem) => {
    setSelectedDocIds([doc.id]);
    setActivePreviewDoc(doc);
    setActiveTab('chat');
  };

  const performSearch = useCallback(
    async (query: string, options?: SearchOptions): Promise<SearchResult[]> => {
      const trimmed = query.trim();
      setSearchQuery(trimmed);

      if (!trimmed) {
        setSearchResults([]);
        setSearchError(null);
        setSearchExecutionTimeMs(null);
        setIsSearching(false);
        return [];
      }

      const currentSearchId = ++latestSearchIdRef.current;
      setIsSearching(true);
      setSearchError(null);

      try {
        const res = await searchApi.search({
          query: trimmed,
          document_ids:
            options?.documentIds && options.documentIds.length > 0
              ? options.documentIds
              : undefined,
          top_k: options?.topK ?? 10,
          threshold: options?.threshold,
          mode: options?.mode ?? 'semantic',
        });

        // Protect against race conditions: discard stale search responses
        if (currentSearchId !== latestSearchIdRef.current) {
          return [];
        }

        const mappedResults = (res.results || []).map(toUiSearchResult);
        setSearchResults(mappedResults);
        setSearchExecutionTimeMs(res.execution_time_ms);
        return mappedResults;
      } catch (err: unknown) {
        if (currentSearchId !== latestSearchIdRef.current) {
          return [];
        }
        const message =
          err instanceof Error
            ? err.message
            : 'Failed to retrieve passages. Please verify ChromaDB and backend are running.';
        setSearchError(message);
        setSearchResults([]);
        setSearchExecutionTimeMs(null);
        return [];
      } finally {
        if (currentSearchId === latestSearchIdRef.current) {
          setIsSearching(false);
        }
      }
    },
    []
  );

  /**
   * Real Ingestion Pipeline:
   * 1. POST /api/documents/upload -> text extraction, metadata, SQLite persistence
   * 2. POST /api/documents/{id}/chunks -> structure-aware sentence & paragraph chunking
   * 3. POST /api/documents/{id}/embeddings -> 384d dense vector generation (local MiniLM)
   * 4. POST /api/documents/{id}/index -> transfer to persistent ChromaDB collection
   */
  const uploadDocument = async ({
    file,
    tag,
    description,
    onProgress,
  }: {
    file: File;
    tag?: DocumentTag;
    description?: string;
    onProgress?: (stage: 'OCR & Layout' | 'Chunking' | 'Embedding' | 'Indexing', progress: number) => void;
  }): Promise<DocumentItem> => {
    // Stage 1: Upload & Text Extraction
    onProgress?.('OCR & Layout', 30);
    const uploaded = await documentApi.upload({
      file,
      tag: tag || 'Technical',
      description: description || '',
    });

    try {
      // Stage 2: Chunking
      setProcessingDocIds((prev) => ({ ...prev, [uploaded.id]: 'chunking' }));
      onProgress?.('Chunking', 60);
      await documentApi.generateChunks(uploaded.id);

      // Stage 3: Embedding
      setProcessingDocIds((prev) => ({ ...prev, [uploaded.id]: 'embedding' }));
      onProgress?.('Embedding', 85);
      await documentApi.generateEmbeddings(uploaded.id);

      // Stage 4: Indexing in ChromaDB
      setProcessingDocIds((prev) => ({ ...prev, [uploaded.id]: 'indexing' }));
      onProgress?.('Indexing', 100);
      await documentApi.indexDocument(uploaded.id);

      // Fetch up-to-date document record with verified chunk and vector counts
      const fullDoc = await documentApi.get(uploaded.id);
      const uiDoc = toUiDocumentItem(fullDoc);

      setDocuments((prev) => [uiDoc, ...prev.filter((d) => d.id !== uiDoc.id)]);
      setSelectedDocIds((prev) => (prev.includes(uiDoc.id) ? prev : [uiDoc.id, ...prev]));

      // Record activity in live activity stream
      const actId = `act-${Date.now()}`;
      const newActivity: ProcessingActivity = {
        id: actId,
        docId: uiDoc.id,
        docName: uiDoc.name,
        stage: 'Indexing',
        progress: 100,
        timestamp: 'Just now',
        status: 'completed',
      };
      setActivities((prev) => [newActivity, ...prev.slice(0, 19)]);

      return uiDoc;
    } finally {
      setProcessingDocIds((prev) => {
        const copy = { ...prev };
        delete copy[uploaded.id];
        return copy;
      });
    }
  };

  /**
   * Generates structure-aware chunks for an existing document.
   */
  const generateChunks = async (
    documentId: string,
    params?: ChunkQueryParams
  ): Promise<DocumentChunksResponse> => {
    setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'chunking' }));
    try {
      const res = await documentApi.generateChunks(documentId, params);
      const updatedWire = await documentApi.get(documentId);
      const uiDoc = toUiDocumentItem(updatedWire);
      setDocuments((prev) => prev.map((d) => (d.id === documentId ? uiDoc : d)));
      if (activePreviewDoc?.id === documentId) {
        setActivePreviewDoc(uiDoc);
      }
      return res;
    } finally {
      setProcessingDocIds((prev) => {
        const copy = { ...prev };
        delete copy[documentId];
        return copy;
      });
    }
  };

  /**
   * Computes 384-dimensional dense vectors using local MiniLM on CPU.
   */
  const generateEmbeddings = async (
    documentId: string,
    forceRecompute: boolean = false
  ): Promise<EmbeddingGenerationResponse> => {
    setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'embedding' }));
    try {
      const res = await documentApi.generateEmbeddings(documentId, forceRecompute);
      const updatedWire = await documentApi.get(documentId);
      const uiDoc = toUiDocumentItem(updatedWire);
      setDocuments((prev) => prev.map((d) => (d.id === documentId ? uiDoc : d)));
      if (activePreviewDoc?.id === documentId) {
        setActivePreviewDoc(uiDoc);
      }
      return res;
    } finally {
      setProcessingDocIds((prev) => {
        const copy = { ...prev };
        delete copy[documentId];
        return copy;
      });
    }
  };

  /**
   * Indexes persisted chunks and embeddings into persistent ChromaDB collection.
   */
  const indexDocument = async (
    documentId: string,
    forceReindex: boolean = false
  ): Promise<IndexDocumentResponse> => {
    setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'indexing' }));
    try {
      const res = await documentApi.indexDocument(documentId, forceReindex);
      const updatedWire = await documentApi.get(documentId);
      const uiDoc = toUiDocumentItem(updatedWire);
      setDocuments((prev) => prev.map((d) => (d.id === documentId ? uiDoc : d)));
      if (activePreviewDoc?.id === documentId) {
        setActivePreviewDoc(uiDoc);
      }
      return res;
    } finally {
      setProcessingDocIds((prev) => {
        const copy = { ...prev };
        delete copy[documentId];
        return copy;
      });
    }
  };

  /**
   * Queries real index status directly from ChromaDB.
   */
  const checkIndexStatus = async (documentId: string): Promise<IndexStatusResponse> => {
    return documentApi.getIndexStatus(documentId);
  };

  /**
   * End-to-end processing pipeline for existing documents:
   * Chunks -> Embeddings -> ChromaDB Indexing
   */
  const processDocumentPipeline = async (
    documentId: string,
    forceReindex: boolean = false
  ): Promise<DocumentItem> => {
    try {
      // 1. Chunking
      setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'chunking' }));
      await documentApi.generateChunks(documentId);

      // 2. Embedding
      setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'embedding' }));
      await documentApi.generateEmbeddings(documentId, forceReindex);

      // 3. Indexing
      setProcessingDocIds((prev) => ({ ...prev, [documentId]: 'indexing' }));
      await documentApi.indexDocument(documentId, forceReindex);

      // 4. Update document state
      const updatedWireDoc = await documentApi.get(documentId);
      const uiDoc = toUiDocumentItem(updatedWireDoc);

      setDocuments((prev) => prev.map((d) => (d.id === documentId ? uiDoc : d)));
      if (activePreviewDoc?.id === documentId) {
        setActivePreviewDoc(uiDoc);
      }

      // Add to activities
      const newActivity: ProcessingActivity = {
        id: `act-${Date.now()}`,
        docId: uiDoc.id,
        docName: uiDoc.name,
        stage: 'Indexing',
        progress: 100,
        timestamp: 'Just now',
        status: 'completed',
      };
      setActivities((prev) => [newActivity, ...prev.slice(0, 19)]);

      return uiDoc;
    } catch (err: unknown) {
      try {
        const fallback = await documentApi.get(documentId);
        const refreshedUi = toUiDocumentItem(fallback);
        setDocuments((prev) => prev.map((d) => (d.id === documentId ? refreshedUi : d)));
      } catch {
        // preserve current state on error
      }
      throw err;
    } finally {
      setProcessingDocIds((prev) => {
        const copy = { ...prev };
        delete copy[documentId];
        return copy;
      });
    }
  };

  const deleteDocument = async (id: string): Promise<void> => {
    await documentApi.delete(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
    setSelectedDocIds((prev) => prev.filter((item) => item !== id));
    if (activePreviewDoc?.id === id) {
      setActivePreviewDoc(null);
    }
  };

  const renameDocument = async (id: string, newName: string): Promise<void> => {
    if (!newName.trim()) return;
    const updated = await documentApi.rename(id, newName.trim());
    const uiDoc = toUiDocumentItem(updated);
    setDocuments((prev) => prev.map((d) => (d.id === id ? uiDoc : d)));
    if (activePreviewDoc?.id === id) {
      setActivePreviewDoc(uiDoc);
    }
  };

  return (
    <DocumentContext.Provider
      value={{
        activeTab,
        setActiveTab,
        documents,
        isLoadingDocuments,
        documentsError,
        refreshDocuments,
        activities,
        selectedDocIds,
        toggleSelectDoc,
        selectAllDocs,
        deselectAllDocs,
        activeCitation,
        setActiveCitation,
        chatMessages,
        isGeneratingResponse,
        chatError,
        clearChatError,
        sendChatMessage,
        clearChat,
        sessions,
        currentSessionId,
        isLoadingSessions,
        isLoadingChatHistory,
        refreshSessions,
        loadSession,
        createNewChat,
        deleteChatSession,
        searchResults,
        searchQuery,
        setSearchQuery,
        similarityThreshold,
        setSimilarityThreshold,
        isSearching,
        searchError,
        searchExecutionTimeMs,
        performSearch,
        isUploadModalOpen,
        setIsUploadModalOpen,
        isCommandPaletteOpen,
        setIsCommandPaletteOpen,
        uploadDocument,
        deleteDocument,
        renameDocument,
        activePreviewDoc,
        setActivePreviewDoc,
        quickAskDocument,
        processingDocIds,
        generateChunks,
        generateEmbeddings,
        indexDocument,
        checkIndexStatus,
        processDocumentPipeline,
      }}
    >
      {children}
    </DocumentContext.Provider>
  );
}

export function useDocumentContext() {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocumentContext must be used within a DocumentProvider');
  }
  return context;
}
