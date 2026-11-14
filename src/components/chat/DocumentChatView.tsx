'use client';

import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  FileText,
  Search,
  CheckSquare,
  Square,
  Sparkles,
  Send,
  ExternalLink,
  Copy,
  Check,
  ZoomIn,
  ZoomOut,
  PanelLeft,
  PanelRight,
  X,
  Layers,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  RotateCcw,
  BookOpen,
  Plus,
  AlertCircle,
  Trash2,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { Citation, DocumentType } from '@/types';

export const DocumentChatView: React.FC = () => {
  const {
    documents,
    selectedDocIds,
    toggleSelectDoc,
    selectAllDocs,
    deselectAllDocs,
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
    loadSession,
    createNewChat,
    deleteChatSession,
    activeCitation,
    setActiveCitation,
    setIsUploadModalOpen,
  } = useDocumentContext();

  const [inputMessage, setInputMessage] = useState('');
  const [retrievalMode, setRetrievalMode] = useState<'hybrid' | 'dense' | 'bm25'>('dense');
  const [docFilter, setDocFilter] = useState('');
  const [activeLeftTab, setActiveLeftTab] = useState<'scope' | 'history'>('scope');
  const [activeRightTab, setActiveRightTab] = useState<'pdfPreview' | 'passages'>('pdfPreview');
  const [zoomLevel, setZoomLevel] = useState(100);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [sourcesExpanded, setSourcesExpanded] = useState(true);
  const [expandedCitationIds, setExpandedCitationIds] = useState<Record<string, boolean>>({});

  // Responsive panel visibility
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);

  // Auto-collapse panels on smaller screens
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1200) {
        setShowLeftPanel(false);
      }
      if (window.innerWidth < 900) {
        setShowRightPanel(false);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isGeneratingResponse]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isGeneratingResponse) return;
    void sendChatMessage(inputMessage, retrievalMode);
    setInputMessage('');
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleCitationClick = (cit: Citation) => {
    setActiveCitation(cit);
    setExpandedCitationIds((prev) => ({ ...prev, [cit.id]: true }));
    setActiveRightTab('pdfPreview');
    setShowRightPanel(true);
  };

  const toggleCitationAccordion = (id: string) => {
    setExpandedCitationIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Filter documents in Left Panel
  const filteredDocs = useMemo(() => {
    return documents.filter(
      (d) =>
        d.name.toLowerCase().includes(docFilter.toLowerCase()) ||
        d.tag.toLowerCase().includes(docFilter.toLowerCase())
    );
  }, [documents, docFilter]);

  // Scoped Token Calculation
  const scopedTokens = useMemo(() => {
    return documents
      .filter((d) => selectedDocIds.includes(d.id))
      .reduce((acc, d) => acc + d.tokenCount, 0);
  }, [documents, selectedDocIds]);

  const promptSuggestions = [
    'What is RAG and how does it work?',
    'What is the deductible for motor vehicle windshield damage?',
    'Summarize the project architecture and cluster topology.',
    'What are the data encryption and retention policies under compliance?',
  ];

  const activeDocForCitation = activeCitation
    ? documents.find((d) => d.id === activeCitation.docId) || documents[0]
    : documents[0];

  const activeTitle = useMemo(() => {
    if (!currentSessionId) return 'New Conversation';
    const found = sessions.find((s) => s.sessionId === currentSessionId);
    return found ? found.title : 'Conversation';
  }, [currentSessionId, sessions]);

  const getFileTypeBadge = (type: DocumentType) => {
    switch (type) {
      case 'pdf':
        return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'txt':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'md':
        return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      default:
        return 'bg-gray-500/10 text-gray-300 border-gray-500/20';
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] w-full overflow-hidden bg-[#0A0C10] relative">
      {/* ======================================================== */}
      {/* 1. LEFT PANEL: Document Context & Query Scope Navigator */}
      {/* ======================================================== */}
      {showLeftPanel && (
        <>
          {/* Mobile Backdrop */}
          <div
            onClick={() => setShowLeftPanel(false)}
            className="fixed inset-0 z-25 bg-black/60 backdrop-blur-xs md:hidden"
            aria-hidden="true"
          />
          <aside className="fixed inset-y-16 left-0 z-30 w-72 md:static md:z-auto shrink-0 border-r border-[#1E232D] bg-[#0E1015] flex flex-col h-full shadow-2xl md:shadow-none animate-in slide-in-from-left-2 duration-150">
            {/* Top Panel Tab Switcher: Context Scope vs. Chat History */}
            <div className="p-3 border-b border-[#1E232D]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-white flex items-center gap-2">
                  <BookOpen className="h-3.5 w-3.5 text-[#3B82F6]" aria-hidden="true" />
                  Document Context
                </span>
                <button
                  onClick={() => setShowLeftPanel(false)}
                  aria-label="Close document context panel"
                  className="md:hidden text-[#717886] hover:text-white p-1 rounded-lg hover:bg-[#181D26]"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

            <div className="flex rounded-lg bg-[#14171E] border border-[#232834] p-0.5 text-xs font-medium">
              <button
                onClick={() => setActiveLeftTab('scope')}
                className={`flex-1 py-1 rounded-md transition-colors ${
                  activeLeftTab === 'scope'
                    ? 'bg-[#1E232D] text-white shadow-xs'
                    : 'text-[#8C93A3] hover:text-white'
                }`}
              >
                Active Scope ({selectedDocIds.length})
              </button>
              <button
                onClick={() => setActiveLeftTab('history')}
                className={`flex-1 py-1 rounded-md transition-colors ${
                  activeLeftTab === 'history'
                    ? 'bg-[#1E232D] text-white shadow-xs'
                    : 'text-[#8C93A3] hover:text-white'
                }`}
              >
                Threads ({sessions.length})
              </button>
            </div>
          </div>

          {activeLeftTab === 'scope' ? (
            /* TAB A: DOCUMENT SCOPE & SELECTION */
            <div className="flex-1 flex flex-col min-h-0">
              {/* Quick Filter & Select All / Deselect All */}
              <div className="p-3 border-b border-[#1E232D]/70 space-y-2">
                <div className="flex items-center justify-between text-[11px] font-mono text-[#717886]">
                  <span>Scope Controls</span>
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={selectAllDocs}
                      className="hover:text-white px-1.5 py-0.5 rounded hover:bg-[#181D26]"
                    >
                      All ({documents.length})
                    </button>
                    <span>·</span>
                    <button
                      onClick={deselectAllDocs}
                      className="hover:text-white px-1.5 py-0.5 rounded hover:bg-[#181D26]"
                    >
                      None
                    </button>
                  </div>
                </div>

                <div className="relative">
                  <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-[#717886]" />
                  <input
                    type="text"
                    value={docFilter}
                    onChange={(e) => setDocFilter(e.target.value)}
                    placeholder="Search scoped documents..."
                    className="w-full rounded-lg bg-[#14171E] border border-[#232834] pl-8 pr-2.5 py-1.5 text-xs text-white placeholder-[#717886] outline-none focus:border-[#3B82F6] transition-colors"
                  />
                  {docFilter && (
                    <button
                      onClick={() => setDocFilter('')}
                      className="absolute right-2 top-2 text-[#717886] hover:text-white"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </div>
              </div>

              {/* Retrievable Tokens Info Banner */}
              <div className="px-3 py-2 bg-[#12151B] border-b border-[#1E232D] flex items-center justify-between text-[11px] font-mono">
                <span className="text-[#8C93A3]">Retrievable Context:</span>
                <span className="text-[#3B82F6] font-semibold">
                  {scopedTokens.toLocaleString()} tokens
                </span>
              </div>

              {/* Documents Checkbox List */}
              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {filteredDocs.length === 0 ? (
                  <div className="p-4 text-center text-xs text-[#717886]">
                    No documents found.
                  </div>
                ) : (
                  filteredDocs.map((doc) => {
                    const isChecked = selectedDocIds.includes(doc.id);
                    const typeBadgeClass = getFileTypeBadge(doc.type);
                    const isIndexed = doc.status === 'indexed';

                    return (
                      <div
                        key={doc.id}
                        onClick={() => toggleSelectDoc(doc.id)}
                        className={`flex items-start gap-2.5 p-2 rounded-lg cursor-pointer transition-colors ${
                          isChecked
                            ? 'bg-[#14171E] border border-[#232834]'
                            : 'hover:bg-[#14171E]/60 border border-transparent opacity-65'
                        }`}
                      >
                        <button className="mt-0.5 text-[#3B82F6] shrink-0">
                          {isChecked ? (
                            <CheckSquare className="h-4 w-4" />
                          ) : (
                            <Square className="h-4 w-4 text-[#717886]" />
                          )}
                        </button>

                        <div className="min-w-0 flex-1">
                          <div className="flex items-center justify-between gap-1">
                            <span className="text-xs font-medium text-white truncate">
                              {doc.name}
                            </span>
                            {isIndexed && (
                              <span className="text-[10px] text-[#10B981] font-mono" title="Indexed in ChromaDB">
                                ✓
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2 mt-1 text-[10px] font-mono text-[#717886]">
                            <span
                              className={`px-1.5 py-0.2 rounded uppercase border text-[9px] ${typeBadgeClass}`}
                            >
                              {doc.type}
                            </span>
                            <span>{doc.chunksCount} chk</span>
                            <span>{doc.size}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Bottom Ingest Shortcut */}
              <div className="p-3 border-t border-[#1E232D] bg-[#0E1015]">
                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  className="w-full flex items-center justify-center gap-1.5 py-1.5 text-xs text-[#8C93A3] hover:text-white hover:bg-[#14171E] rounded-lg border border-dashed border-[#232834] transition-colors"
                >
                  <Plus className="h-3.5 w-3.5 text-[#3B82F6]" />
                  <span>Upload & Add to Context</span>
                </button>
              </div>
            </div>
          ) : (
            /* TAB B: CONVERSATION THREADS HISTORY */
            <div className="flex-1 flex flex-col min-h-0">
              <div className="p-3 border-b border-[#1E232D]/70">
                <button
                  onClick={createNewChat}
                  className="w-full flex items-center justify-center gap-2 py-1.5 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-medium text-white transition-colors"
                >
                  <Plus className="h-3.5 w-3.5" />
                  <span>New Conversation</span>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {isLoadingSessions ? (
                  <div className="p-4 text-center text-xs text-[#717886] font-mono">
                    Loading conversations...
                  </div>
                ) : sessions.length === 0 ? (
                  <div className="p-4 text-center text-xs text-[#717886]">
                    No conversation threads yet. Start asking questions!
                  </div>
                ) : (
                  sessions.map((s) => {
                    const isActive = s.sessionId === currentSessionId;
                    return (
                      <div
                        key={s.sessionId}
                        onClick={() => void loadSession(s.sessionId)}
                        className={`group flex items-center justify-between w-full px-2.5 py-2 rounded-lg text-left text-xs transition-colors cursor-pointer ${
                          isActive
                            ? 'bg-[#181C26] text-white font-medium border border-[#232834]'
                            : 'text-[#8C93A3] hover:text-white hover:bg-[#14171E]'
                        }`}
                      >
                        <div className="min-w-0 flex-1 pr-2">
                          <span className="truncate block font-medium text-white">{s.title}</span>
                          <span className="text-[10px] font-mono text-[#717886] block mt-0.5">
                            {s.messageCount} turns · {s.updatedAt}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            void deleteChatSession(s.sessionId);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 text-[#717886] hover:text-red-400 transition-opacity rounded hover:bg-[#1E232D]"
                          title="Delete thread"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </aside>
      </>
    )}

      {/* ======================================================== */}
      {/* 2. CENTER PANEL: AI Conversational Thread (Hero Focus)  */}
      {/* ======================================================== */}
      <div className="flex-1 flex flex-col h-full bg-[#0E1015] relative min-w-0">
        {/* Workspace Subheader Toolbar */}
        <div className="h-12 border-b border-[#1E232D] bg-[#0E1015] px-4 flex items-center justify-between text-xs shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowLeftPanel(!showLeftPanel)}
              aria-label={showLeftPanel ? 'Close Document Context Panel' : 'Open Document Context Panel'}
              className={`p-1.5 rounded-lg transition-colors ${
                showLeftPanel
                  ? 'bg-[#181C26] text-[#3B82F6] border border-[#232834]'
                  : 'text-[#8C93A3] hover:text-white hover:bg-[#161A22]'
              }`}
              title="Toggle Document Context Panel"
            >
              <PanelLeft className="h-4 w-4" aria-hidden="true" />
            </button>
            <div className="h-3.5 w-px bg-[#1E232D]" />
            <div className="flex items-center gap-2">
              <h2 className="font-semibold text-white text-xs tracking-tight truncate max-w-[200px] sm:max-w-xs">
                {activeTitle}
              </h2>
              <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono text-[#8C93A3] bg-[#14171E] border border-[#232834]">
                {selectedDocIds.length} scoped
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Retrieval Algorithm Pill */}
            <div className="hidden sm:flex items-center rounded-lg bg-[#14171E] border border-[#232834] p-0.5 text-[11px] font-mono">
              <button
                onClick={() => setRetrievalMode('dense')}
                className={`px-2.5 py-0.5 rounded transition-colors ${
                  retrievalMode === 'dense'
                    ? 'bg-[#1E232D] text-[#3B82F6] font-medium'
                    : 'text-[#8C93A3] hover:text-white'
                }`}
                title="Dense vector cosine similarity via ChromaDB"
              >
                Dense Vectors
              </button>
              <button
                onClick={() => setRetrievalMode('hybrid')}
                className={`px-2.5 py-0.5 rounded transition-colors ${
                  retrievalMode === 'hybrid'
                    ? 'bg-[#1E232D] text-[#3B82F6] font-medium'
                    : 'text-[#8C93A3] hover:text-white'
                }`}
                title="Dense embeddings + sparse BM25 fusion"
              >
                Hybrid BM25
              </button>
            </div>

            <button
              onClick={clearChat}
              aria-label="Reset Conversation"
              className="p-1.5 text-[#8C93A3] hover:text-white rounded-lg hover:bg-[#161A22] transition-colors"
              title="Reset Conversation"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            </button>

            <button
              onClick={() => setShowRightPanel(!showRightPanel)}
              aria-label={showRightPanel ? 'Close Evidence & PDF Inspector' : 'Open Evidence & PDF Inspector'}
              className={`p-1.5 rounded-lg transition-colors ${
                showRightPanel
                  ? 'bg-[#181C26] text-[#3B82F6] border border-[#232834]'
                  : 'text-[#8C93A3] hover:text-white hover:bg-[#161A22]'
              }`}
              title="Toggle Evidence & PDF Inspector"
            >
              <PanelRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>

        {/* Global Chat Error Notification Banner */}
        {chatError && (
          <div className="mx-4 md:mx-8 mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300 flex items-start justify-between gap-3 shadow-md shrink-0">
            <div className="flex items-start gap-2.5 min-w-0">
              <AlertCircle className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-white block">RAG Interaction Notice</span>
                <p className="mt-0.5 leading-relaxed text-red-200/90">{chatError}</p>
              </div>
            </div>
            <button
              onClick={clearChatError}
              className="text-red-400 hover:text-white p-1 rounded hover:bg-red-500/20 transition-colors shrink-0"
              title="Dismiss error"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {/* Conversation Message Stream */}
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
          {isLoadingChatHistory ? (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
              <div className="h-8 w-8 rounded-full bg-[#182338] text-[#3B82F6] flex items-center justify-center animate-spin">
                <Layers className="h-4 w-4" />
              </div>
              <p className="text-xs text-[#8C93A3] font-mono">
                Loading conversational turns and verified citations from backend...
              </p>
            </div>
          ) : chatMessages.length === 0 ? (
            /* Empty State with Suggested Questions */
            <div className="flex flex-col items-center justify-center h-full max-w-lg mx-auto text-center space-y-5">
              <div className="h-11 w-11 rounded-xl bg-[#14171E] border border-[#232834] flex items-center justify-center text-[#3B82F6] shadow-sm">
                <Sparkles className="h-6 w-6" />
              </div>

              <div>
                <h3 className="text-sm font-semibold text-white tracking-tight">
                  SIFT Document Intelligence Engine
                </h3>
                <p className="text-xs text-[#8C93A3] mt-1.5 leading-relaxed">
                  Query across your scoped files with multi-document cross-referencing and verifiable page citations.
                </p>
              </div>

              {/* Prompt Suggestions Grid */}
              <div className="w-full space-y-2 pt-2">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[#525A6A]">
                  Suggested Questions
                </span>
                <div className="grid grid-cols-1 gap-2 text-left">
                  {promptSuggestions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => void sendChatMessage(q, retrievalMode)}
                      className="p-3 rounded-xl bg-[#14171E] hover:bg-[#181D26] border border-[#1E232D] hover:border-[#2A3140] text-xs text-white transition-all flex items-center justify-between group"
                    >
                      <span className="truncate">{q}</span>
                      <ArrowRight className="h-3.5 w-3.5 text-[#525A6A] group-hover:text-[#3B82F6] shrink-0 ml-2 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            chatMessages.map((msg) => {
              const isUser = msg.role === 'user';
              return (
                <div key={msg.id} className="space-y-3">
                  {isUser ? (
                    /* User Question Bubble (Right aligned, elevated dark charcoal pill) */
                    <div className="flex justify-end">
                      <div className="max-w-xl rounded-2xl rounded-tr-sm bg-[#1A1F29] border border-[#262C3A] px-4 py-2.5 text-xs text-white leading-relaxed shadow-sm">
                        {msg.content}
                      </div>
                    </div>
                  ) : (
                    /* AI Assistant Response Bubble */
                    <div className="flex items-start gap-3.5 max-w-3xl">
                      <div className="flex h-7 w-7 items-center justify-center rounded bg-[#2563EB] text-white font-bold text-xs shrink-0 mt-0.5">
                        AI
                      </div>

                      <div className="flex-1 space-y-3 min-w-0">
                        <div className="flex items-center gap-2 text-xs">
                          <span className="font-semibold text-white">AI Assistant</span>
                          <span className="text-[11px] text-[#717886] font-sans">
                            {msg.timestamp}
                          </span>
                          {msg.latencyMs && (
                            <>
                              <span className="text-[#525A6A]">·</span>
                              <span className="text-[11px] font-mono text-[#10B981]">
                                {msg.latencyMs}ms
                              </span>
                            </>
                          )}
                        </div>

                        {/* Message Body Content */}
                        <div className="text-xs text-[#E8EAF0] leading-relaxed space-y-3 font-sans">
                          {msg.content.split('\n\n').map((block, bIdx) => {
                            if (block.startsWith('### ')) {
                              return (
                                <h4 key={bIdx} className="font-semibold text-white text-sm mt-2 mb-1">
                                  {block.replace('### ', '')}
                                </h4>
                              );
                            }
                            if (block.startsWith('• ') || block.startsWith('- ') || block.startsWith('1. ')) {
                              return (
                                <div key={bIdx} className="pl-2 space-y-1 my-1">
                                  {block.split('\n').map((line, lIdx) => (
                                    <p key={lIdx} className="leading-relaxed">
                                      {line}
                                    </p>
                                  ))}
                                </div>
                              );
                            }
                            return (
                              <p key={bIdx} className="leading-relaxed">
                                {block}
                              </p>
                            );
                          })}

                          {/* Numbered Inline Citation Links */}
                          {msg.citations && msg.citations.length > 0 && (
                            <div className="flex flex-wrap items-center gap-1.5 pt-1">
                              <span className="text-[11px] text-[#717886] font-mono">
                                Cited Sources:
                              </span>
                              {msg.citations.map((c, cIdx) => {
                                const isCurrent = activeCitation?.id === c.id;
                                return (
                                  <button
                                    key={c.id}
                                    onClick={() => handleCitationClick(c)}
                                    className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium transition-all ${
                                      isCurrent
                                        ? 'bg-[#2563EB] text-white'
                                        : 'bg-[#181D26] text-[#3B82F6] border border-[#232834] hover:border-[#3B82F6]'
                                    }`}
                                  >
                                    Document {cIdx + 1}
                                  </button>
                                );
                              })}
                            </div>
                          )}
                        </div>

                        {/* SOURCES ACCORDION SECTION */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-[#1E232D] space-y-3">
                            <div className="flex items-center justify-between">
                              <button
                                onClick={() => setSourcesExpanded(!sourcesExpanded)}
                                className="flex items-center gap-2 text-xs font-semibold text-[#E8EAF0] hover:text-white"
                              >
                                <span>Sources ({msg.citations.length})</span>
                                {sourcesExpanded ? (
                                  <ChevronUp className="h-3.5 w-3.5 text-[#717886]" />
                                ) : (
                                  <ChevronDown className="h-3.5 w-3.5 text-[#717886]" />
                                )}
                              </button>

                              <div className="flex items-center gap-2">
                                <span className="text-[11px] font-mono text-[#F59E0B] bg-[#F59E0B]/10 px-2 py-0.5 rounded border border-[#F59E0B]/30 font-medium">
                                  Avg: {Math.round((msg.confidenceScore || 0.5) * 100)}%
                                </span>
                              </div>
                            </div>

                            {/* Detailed Citation Cards */}
                            {sourcesExpanded && (
                              <div className="space-y-2.5">
                                {msg.citations.map((cit, idx) => {
                                  const isExpanded = !!expandedCitationIds[cit.id];
                                  const isSelected = activeCitation?.id === cit.id;

                                  const pageLabel =
                                    cit.pageStart !== null && cit.pageStart !== undefined
                                      ? cit.pageEnd && cit.pageEnd !== cit.pageStart
                                        ? `pp. ${cit.pageStart}–${cit.pageEnd}`
                                        : `p. ${cit.pageStart}`
                                      : 'TXT Passage';

                                  return (
                                    <div
                                      key={cit.id}
                                      className={`rounded-xl bg-[#14171E] border transition-all overflow-hidden ${
                                        isSelected
                                          ? 'border-[#3B82F6]'
                                          : 'border-[#1E232D] hover:border-[#2A3140]'
                                      }`}
                                    >
                                      {/* Accordion Trigger Header */}
                                      <div
                                        onClick={() => toggleCitationAccordion(cit.id)}
                                        className="p-3 flex items-center justify-between cursor-pointer hover:bg-[#181D26] transition-colors"
                                      >
                                        <div className="flex items-center gap-2.5 min-w-0">
                                          <span className="flex h-5 w-5 items-center justify-center rounded bg-[#1C212C] text-[11px] font-mono text-[#8C93A3]">
                                            {idx + 1}
                                          </span>
                                          <FileText className="h-3.5 w-3.5 text-[#717886] shrink-0" />
                                          <span className="text-xs font-medium text-white truncate max-w-xs">
                                            {cit.docName}
                                          </span>
                                          <span className="text-[10px] font-mono text-[#717886]">
                                            {pageLabel}
                                          </span>
                                        </div>

                                        <div className="flex items-center gap-2.5">
                                          <span
                                            className={`text-[10px] font-mono px-2 py-0.5 rounded font-medium ${
                                              cit.similarityScore >= 0.70
                                                ? 'text-[#10B981] bg-[#10B981]/10 border border-[#10B981]/30'
                                                : cit.similarityScore >= 0.50
                                                ? 'text-[#38BDF8] bg-[#38BDF8]/10 border border-[#38BDF8]/30'
                                                : 'text-[#F59E0B] bg-[#F59E0B]/10 border border-[#F59E0B]/30'
                                            }`}
                                            title={cit.distance !== undefined ? `Distance: ${cit.distance.toFixed(4)}` : undefined}
                                          >
                                            ⚡ {(cit.similarityScore * 100).toFixed(0)}%
                                          </span>
                                          {isExpanded ? (
                                            <ChevronUp className="h-3.5 w-3.5 text-[#717886]" />
                                          ) : (
                                            <ChevronDown className="h-3.5 w-3.5 text-[#717886]" />
                                          )}
                                        </div>
                                      </div>

                                      {/* Accordion Content Body */}
                                      {isExpanded && (
                                        <div className="px-4 pb-3 space-y-3 border-t border-[#1E232D]/60 pt-3">
                                          <blockquote className="text-xs text-[#8C93A3] leading-relaxed italic border-l-2 border-[#3B82F6] pl-3 py-1">
                                            &quot;{cit.passage}&quot;
                                          </blockquote>

                                          <div className="flex items-center justify-between pt-1 text-[11px] text-[#717886]">
                                            <div className="flex items-center gap-2">
                                              <span className="text-[#8C93A3] truncate max-w-xs">
                                                {cit.sectionTitle}
                                              </span>
                                              <span className="text-[10px] text-[#10B981] bg-[#10B981]/10 px-1.5 py-0.2 rounded border border-[#10B981]/20">
                                                Verified
                                              </span>
                                            </div>

                                            <div className="flex items-center gap-3 shrink-0">
                                              <button
                                                onClick={() => handleCitationClick(cit)}
                                                className="text-xs text-[#3B82F6] hover:underline flex items-center gap-1 font-medium"
                                              >
                                                <span>View in PDF</span>
                                                <ExternalLink className="h-3 w-3" />
                                              </button>
                                              <button
                                                onClick={() => handleCopy(cit.passage, cit.id)}
                                                className="hover:text-white"
                                                title="Copy quote"
                                              >
                                                {copiedId === cit.id ? (
                                                  <Check className="h-3.5 w-3.5 text-[#10B981]" />
                                                ) : (
                                                  <Copy className="h-3.5 w-3.5" />
                                                )}
                                              </button>
                                            </div>
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* In-Flight Synthesizing Animation */}
          {isGeneratingResponse && (
            <div className="flex items-start gap-3.5">
              <div className="h-7 w-7 rounded bg-[#2563EB] text-white flex items-center justify-center font-bold text-xs shrink-0">
                AI
              </div>
              <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-3 text-xs text-[#8C93A3] flex items-center gap-2.5">
                <span className="h-2 w-2 rounded-full bg-[#3B82F6] animate-ping" />
                <span>Scanning {selectedDocIds.length} scoped documents & synthesizing citation proofs...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Question Input Box (Ask SIFT...) */}
        <div className="p-4 border-t border-[#1E232D] bg-[#0E1015]">
          <form onSubmit={handleSubmit} className="relative">
            <div className="rounded-xl border border-[#232834] bg-[#14171E] focus-within:border-[#3B82F6] transition-colors p-3">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask SIFT anything about your scoped documents..."
                rows={2}
                className="w-full bg-transparent text-xs text-white placeholder-[#717886] outline-none resize-none leading-relaxed"
              />

              <div className="flex items-center justify-between pt-2 border-t border-[#1E232D] mt-1">
                <div className="flex items-center gap-2 text-[11px] text-[#717886] font-mono">
                  <span className="hidden sm:inline">
                    Querying {selectedDocIds.length} docs
                  </span>
                  <span className="hidden sm:inline">·</span>
                  <kbd className="bg-[#1C212C] px-1.5 py-0.5 rounded border border-[#282F3E] text-[10px]">
                    Enter
                  </kbd>
                  <span className="hidden sm:inline">to submit</span>
                </div>

                <button
                  type="submit"
                  disabled={!inputMessage.trim() || isGeneratingResponse}
                  className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    inputMessage.trim() && !isGeneratingResponse
                      ? 'bg-[#2563EB] hover:bg-[#1D4ED8] text-white shadow-sm'
                      : 'bg-[#181D26] text-[#525A6A] cursor-not-allowed'
                  }`}
                >
                  <span>Query</span>
                  <Send className="h-3 w-3" />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* ======================================================== */}
      {/* 3. RIGHT PANEL: Source Evidence & PDF Page Inspector   */}
      {/* ======================================================== */}
      {showRightPanel && (
        <>
          {/* Mobile Backdrop */}
          <div
            onClick={() => setShowRightPanel(false)}
            className="fixed inset-0 z-25 bg-black/60 backdrop-blur-xs md:hidden"
            aria-hidden="true"
          />
          <aside className="fixed inset-y-16 right-0 z-30 w-full sm:w-96 md:static md:z-auto shrink-0 border-l border-[#1E232D] bg-[#0E1015] flex flex-col h-full shadow-2xl md:shadow-none animate-in slide-in-from-right-2 duration-150">
            {/* Tabs: PDF Inspector vs. All Chunks */}
            <div className="p-2 border-b border-[#1E232D] bg-[#0E1015]">
              <div className="flex items-center justify-between gap-1">
                <div className="flex flex-1 rounded-lg bg-[#14171E] border border-[#232834] p-0.5 text-xs font-medium">
                  <button
                    onClick={() => setActiveRightTab('pdfPreview')}
                    className={`flex-1 py-1 rounded-md transition-colors ${
                      activeRightTab === 'pdfPreview'
                        ? 'bg-[#1E232D] text-white shadow-xs'
                        : 'text-[#8C93A3] hover:text-white'
                    }`}
                  >
                    PDF Inspector
                  </button>
                  <button
                    onClick={() => setActiveRightTab('passages')}
                    className={`flex-1 py-1 rounded-md transition-colors ${
                      activeRightTab === 'passages'
                        ? 'bg-[#1E232D] text-white shadow-xs'
                        : 'text-[#8C93A3] hover:text-white'
                    }`}
                  >
                    Extracted Passages
                  </button>
                </div>

                <button
                  onClick={() => setShowRightPanel(false)}
                  aria-label="Close evidence and PDF inspector"
                  className="md:hidden text-[#717886] hover:text-white p-1.5 rounded-lg hover:bg-[#181D26]"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

          {activeRightTab === 'pdfPreview' ? (
            /* TAB 1: PDF PAGE CANVAS INSPECTOR */
            <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#0A0C10]">
              {/* Document Meta & Zoom Bar */}
              <div className="px-3.5 py-2.5 bg-[#0E1015] border-b border-[#1E232D] flex items-center justify-between text-xs">
                <div className="min-w-0">
                  <p className="font-semibold text-white truncate max-w-[160px]">
                    {activeCitation ? activeCitation.docName : 'No citation selected'}
                  </p>
                  <p className="text-[10px] font-mono text-[#717886]">
                    {activeCitation?.pageStart !== null && activeCitation?.pageStart !== undefined
                      ? `Page ${activeCitation.pageStart}${activeCitation.pageEnd && activeCitation.pageEnd !== activeCitation.pageStart ? `–${activeCitation.pageEnd}` : ''} of ${activeDocForCitation?.pageCount || 1}`
                      : 'TXT Document (No pages)'}
                  </p>
                </div>

                <div className="flex items-center gap-1 text-[11px] font-mono text-[#8C93A3]">
                  <button
                    onClick={() => setZoomLevel((z) => Math.max(z - 10, 80))}
                    className="p-1 rounded hover:bg-[#181D26]"
                    title="Zoom Out"
                  >
                    <ZoomOut className="h-3.5 w-3.5" />
                  </button>
                  <span>{zoomLevel}%</span>
                  <button
                    onClick={() => setZoomLevel((z) => Math.min(z + 10, 130))}
                    className="p-1 rounded hover:bg-[#181D26]"
                    title="Zoom In"
                  >
                    <ZoomIn className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Document Page Canvas Container */}
              <div className="flex-1 overflow-y-auto p-4 flex justify-center bg-[#07080B]">
                <div
                  className="w-full bg-[#12151B] border border-[#1E232D] rounded-lg shadow-2xl p-5 text-xs text-[#E8EAF0] relative space-y-4"
                  style={{
                    transform: `scale(${zoomLevel / 100})`,
                    transformOrigin: 'top center',
                  }}
                >
                  {/* Top Page Header Stamp */}
                  <div className="border-b border-[#1E232D] pb-3 text-center space-y-1">
                    <span className="text-[9px] font-mono text-[#525A6A] uppercase tracking-widest block">
                      CONFIDENTIAL · ENTERPRISE KNOWLEDGE CORPUS
                    </span>
                    <h4 className="text-xs font-bold text-white uppercase">
                      {activeCitation?.docName || 'SIFT DOCUMENT REPOSITORY'}
                    </h4>
                    <p className="text-[10px] font-mono text-[#8C93A3]">
                      {activeCitation?.sectionTitle || 'Source Provenance'}
                    </p>
                  </div>

                  {/* Preceding context text */}
                  <div className="space-y-2 text-[11px] text-[#525A6A] leading-relaxed select-none">
                    <p>
                      Verified document coordinates mapped from persistent storage and local all-MiniLM-L6-v2 embeddings.
                    </p>
                  </div>

                  {/* HIGHLIGHTED CITATION BOUNDING BOX */}
                  {activeCitation ? (
                    <div className="rounded-lg bg-[#3B82F6]/10 border-2 border-[#3B82F6] p-3.5 text-xs text-white leading-relaxed shadow-sm relative">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[9px] font-mono text-[#3B82F6] font-bold uppercase">
                          MATCHED CITATION EXCERPT
                        </span>
                        <span className="text-[9px] font-mono text-[#10B981] bg-[#10B981]/15 px-1.5 py-0.2 rounded border border-[#10B981]/30">
                          {(activeCitation.similarityScore * 100).toFixed(0)}% RELEVANCE
                        </span>
                      </div>
                      <p className="leading-relaxed font-normal">{activeCitation.passage}</p>
                    </div>
                  ) : (
                    <div className="p-4 rounded-lg border border-dashed border-[#1E232D] text-center text-xs text-[#717886]">
                      Click any citation pill in the center conversation to inspect the verified passage on this page.
                    </div>
                  )}

                  {/* Trailing context */}
                  <div className="space-y-2 text-[11px] text-[#525A6A] leading-relaxed select-none">
                    <p>
                      Grounded answer synthesis anchors claims strictly to retrieved passage tokens.
                    </p>
                  </div>

                  {/* Page Footer */}
                  <div className="pt-4 border-t border-[#1E232D] flex items-center justify-between text-[10px] font-mono text-[#717886]">
                    <span>
                      {activeCitation?.pageStart !== null && activeCitation?.pageStart !== undefined
                        ? `Page ${activeCitation.pageStart} of ${activeDocForCitation?.pageCount || 1}`
                        : 'TXT Passage'}
                    </span>
                    <span>SHA-256 Provenance Validated</span>
                  </div>
                </div>
              </div>

              {/* Bottom Citation Actions */}
              {activeCitation && (
                <div className="p-3 bg-[#0E1015] border-t border-[#1E232D] flex items-center justify-between text-xs">
                  <div className="font-mono text-[10px] text-[#717886] truncate max-w-[200px]">
                    Chunk: {activeCitation.chunkId}
                  </div>
                  <button
                    onClick={() => handleCopy(activeCitation.passage, activeCitation.id)}
                    className="flex items-center gap-1.5 text-xs text-[#3B82F6] hover:text-[#60A5FA]"
                  >
                    {copiedId === activeCitation.id ? (
                      <>
                        <Check className="h-3.5 w-3.5 text-[#10B981]" />
                        <span className="text-[#10B981]">Copied Quote</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3.5 w-3.5" />
                        <span>Copy Quote</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* TAB 2: EXTRACTED PASSAGES CARDS */
            <div className="flex-1 overflow-y-auto p-3.5 space-y-3 bg-[#0E1015]">
              {chatMessages
                .flatMap((m) => m.citations || [])
                .map((cit) => {
                  const isSelected = activeCitation?.id === cit.id;
                  const pageLabel =
                    cit.pageStart !== null && cit.pageStart !== undefined
                      ? cit.pageEnd && cit.pageEnd !== cit.pageStart
                        ? `pp. ${cit.pageStart}–${cit.pageEnd}`
                        : `p. ${cit.pageStart}`
                      : 'TXT Passage';

                  return (
                    <div
                      key={cit.id}
                      onClick={() => handleCitationClick(cit)}
                      className={`p-3.5 rounded-xl cursor-pointer transition-all border ${
                        isSelected
                          ? 'bg-[#181C26] border-[#3B82F6]'
                          : 'bg-[#14171E] border-[#1E232D] hover:border-[#2A3140]'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="font-medium text-white truncate max-w-[170px]">
                          {cit.docName}
                        </span>
                        <span className="font-mono text-[10px] text-[#F59E0B] bg-[#F59E0B]/10 px-1.5 py-0.2 rounded border border-[#F59E0B]/30 font-medium">
                          {(cit.similarityScore * 100).toFixed(0)}%
                        </span>
                      </div>

                      <div className="text-[10px] font-mono text-[#3B82F6] mb-1">
                        {cit.sectionTitle} · {pageLabel}
                      </div>

                      <p className="text-xs text-[#8C93A3] leading-relaxed line-clamp-3">
                        {cit.passage}
                      </p>

                      <div className="mt-2.5 pt-2 border-t border-[#1E232D] flex items-center justify-between text-[10px] font-mono text-[#717886]">
                        <span className="truncate max-w-[140px]">{cit.chunkId}</span>
                        <span className="text-[#3B82F6] hover:underline flex items-center gap-1">
                          View in PDF →
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </aside>
      </>
    )}
    </div>
  );
};
