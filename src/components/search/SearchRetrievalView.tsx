'use client';

import React, { useState, useMemo } from 'react';
import {
  Search,
  FileText,
  ArrowRight,
  MessageSquare,
  Copy,
  Check,
  X,
  Layers,
  Zap,
  Clock,
  AlertCircle,
  RotateCcw,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { SearchResult, DocumentType } from '@/types';

export const SearchRetrievalView: React.FC = () => {
  const {
    searchResults,
    searchQuery,
    similarityThreshold,
    setSimilarityThreshold,
    isSearching,
    searchError,
    searchExecutionTimeMs,
    performSearch,
    quickAskDocument,
    documents,
    selectedDocIds,
    sendChatMessage,
    setActiveTab,
  } = useDocumentContext();

  const [inputVal, setInputVal] = useState(
    searchQuery || 'Explain the key security risks discussed across my uploaded documents.'
  );
  const [selectedResultId, setSelectedResultId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedPassageIds, setExpandedPassageIds] = useState<Set<string>>(new Set());

  // Filters State
  const [selectedDocFilter, setSelectedDocFilter] = useState<string>('all');
  const [selectedTypeFilter, setSelectedTypeFilter] = useState<string>('all');
  const [searchMode, setSearchMode] = useState<'hybrid' | 'dense' | 'bm25'>('dense');
  const [resultLimit, setResultLimit] = useState<number>(10);
  const [sortBy, setSortBy] = useState<'score-desc' | 'score-asc' | 'date-newest' | 'page-asc'>('score-desc');

  const exampleQueries = [
    'Explain the key security risks discussed across my uploaded documents.',
    'What is the deductible for motor vehicle windshield damage?',
    'What are the data encryption and retention policies under compliance?',
    'Compare sliding-window chunking vs fixed-size segmentation.',
  ];

  const executeSearch = async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed) return;

    const docFilter = selectedDocFilter !== 'all' ? [selectedDocFilter] : undefined;
    await performSearch(trimmed, {
      documentIds: docFilter,
      topK: resultLimit,
      threshold: similarityThreshold,
      mode: searchMode === 'dense' ? 'semantic' : searchMode,
    });
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void executeSearch(inputVal);
  };

  const handleExampleClick = (query: string) => {
    setInputVal(query);
    void executeSearch(query);
  };

  const handleRetrySearch = () => {
    void executeSearch(inputVal || searchQuery);
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleExpandPassage = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedPassageIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleAskAboutResult = (res: SearchResult) => {
    const matchedDoc = documents.find((d) => d.id === res.docId);
    if (matchedDoc) {
      quickAskDocument(matchedDoc);
    } else {
      setActiveTab('chat');
    }
    sendChatMessage(`Based on ${res.sectionTitle} in ${res.docName}, explain: ${res.passage.slice(0, 140)}...`);
  };

  // Filter and Sort Logic
  const filteredAndSortedResults = useMemo(() => {
    return searchResults
      .filter((res) => {
        const meetsThreshold = res.score >= similarityThreshold;
        const meetsDoc = selectedDocFilter === 'all' || res.docId === selectedDocFilter;
        const meetsType = selectedTypeFilter === 'all' || res.docType === selectedTypeFilter;
        return meetsThreshold && meetsDoc && meetsType;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'score-asc':
            return a.score - b.score;
          case 'page-asc':
            return (a.pageStart ?? a.pageNumber) - (b.pageStart ?? b.pageNumber);
          case 'date-newest':
            return (b.id || '').localeCompare(a.id || '');
          case 'score-desc':
          default:
            return b.score - a.score;
        }
      })
      .slice(0, resultLimit);
  }, [
    searchResults,
    similarityThreshold,
    selectedDocFilter,
    selectedTypeFilter,
    sortBy,
    resultLimit,
  ]);

  // Derived selected result: stays pinned if present, falls back to top match, or null if none
  const selectedResult = useMemo(() => {
    if (filteredAndSortedResults.length === 0) return null;
    if (selectedResultId) {
      const found = filteredAndSortedResults.find((r) => r.id === selectedResultId);
      if (found) return found;
    }
    return filteredAndSortedResults[0];
  }, [filteredAndSortedResults, selectedResultId]);

  const getFileTypeBadge = (type?: DocumentType) => {
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
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6 bg-[#0E1015]">
      {/* 1. Header Section */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-sans">
            Search & Retrieval
          </h1>
          <p className="text-xs text-[#8C93A3] mt-1 font-sans">
            Natural-language semantic vector retrieval and neural passage inspection
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {searchExecutionTimeMs !== null && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-mono text-[#10B981] bg-[#10B981]/10 border border-[#10B981]/20">
              <Clock className="h-3 w-3" aria-hidden="true" />
              {searchExecutionTimeMs.toFixed(1)}ms
            </span>
          )}
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono text-[#3B82F6] bg-[#182338] border border-[#23354E]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#3B82F6]" aria-hidden="true" />
            sentence-transformers/all-MiniLM-L6-v2 · 384d
          </span>
        </div>
      </div>

      {/* 2. Search Input Box */}
      <div className="space-y-3">
        <form onSubmit={handleSearchSubmit}>
          <div className="flex items-center gap-2 sm:gap-3 rounded-xl bg-[#14171E] border border-[#232834] px-3 sm:px-4 py-2.5 sm:py-3 focus-within:border-[#3B82F6] transition-colors shadow-lg">
            <Search className="h-5 w-5 text-[#717886] shrink-0" aria-hidden="true" />
            <input
              type="text"
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              placeholder="Search across all indexed documents using natural language..."
              aria-label="Natural-language document search query"
              className="flex-1 bg-transparent text-xs sm:text-sm text-white placeholder-[#717886] outline-none min-w-0"
            />
            {inputVal && (
              <button
                type="button"
                onClick={() => setInputVal('')}
                aria-label="Clear search input"
                className="text-[#717886] hover:text-white p-1 rounded-lg hover:bg-[#1E232D]"
              >
                <X className="h-4 w-4" />
              </button>
            )}
            <button
              type="submit"
              disabled={isSearching || !inputVal.trim()}
              aria-label="Retrieve Passages"
              className="px-3.5 sm:px-5 py-1.5 sm:py-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-semibold text-white transition-colors shadow-sm disabled:opacity-50 shrink-0"
            >
              {isSearching ? 'Retrieving...' : 'Retrieve Passages'}
            </button>
          </div>
        </form>

        {/* Example Queries & Scoped Scope Pill */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-1 text-xs text-[#8C93A3]">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-mono text-[#525A6A] uppercase mr-1">
              Try Inquiries:
            </span>
            {exampleQueries.map((q, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleExampleClick(q)}
                className="text-[11px] px-2.5 py-1 rounded-lg bg-[#14171E] hover:bg-[#181D26] border border-[#1E232D] text-[#8C93A3] hover:text-white transition-colors truncate max-w-xs"
              >
                {q}
              </button>
            ))}
          </div>

          <span className="text-[11px] font-mono text-[#717886] shrink-0">
            Scoped to {selectedDocIds.length} of {documents.length} documents
          </span>
        </div>
      </div>

      {/* 3. Search Filters Toolbar */}
      <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-4 flex flex-wrap items-center justify-between gap-4 text-xs">
        {/* Left Side: Filter Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Search Mode Toggle */}
          <div className="flex rounded-lg bg-[#0E1015] border border-[#232834] p-0.5 text-[11px] font-mono">
            <button
              onClick={() => setSearchMode('dense')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                searchMode === 'dense'
                  ? 'bg-[#1E232D] text-[#3B82F6] font-medium'
                  : 'text-[#8C93A3] hover:text-white'
              }`}
            >
              Dense Vectors
            </button>
            <button
              onClick={() => setSearchMode('hybrid')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                searchMode === 'hybrid'
                  ? 'bg-[#1E232D] text-[#3B82F6] font-medium'
                  : 'text-[#8C93A3] hover:text-white'
              }`}
            >
              Hybrid Blend
            </button>
            <button
              onClick={() => setSearchMode('bm25')}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                searchMode === 'bm25'
                  ? 'bg-[#1E232D] text-[#3B82F6] font-medium'
                  : 'text-[#8C93A3] hover:text-white'
              }`}
            >
              BM25 Lexical
            </button>
          </div>

          {/* Document Filter */}
          <div className="flex items-center gap-1.5 bg-[#0E1015] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Scope:</span>
            <select
              value={selectedDocFilter}
              onChange={(e) => setSelectedDocFilter(e.target.value)}
              className="bg-transparent text-white outline-none cursor-pointer max-w-[150px] truncate"
            >
              <option value="all" className="bg-[#14171E] text-white">All Documents</option>
              {documents.map((d) => (
                <option key={d.id} value={d.id} className="bg-[#14171E] text-white">
                  {d.name} {d.status === 'indexed' ? '✓' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* File Type Filter */}
          <div className="flex items-center gap-1.5 bg-[#0E1015] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Type:</span>
            <select
              value={selectedTypeFilter}
              onChange={(e) => setSelectedTypeFilter(e.target.value)}
              className="bg-transparent text-white outline-none cursor-pointer"
            >
              <option value="all" className="bg-[#14171E] text-white">All Formats</option>
              <option value="pdf" className="bg-[#14171E] text-white">PDF</option>
              <option value="txt" className="bg-[#14171E] text-white">TXT</option>
              <option value="md" className="bg-[#14171E] text-white">MD</option>
            </select>
          </div>

          {/* Sort By Dropdown */}
          <div className="flex items-center gap-1.5 bg-[#0E1015] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) =>
                setSortBy(
                  e.target.value as 'score-desc' | 'score-asc' | 'date-newest' | 'page-asc'
                )
              }
              className="bg-transparent text-white outline-none cursor-pointer"
            >
              <option value="score-desc" className="bg-[#14171E] text-white">Relevance (High to Low)</option>
              <option value="score-asc" className="bg-[#14171E] text-white">Relevance (Low to High)</option>
              <option value="page-asc" className="bg-[#14171E] text-white">Page Number</option>
              <option value="date-newest" className="bg-[#14171E] text-white">Date (Newest)</option>
            </select>
          </div>
        </div>

        {/* Right Side: Relevance Threshold Slider & Limit */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-white font-medium text-[11px] font-mono">
              Similarity &ge; {(similarityThreshold * 100).toFixed(0)}%
            </span>
            <input
              type="range"
              min="0.10"
              max="0.95"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-28 accent-[#3B82F6] cursor-pointer"
            />
          </div>

          <div className="flex items-center gap-1.5 bg-[#0E1015] border border-[#232834] rounded-lg px-2 py-1 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Limit:</span>
            <select
              value={resultLimit}
              onChange={(e) => setResultLimit(parseInt(e.target.value, 10))}
              className="bg-transparent text-white outline-none cursor-pointer font-mono"
            >
              <option value={5} className="bg-[#14171E] text-white">Top 5</option>
              <option value={10} className="bg-[#14171E] text-white">Top 10</option>
              <option value={20} className="bg-[#14171E] text-white">Top 20</option>
            </select>
          </div>
        </div>
      </div>

      {/* 4. Results Section: Left List + Right Contextual Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Retrieved Passages List (7 cols) */}
        <div className="lg:col-span-7 space-y-3">
          {searchError ? (
            /* Error State */
            <div className="rounded-xl bg-[#14171E] border border-red-500/30 p-8 text-center space-y-3">
              <div className="mx-auto h-10 w-10 rounded-full bg-red-500/10 text-red-400 flex items-center justify-center">
                <AlertCircle className="h-5 w-5" />
              </div>
              <h3 className="text-sm font-semibold text-white">Search Retrieval Failed</h3>
              <p className="text-xs text-[#8C93A3] max-w-md mx-auto">{searchError}</p>
              <button
                onClick={handleRetrySearch}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-semibold text-white transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Retry Search</span>
              </button>
            </div>
          ) : isSearching ? (
            /* Loading Skeleton State */
            <div className="space-y-3">
              <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-4 text-center">
                <div className="flex items-center justify-center gap-2 text-xs text-[#3B82F6]">
                  <Zap className="h-4 w-4 animate-spin" />
                  <span className="font-medium text-white">Retrieving passages from ChromaDB...</span>
                </div>
              </div>
              {[1, 2, 3].map((n) => (
                <div
                  key={n}
                  className="rounded-xl p-4 bg-[#14171E] border border-[#1E232D] space-y-2.5 animate-pulse"
                >
                  <div className="flex items-center justify-between">
                    <div className="h-4 w-48 bg-[#1E232D] rounded" />
                    <div className="h-4 w-20 bg-[#1E232D] rounded-full" />
                  </div>
                  <div className="h-3 w-32 bg-[#1A1F29] rounded" />
                  <div className="space-y-1.5 pt-1">
                    <div className="h-3 w-full bg-[#1E232D] rounded" />
                    <div className="h-3 w-5/6 bg-[#1E232D] rounded" />
                    <div className="h-3 w-2/3 bg-[#1A1F29] rounded" />
                  </div>
                  <div className="pt-2 border-t border-[#1E232D] flex justify-between">
                    <div className="h-3 w-24 bg-[#1E232D] rounded" />
                    <div className="h-3 w-16 bg-[#1E232D] rounded" />
                  </div>
                </div>
              ))}
            </div>
          ) : searchResults.length === 0 && !searchQuery ? (
            /* Initial Welcoming State */
            <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-12 text-center space-y-3">
              <div className="mx-auto h-10 w-10 rounded-full bg-[#182338] text-[#3B82F6] flex items-center justify-center">
                <Search className="h-5 w-5" aria-hidden="true" />
              </div>
              <h3 className="text-sm font-semibold text-white">Ready to Search Documents</h3>
              <p className="text-xs text-[#8C93A3] max-w-md mx-auto">
                Enter a natural-language inquiry above or pick one of the sample inquiries to retrieve relevant passages with verified source coordinates.
              </p>
            </div>
          ) : filteredAndSortedResults.length === 0 ? (
            /* Empty State */
            <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-10 text-center space-y-3">
              <div className="mx-auto h-9 w-9 rounded-full bg-[#181D26] text-[#717886] flex items-center justify-center">
                <Search className="h-4 w-4" aria-hidden="true" />
              </div>
              <p className="text-xs text-white font-medium">
                No passages found matching similarity &ge; {(similarityThreshold * 100).toFixed(0)}%
              </p>
              <p className="text-[11px] text-[#8C93A3] max-w-sm mx-auto">
                Try lowering the relevance threshold slider, choosing &ldquo;All Documents&rdquo; in the scope dropdown, or ensuring your documents are indexed.
              </p>
              <div className="flex items-center justify-center gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setSimilarityThreshold(0.20)}
                  className="px-3 py-1.5 rounded-lg bg-[#182338] hover:bg-[#23354E] border border-[#23354E] text-xs text-[#60A5FA] font-medium transition-colors"
                >
                  Lower Threshold to 20%
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setSelectedDocFilter('all');
                    setSelectedTypeFilter('all');
                    setSimilarityThreshold(0.30);
                  }}
                  className="px-3 py-1.5 rounded-lg bg-[#14171E] hover:bg-[#1E232D] border border-[#232834] text-xs text-[#8C93A3] hover:text-white transition-colors"
                >
                  Reset Filters
                </button>
              </div>
            </div>
          ) : (
            filteredAndSortedResults.map((res, idx) => {
              const isSelected = selectedResult?.id === res.id;
              const typeBadge = getFileTypeBadge(res.docType);
              const isExpanded = expandedPassageIds.has(res.id);
              const distanceVal = res.cosineDistance ?? res.distance ?? 0;
              const scoreVal = res.similarityScore ?? res.score ?? 0;

              // Format honest page reference
              const pageLabel =
                res.docType === 'pdf' && res.pageStart !== null && res.pageStart !== undefined
                  ? res.pageEnd && res.pageEnd !== res.pageStart
                    ? `pp. ${res.pageStart}–${res.pageEnd}`
                    : `p. ${res.pageStart}`
                  : 'TXT Passage';

              return (
                <div
                  key={res.id}
                  onClick={() => setSelectedResultId(res.id)}
                  className={`rounded-xl p-4 cursor-pointer transition-all border ${
                    isSelected
                      ? 'bg-[#181D26] border-[#3B82F6] shadow-md'
                      : 'bg-[#14171E] border-[#1E232D] hover:border-[#2A3140]'
                  }`}
                >
                  {/* Top Result Card Metadata Header */}
                  <div className="flex items-center justify-between text-xs mb-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="flex h-5 w-5 items-center justify-center rounded bg-[#1C212C] text-[10px] font-mono text-[#8C93A3]">
                        #{idx + 1}
                      </span>
                      <FileText className="h-3.5 w-3.5 text-[#717886] shrink-0" />
                      <span className="font-semibold text-white truncate max-w-xs">
                        {res.docName}
                      </span>
                      <span
                        className={`text-[9px] font-mono px-1.5 py-0.2 rounded uppercase border ${typeBadge}`}
                      >
                        {res.docType || 'PDF'}
                      </span>
                      <span className="text-[10px] font-mono text-[#717886]">
                        {pageLabel}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <span
                        className="font-mono text-[10px] px-2 py-0.5 rounded text-[#8C93A3] bg-[#0E1015] border border-[#232834]"
                        title="ChromaDB Cosine Distance: Lower indicates closer match"
                      >
                        Distance: {distanceVal.toFixed(4)}
                      </span>
                      <span
                        className={`font-mono text-[10px] px-2 py-0.5 rounded font-medium ${
                          scoreVal >= 0.70
                            ? 'text-[#10B981] bg-[#10B981]/10 border border-[#10B981]/30'
                            : scoreVal >= 0.50
                            ? 'text-[#38BDF8] bg-[#38BDF8]/10 border border-[#38BDF8]/30'
                            : 'text-[#F59E0B] bg-[#F59E0B]/10 border border-[#F59E0B]/30'
                        }`}
                        title="Similarity Score: Computed as 1 - cosine distance"
                      >
                        {(scoreVal * 100).toFixed(1)}% match
                      </span>
                    </div>
                  </div>

                  {/* Section Title */}
                  <div className="text-[11px] font-mono text-[#3B82F6] mb-1.5">
                    {res.sectionTitle}
                  </div>

                  {/* Extracted Passage with Expand/Collapse */}
                  <p
                    className={`text-xs text-[#8C93A3] leading-relaxed font-sans ${
                      isExpanded ? '' : 'line-clamp-3'
                    }`}
                  >
                    {res.passage}
                  </p>

                  {res.passage.length > 220 && (
                    <button
                      type="button"
                      onClick={(e) => toggleExpandPassage(res.id, e)}
                      className="mt-1 text-[11px] font-mono text-[#3B82F6] hover:underline"
                    >
                      {isExpanded ? 'Show less' : 'Show full passage'}
                    </button>
                  )}

                  {/* Highlights Anchors Strip */}
                  {res.matchHighlights.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                      {res.matchHighlights.map((term, tIdx) => (
                        <span
                          key={tIdx}
                          className="text-[10px] font-mono text-[#60A5FA] bg-[#1E293B] px-1.5 py-0.2 rounded border border-[#2B3B52]"
                        >
                          {term}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Card Actions Footer */}
                  <div className="mt-3 pt-2.5 border-t border-[#1E232D] flex items-center justify-between text-[11px] text-[#717886]">
                    <span className="font-mono text-[10px]">
                      {res.chunkId || `chunk-${res.chunkIndex ?? 0}`} · {res.tokenCount} tokens
                    </span>

                    <div className="flex items-center gap-3">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAskAboutResult(res);
                        }}
                        className="text-[#3B82F6] hover:underline flex items-center gap-1 font-medium text-xs"
                      >
                        <MessageSquare className="h-3 w-3" />
                        <span>Add to Chat</span>
                      </button>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleCopy(res.passage, res.id);
                        }}
                        className="hover:text-white"
                        title="Copy passage"
                      >
                        {copiedId === res.id ? (
                          <Check className="h-3.5 w-3.5 text-[#10B981]" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Right Column: Contextual Retrieval Inspector Panel (5 cols) */}
        <div className="lg:col-span-5">
          {selectedResult ? (
            <div className="sticky top-6 rounded-xl bg-[#14171E] border border-[#1E232D] p-5 space-y-4 shadow-xl">
              {/* Header */}
              <div className="flex items-center justify-between border-b border-[#1E232D] pb-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#182338] text-[#3B82F6]">
                    <Layers className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="text-xs font-semibold text-white">Retrieval Inspector</h3>
                    <p className="text-[10px] font-mono text-[#717886]">Vector Coordinates & Provenance</p>
                  </div>
                </div>

                <span className="font-mono text-xs text-[#10B981] bg-[#10B981]/10 px-2 py-0.5 rounded border border-[#10B981]/30 font-semibold">
                  {((selectedResult.similarityScore ?? selectedResult.score ?? 0) * 100).toFixed(1)}% Match
                </span>
              </div>

              {/* Technical Telemetry Strip */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-[#0E1015] border border-[#1E232D] p-2.5 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[#525A6A]">Cosine Distance</span>
                    <span className="text-[9px] text-[#3B82F6]">Lower is closer</span>
                  </div>
                  <span className="text-white font-medium mt-0.5 block">
                    {(selectedResult.cosineDistance ?? selectedResult.distance ?? 0).toFixed(4)}
                  </span>
                </div>
                <div className="bg-[#0E1015] border border-[#1E232D] p-2.5 rounded-lg">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-[#525A6A]">Similarity Score</span>
                    <span className="text-[9px] text-[#717886]">1 - Distance</span>
                  </div>
                  <span className="text-white font-medium mt-0.5 block">
                    {((selectedResult.similarityScore ?? selectedResult.score ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Source Document Context */}
              <div className="space-y-1.5 text-xs">
                <span className="text-[10px] font-mono text-[#525A6A] uppercase tracking-wider block">
                  Source Provenance
                </span>
                <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[#717886]">Document:</span>
                    <span className="text-white font-medium truncate max-w-[170px] text-right">
                      {selectedResult.docName}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#717886]">Page Reference:</span>
                    <span className="text-white font-mono">
                      {selectedResult.docType === 'pdf' && selectedResult.pageStart !== null && selectedResult.pageStart !== undefined
                        ? selectedResult.pageEnd && selectedResult.pageEnd !== selectedResult.pageStart
                          ? `Pages ${selectedResult.pageStart}–${selectedResult.pageEnd}`
                          : `Page ${selectedResult.pageStart}`
                        : 'TXT Passage (No pages)'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#717886]">Section:</span>
                    <span className="text-[#8C93A3] truncate max-w-[170px] text-right">
                      {selectedResult.sectionTitle}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#717886]">Chunk ID:</span>
                    <span className="text-[#3B82F6] font-mono truncate max-w-[170px] text-right">
                      {selectedResult.chunkId}
                    </span>
                  </div>
                  {selectedResult.chunkIndex !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-[#717886]">Chunk Index:</span>
                      <span className="text-white font-mono">#{selectedResult.chunkIndex}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Full Extracted Text Passage Box */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[10px] font-mono text-[#525A6A] uppercase tracking-wider">
                    Full Extracted Passage
                  </span>
                  <button
                    onClick={() => handleCopy(selectedResult.passage, selectedResult.id)}
                    className="text-[#3B82F6] hover:text-[#60A5FA] text-[11px] flex items-center gap-1 font-mono"
                  >
                    {copiedId === selectedResult.id ? (
                      <>
                        <Check className="h-3 w-3 text-[#10B981]" />
                        <span className="text-[#10B981]">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3 w-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
                <div className="p-3.5 rounded-lg bg-[#0E1015] border border-[#1E232D] text-xs text-[#E8EAF0] leading-relaxed max-h-48 overflow-y-auto">
                  {selectedResult.passage}
                </div>
              </div>

              {/* Retrieval Config Snapshot */}
              <div className="bg-[#0E1015] border border-[#1E232D] p-2.5 rounded-lg text-[10px] font-mono text-[#717886] space-y-1">
                <div className="flex justify-between">
                  <span>Vector Store:</span>
                  <span className="text-white">ChromaDB (Persistent)</span>
                </div>
                <div className="flex justify-between">
                  <span>Embedding Model:</span>
                  <span className="text-white">all-MiniLM-L6-v2 (384d)</span>
                </div>
                <div className="flex justify-between">
                  <span>Distance Metric:</span>
                  <span className="text-white">Cosine</span>
                </div>
                <div className="flex justify-between">
                  <span>Execution Time:</span>
                  <span className="text-white">
                    {searchExecutionTimeMs !== null ? `${searchExecutionTimeMs.toFixed(1)} ms` : '< 15 ms'}
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-2">
                <button
                  onClick={() => handleAskAboutResult(selectedResult)}
                  className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-semibold text-white shadow-sm transition-colors"
                >
                  <MessageSquare className="h-3.5 w-3.5" />
                  <span>Synthesize with AI in Chat</span>
                  <ArrowRight className="h-3 w-3" />
                </button>
              </div>
            </div>
          ) : (
            <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-8 text-center text-xs text-[#717886]">
              Select a retrieved passage from the left to inspect its vector coordinates and token provenance.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
