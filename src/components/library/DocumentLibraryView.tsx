'use client';

import React, { useState, useMemo } from 'react';
import {
  Search,
  Upload,
  Trash2,
  MessageSquare,
  Eye,
  X,
  Edit2,
  FileText,
  ArrowUpDown,
  CheckSquare,
  Square,
  Loader2,
  AlertCircle,
  RefreshCw,
  Database,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { DocumentItem, DocumentType } from '@/types';
import { DocumentPreviewModal } from './DocumentPreviewModal';
import { RenameModal } from './RenameModal';
import { DeleteConfirmModal } from './DeleteConfirmModal';

export const DocumentLibraryView: React.FC = () => {
  const {
    documents,
    isLoadingDocuments,
    documentsError,
    refreshDocuments,
    deleteDocument,
    renameDocument,
    quickAskDocument,
    setIsUploadModalOpen,
    processingDocIds,
    processDocumentPipeline,
  } = useDocumentContext();

  const [searchTerm, setSearchTerm] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [sortOption, setSortOption] = useState<
    'newest' | 'oldest' | 'name-asc' | 'name-desc' | 'size-desc' | 'size-asc'
  >('newest');

  // Selected row IDs for batch actions
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [isBatchDeleting, setIsBatchDeleting] = useState(false);
  const [isBatchIndexing, setIsBatchIndexing] = useState(false);

  // Modals state
  const [previewDoc, setPreviewDoc] = useState<DocumentItem | null>(null);
  const [renameDoc, setRenameDoc] = useState<DocumentItem | null>(null);
  const [deleteDoc, setDeleteDoc] = useState<DocumentItem | null>(null);

  // Storage calculations
  const totalStorageBytes = useMemo(() => {
    return documents.reduce((acc, d) => acc + (d.sizeBytes || 1048576), 0);
  }, [documents]);

  const storageUsedMB = (totalStorageBytes / (1024 * 1024)).toFixed(1);
  const maxStorageMB = 500;
  const storagePercentage = Math.min(100, (parseFloat(storageUsedMB) / maxStorageMB) * 100);

  // Filter & Sort Logic
  const filteredAndSortedDocs = useMemo(() => {
    return documents
      .filter((doc) => {
        const matchesSearch =
          doc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          doc.description.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesType = selectedType === 'all' || doc.type === selectedType;

        const isIndexed = doc.status === 'indexed' && (doc.vectorCount > 0 || doc.chunksCount > 0);
        const isCurrentlyProcessing = Boolean(processingDocIds[doc.id]);
        const isFailed = doc.status === 'error';
        const isUnindexed = !isIndexed && !isCurrentlyProcessing && !isFailed;

        let matchesStatus = true;
        if (selectedStatus === 'indexed') matchesStatus = isIndexed;
        else if (selectedStatus === 'unindexed') matchesStatus = isUnindexed;
        else if (selectedStatus === 'processing') matchesStatus = isCurrentlyProcessing;
        else if (selectedStatus === 'error') matchesStatus = isFailed;

        return matchesSearch && matchesType && matchesStatus;
      })
      .sort((a, b) => {
        switch (sortOption) {
          case 'name-asc':
            return a.name.localeCompare(b.name);
          case 'name-desc':
            return b.name.localeCompare(a.name);
          case 'size-desc':
            return (b.sizeBytes || 0) - (a.sizeBytes || 0);
          case 'size-asc':
            return (a.sizeBytes || 0) - (b.sizeBytes || 0);
          case 'oldest':
            return a.id.localeCompare(b.id);
          case 'newest':
          default:
            return b.id.localeCompare(a.id);
        }
      });
  }, [documents, searchTerm, selectedType, selectedStatus, sortOption, processingDocIds]);

  const toggleSelectRow = (id: string) => {
    setSelectedRowIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const toggleSelectAll = () => {
    if (selectedRowIds.length === filteredAndSortedDocs.length) {
      setSelectedRowIds([]);
    } else {
      setSelectedRowIds(filteredAndSortedDocs.map((d) => d.id));
    }
  };

  const handleBatchIndex = async () => {
    if (isBatchIndexing || selectedRowIds.length === 0) return;
    setIsBatchIndexing(true);
    try {
      for (const id of selectedRowIds) {
        try {
          await processDocumentPipeline(id);
        } catch (err: unknown) {
          console.error(`Error indexing document ${id}:`, err);
        }
      }
    } finally {
      setIsBatchIndexing(false);
    }
  };

  const handleBatchDelete = async () => {
    if (isBatchDeleting || selectedRowIds.length === 0) return;
    setIsBatchDeleting(true);
    try {
      for (const id of selectedRowIds) {
        await deleteDocument(id);
      }
      setSelectedRowIds([]);
    } finally {
      setIsBatchDeleting(false);
    }
  };

  const getFileTypeBadge = (type: DocumentType) => {
    switch (type) {
      case 'pdf':
        return {
          label: 'PDF',
          bg: 'bg-red-500/10 text-red-400 border-red-500/20',
          iconColor: 'text-red-400',
        };
      case 'txt':
        return {
          label: 'TXT',
          bg: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
          iconColor: 'text-blue-400',
        };
      default:
        return {
          label: (type as string).toUpperCase(),
          bg: 'bg-gray-500/10 text-gray-300 border-gray-500/20',
          iconColor: 'text-gray-400',
        };
    }
  };

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6 bg-[#0E1015]">
      {/* 1. Header Section */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-sans">
            Document Library
          </h1>
          <p className="text-xs text-[#8C93A3] mt-1 font-sans">
            Manage, organize and explore your knowledge base.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={refreshDocuments}
            disabled={isLoadingDocuments}
            aria-label="Refresh document list"
            className="flex items-center gap-1.5 rounded-lg bg-[#14171E] hover:bg-[#1E232D] border border-[#232834] px-3 py-2 text-xs font-medium text-[#8C93A3] hover:text-white transition-colors disabled:opacity-50"
            title="Refresh document list"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoadingDocuments ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span className="hidden sm:inline">Refresh</span>
          </button>

          <button
            onClick={() => setIsUploadModalOpen(true)}
            aria-label="Upload document"
            className="flex items-center gap-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] px-4 py-2 text-xs font-medium text-white shadow-sm transition-colors shrink-0"
          >
            <Upload className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Upload Document</span>
          </button>
        </div>
      </div>

      {/* Backend Error Banner if unreachable */}
      {documentsError && (
        <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4 flex items-start justify-between gap-3 text-xs text-red-200">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
            <span>{documentsError}</span>
          </div>
          <button
            onClick={refreshDocuments}
            className="px-3 py-1 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-200 text-xs font-medium shrink-0 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* 2. Corpus Statistics & Storage Strip */}
      <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs">
        <div className="flex flex-wrap items-center gap-6 text-[#8C93A3]">
          <div>
            <span className="text-[#525A6A] font-mono uppercase text-[10px] block">
              Total Documents
            </span>
            <span className="text-sm font-semibold text-white font-mono">
              {documents.length} files
            </span>
          </div>

          <div className="h-6 w-px bg-[#1E232D] hidden sm:block" />

          <div>
            <span className="text-[#525A6A] font-mono uppercase text-[10px] block">
              Indexed Chunks
            </span>
            <span className="text-sm font-semibold text-white font-mono">
              {documents.reduce((acc, d) => acc + d.chunksCount, 0).toLocaleString()} vectors
            </span>
          </div>

          <div className="h-6 w-px bg-[#1E232D] hidden sm:block" />

          <div>
            <span className="text-[#525A6A] font-mono uppercase text-[10px] block">
              Knowledge Volume
            </span>
            <span className="text-sm font-semibold text-white font-mono">
              {documents.reduce((acc, d) => acc + d.tokenCount, 0).toLocaleString()} tokens
            </span>
          </div>
        </div>

        {/* Storage Bar Indicator */}
        <div className="w-full md:w-64 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] font-mono">
            <span className="text-[#717886]">Storage Utilization</span>
            <span className="text-white font-medium">
              {storageUsedMB} MB / {maxStorageMB} MB
            </span>
          </div>
          <div className="h-1.5 w-full bg-[#0E1015] rounded-full overflow-hidden border border-[#1E232D]">
            <div
              className="h-full bg-[#2563EB] rounded-full transition-all duration-300"
              style={{ width: `${Math.max(4, storagePercentage)}%` }}
            />
          </div>
        </div>
      </div>

      {/* 3. Search and Filter Bar */}
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3">
        {/* Search Input */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-[#717886]" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search documents by name, keyword, or metadata..."
            className="w-full rounded-lg bg-[#14171E] border border-[#232834] pl-9 pr-8 py-2 text-xs text-[#E8EAF0] placeholder-[#717886] outline-none focus:border-[#3B82F6] transition-colors"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-2.5 top-2.5 text-[#717886] hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        {/* Filters and Sorting Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* File Type Filter */}
          <div className="flex items-center gap-1.5 bg-[#14171E] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Type:</span>
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="bg-transparent text-white outline-none cursor-pointer"
            >
              <option value="all" className="bg-[#14171E] text-white">All Types</option>
              <option value="pdf" className="bg-[#14171E] text-white">PDF Documents</option>
              <option value="txt" className="bg-[#14171E] text-white">TXT Plaintext</option>
            </select>
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-1.5 bg-[#14171E] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <span className="text-[#525A6A] font-mono text-[10px] uppercase">Status:</span>
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="bg-transparent text-white outline-none cursor-pointer"
            >
              <option value="all" className="bg-[#14171E] text-white">All Statuses</option>
              <option value="indexed" className="bg-[#14171E] text-white">Indexed (Vectors Ready)</option>
              <option value="unindexed" className="bg-[#14171E] text-white">Unindexed</option>
              <option value="processing" className="bg-[#14171E] text-white">Processing...</option>
              <option value="error" className="bg-[#14171E] text-white">Failed</option>
            </select>
          </div>

          {/* Sort By Dropdown */}
          <div className="flex items-center gap-1.5 bg-[#14171E] border border-[#232834] rounded-lg px-2.5 py-1.5 text-xs text-[#8C93A3]">
            <ArrowUpDown className="h-3 w-3 text-[#717886]" />
            <select
              value={sortOption}
              onChange={(e) =>
                setSortOption(
                  e.target.value as
                    | 'newest'
                    | 'oldest'
                    | 'name-asc'
                    | 'name-desc'
                    | 'size-desc'
                    | 'size-asc'
                )
              }
              className="bg-transparent text-white outline-none cursor-pointer"
            >
              <option value="newest" className="bg-[#14171E] text-white">Upload Date (Newest)</option>
              <option value="oldest" className="bg-[#14171E] text-white">Upload Date (Oldest)</option>
              <option value="name-asc" className="bg-[#14171E] text-white">Name (A–Z)</option>
              <option value="name-desc" className="bg-[#14171E] text-white">Name (Z–A)</option>
              <option value="size-desc" className="bg-[#14171E] text-white">Size (Largest)</option>
              <option value="size-asc" className="bg-[#14171E] text-white">Size (Smallest)</option>
            </select>
          </div>
        </div>
      </div>

      {/* 4. Batch Selection Notification Bar */}
      {selectedRowIds.length > 0 && (
        <div className="flex items-center justify-between p-3 rounded-lg bg-[#182338] border border-[#23354E] text-xs text-white animate-in fade-in-0 duration-150">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#60A5FA] font-mono">
              {selectedRowIds.length} {selectedRowIds.length === 1 ? 'document' : 'documents'} selected
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                const first = documents.find((d) => d.id === selectedRowIds[0]);
                if (first) quickAskDocument(first);
              }}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#2563EB] hover:bg-[#1D4ED8] text-white text-xs font-medium transition-colors"
            >
              <MessageSquare className="h-3.5 w-3.5" />
              <span>Ask in Chat</span>
            </button>
            <button
              onClick={handleBatchIndex}
              disabled={isBatchIndexing || isBatchDeleting}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-[#182338] hover:bg-[#23354E] border border-[#23354E] text-[#60A5FA] text-xs font-medium transition-colors disabled:opacity-50"
            >
              {isBatchIndexing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Database className="h-3.5 w-3.5" />
              )}
              <span>{isBatchIndexing ? 'Indexing...' : 'Index Selected'}</span>
            </button>
            <button
              onClick={handleBatchDelete}
              disabled={isBatchDeleting}
              className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-red-600 hover:bg-red-500 text-white text-xs font-medium transition-colors disabled:opacity-50"
            >
              {isBatchDeleting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              <span>{isBatchDeleting ? 'Deleting...' : 'Delete Selected'}</span>
            </button>
            <button
              onClick={() => setSelectedRowIds([])}
              className="px-2 py-1 text-xs text-[#8C93A3] hover:text-white"
            >
              Deselect
            </button>
          </div>
        </div>
      )}

      {/* 5. Document Management Data Table */}
      <div className="rounded-xl bg-[#14171E] border border-[#1E232D] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-[#1E232D] text-[#717886] font-medium font-sans bg-[#12151B]">
                <th className="py-3.5 px-4 w-10">
                  <button
                    onClick={toggleSelectAll}
                    aria-label={selectedRowIds.length === filteredAndSortedDocs.length && filteredAndSortedDocs.length > 0 ? 'Deselect all documents' : 'Select all documents'}
                    className="text-[#717886] hover:text-white flex items-center"
                  >
                    {selectedRowIds.length === filteredAndSortedDocs.length && filteredAndSortedDocs.length > 0 ? (
                      <CheckSquare className="h-4 w-4 text-[#3B82F6]" aria-hidden="true" />
                    ) : (
                      <Square className="h-4 w-4" aria-hidden="true" />
                    )}
                  </button>
                </th>
                <th className="py-3.5 px-3 font-normal">Document Name</th>
                <th className="py-3.5 px-3 font-normal">Type</th>
                <th className="py-3.5 px-3 font-normal">Size</th>
                <th className="py-3.5 px-3 font-normal">Uploaded</th>
                <th className="py-3.5 px-3 font-normal">Processing Status</th>
                <th className="py-3.5 px-4 font-normal text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1E232D]/60 font-sans">
              {isLoadingDocuments ? (
                <>
                  {[1, 2, 3].map((idx) => (
                    <tr key={idx} className="animate-pulse">
                      <td className="py-3.5 px-4">
                        <div className="h-4 w-4 rounded bg-[#1E232D]" />
                      </td>
                      <td className="py-3.5 px-3">
                        <div className="flex items-center gap-2.5">
                          <div className="h-4 w-4 rounded bg-[#1E232D] shrink-0" />
                          <div className="space-y-1.5 flex-1 max-w-xs">
                            <div className="h-3.5 w-3/4 bg-[#1E232D] rounded" />
                            <div className="h-2.5 w-1/2 bg-[#1A1F29] rounded" />
                          </div>
                        </div>
                      </td>
                      <td className="py-3.5 px-3">
                        <div className="h-4 w-10 bg-[#1E232D] rounded" />
                      </td>
                      <td className="py-3.5 px-3">
                        <div className="h-3.5 w-14 bg-[#1E232D] rounded" />
                      </td>
                      <td className="py-3.5 px-3">
                        <div className="h-3.5 w-20 bg-[#1E232D] rounded" />
                      </td>
                      <td className="py-3.5 px-3">
                        <div className="h-4 w-24 bg-[#1E232D] rounded-full" />
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="h-4 w-24 bg-[#1E232D] rounded ml-auto" />
                      </td>
                    </tr>
                  ))}
                </>
              ) : filteredAndSortedDocs.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-16 text-center text-[#717886]">
                    {documents.length === 0 ? (
                      <div className="flex flex-col items-center justify-center gap-3 max-w-sm mx-auto">
                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#182338] text-[#3B82F6]">
                          <FileText className="h-5 w-5" aria-hidden="true" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-white">No documents in your library yet</p>
                          <p className="text-xs text-[#717886] mt-1">
                            Upload your first PDF or TXT document to begin indexing and neural retrieval.
                          </p>
                        </div>
                        <button
                          onClick={() => setIsUploadModalOpen(true)}
                          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-medium text-white transition-colors"
                        >
                          <Upload className="h-3.5 w-3.5" aria-hidden="true" />
                          <span>Upload First Document</span>
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center gap-2 py-4">
                        <span>No documents match your query or filters.</span>
                        <button
                          onClick={() => {
                            setSearchTerm('');
                            setSelectedType('all');
                            setSelectedStatus('all');
                          }}
                          className="text-xs text-[#3B82F6] hover:underline"
                        >
                          Reset Filters
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ) : (
                filteredAndSortedDocs.map((doc) => {
                  const badge = getFileTypeBadge(doc.type);
                  const isSelected = selectedRowIds.includes(doc.id);
                  const isCurrentlyProcessing = Boolean(processingDocIds[doc.id]);
                  const currentStage = processingDocIds[doc.id];
                  const isIndexed = doc.status === 'indexed' && (doc.vectorCount > 0 || doc.chunksCount > 0);

                  return (
                    <tr
                      key={doc.id}
                      className={`hover:bg-[#181D26] transition-colors group ${
                        isSelected ? 'bg-[#182030]/60' : ''
                      }`}
                    >
                      {/* Checkbox */}
                      <td className="py-3 px-4">
                        <button
                          onClick={() => toggleSelectRow(doc.id)}
                          aria-label={isSelected ? `Deselect ${doc.name}` : `Select ${doc.name}`}
                          className="text-[#717886] hover:text-white flex items-center"
                        >
                          {isSelected ? (
                            <CheckSquare className="h-4 w-4 text-[#3B82F6]" aria-hidden="true" />
                          ) : (
                            <Square className="h-4 w-4" aria-hidden="true" />
                          )}
                        </button>
                      </td>

                      {/* Name & Details */}
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2.5">
                          <FileText
                            className={`h-4 w-4 shrink-0 transition-colors ${badge.iconColor}`}
                            aria-hidden="true"
                          />
                          <div className="min-w-0">
                            <button
                              onClick={() => setPreviewDoc(doc)}
                              className="font-medium text-[#E8EAF0] group-hover:text-white hover:underline truncate max-w-sm text-left block"
                            >
                              {doc.name}
                            </button>
                            <div className="text-[10px] text-[#717886] font-mono flex items-center gap-2 mt-0.5">
                              <span>{doc.chunksCount} chunks</span>
                              <span>·</span>
                              <span>{doc.tokenCount.toLocaleString()} tokens</span>
                              <span>·</span>
                              <span>{doc.tag}</span>
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* Type Badge */}
                      <td className="py-3 px-3">
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badge.bg}`}
                        >
                          {badge.label}
                        </span>
                      </td>

                      {/* Size */}
                      <td className="py-3 px-3 font-mono text-[#8C93A3]">
                        {doc.size}
                      </td>

                      {/* Uploaded */}
                      <td className="py-3 px-3 text-[#717886]">
                        {doc.uploadedAt}
                      </td>

                      {/* Status */}
                      <td className="py-3 px-3">
                        {isCurrentlyProcessing ? (
                          <span className="inline-flex items-center gap-1.5 text-xs text-[#3B82F6] font-medium">
                            <Loader2 className="h-3 w-3 animate-spin text-[#3B82F6]" aria-hidden="true" />
                            {currentStage === 'chunking'
                              ? 'Chunking...'
                              : currentStage === 'embedding'
                              ? 'Embedding...'
                              : currentStage === 'indexing'
                              ? 'Indexing...'
                              : 'Processing...'}
                          </span>
                        ) : isIndexed ? (
                          <span
                            className="inline-flex items-center gap-1.5 text-xs text-[#10B981] font-medium"
                            title={`${doc.vectorCount || doc.chunksCount} vectors indexed in ChromaDB`}
                          >
                            <span className="h-1.5 w-1.5 rounded-full bg-[#10B981]" aria-hidden="true" />
                            Indexed ({doc.vectorCount || doc.chunksCount})
                          </span>
                        ) : doc.status === 'error' ? (
                          <span className="inline-flex items-center gap-1.5 text-xs text-red-400 font-medium">
                            <span className="h-1.5 w-1.5 rounded-full bg-red-400" aria-hidden="true" />
                            Failed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 text-xs text-amber-400 font-medium">
                            <span className="h-1.5 w-1.5 rounded-full bg-amber-400" aria-hidden="true" />
                            Unindexed
                          </span>
                        )}
                      </td>

                      {/* Actions */}
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1 text-[#717886]">
                          <button
                            onClick={async () => {
                              try {
                                await processDocumentPipeline(doc.id, isIndexed);
                              } catch (err: unknown) {
                                console.error('Failed to process document:', err);
                              }
                            }}
                            disabled={isCurrentlyProcessing}
                            aria-label={isIndexed ? `Re-index ${doc.name} in ChromaDB` : `Process and index ${doc.name}`}
                            className="p-1.5 rounded-lg hover:bg-[#232834] hover:text-[#3B82F6] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                            title={isIndexed ? 'Re-index document in ChromaDB' : 'Process & index document'}
                          >
                            {isCurrentlyProcessing ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin text-[#3B82F6]" aria-hidden="true" />
                            ) : (
                              <Database className="h-3.5 w-3.5" aria-hidden="true" />
                            )}
                          </button>

                          <button
                            onClick={() => setPreviewDoc(doc)}
                            aria-label={`Preview ${doc.name}`}
                            className="p-1.5 rounded-lg hover:bg-[#232834] hover:text-white transition-colors"
                            title="Preview Document"
                          >
                            <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>

                          <button
                            onClick={() => quickAskDocument(doc)}
                            aria-label={`Open ${doc.name} in Document Chat`}
                            className="p-1.5 rounded-lg hover:bg-[#232834] hover:text-[#3B82F6] transition-colors"
                            title="Open in Document Chat"
                          >
                            <MessageSquare className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>

                          <button
                            onClick={() => setRenameDoc(doc)}
                            aria-label={`Rename ${doc.name}`}
                            className="p-1.5 rounded-lg hover:bg-[#232834] hover:text-white transition-colors"
                            title="Rename"
                          >
                            <Edit2 className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>

                          <button
                            onClick={() => setDeleteDoc(doc)}
                            aria-label={`Delete ${doc.name}`}
                            className="p-1.5 rounded-lg hover:bg-[#232834] hover:text-red-400 transition-colors"
                            title="Delete Document"
                          >
                            <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modals */}
      <DocumentPreviewModal
        key={previewDoc?.id || 'preview'}
        doc={previewDoc}
        onClose={() => setPreviewDoc(null)}
        onOpenChat={(doc) => quickAskDocument(doc)}
      />

      <RenameModal
        key={renameDoc?.id || 'rename'}
        doc={renameDoc}
        onClose={() => setRenameDoc(null)}
        onRename={(id, newName) => renameDocument(id, newName)}
      />

      <DeleteConfirmModal
        key={deleteDoc?.id || 'delete'}
        doc={deleteDoc}
        onClose={() => setDeleteDoc(null)}
        onConfirm={(id) => deleteDocument(id)}
      />
    </div>
  );
};
