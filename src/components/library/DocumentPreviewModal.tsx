'use client';

import React, { useState, useEffect } from 'react';
import {
  FileText,
  X,
  MessageSquare,
  Copy,
  Check,
  Layers,
  Calendar,
  HardDrive,
  FileCode,
  Loader2,
  Database,
} from 'lucide-react';
import { DocumentItem } from '@/types';
import { documentApi } from '@/lib/api';
import { IndexStatusResponse } from '@/lib/api/types';
import { useDocumentContext } from '@/context/DocumentContext';

interface DocumentPreviewModalProps {
  doc: DocumentItem | null;
  onClose: () => void;
  onOpenChat: (doc: DocumentItem) => void;
}

export const DocumentPreviewModal: React.FC<DocumentPreviewModalProps> = ({
  doc,
  onClose,
  onOpenChat,
}) => {
  const { processingDocIds, processDocumentPipeline } = useDocumentContext();

  const [copiedHash, setCopiedHash] = useState(false);
  const [fullText, setFullText] = useState<string | null>(null);
  const [isLoadingText, setIsLoadingText] = useState<boolean>(true);
  const [indexStatus, setIndexStatus] = useState<IndexStatusResponse | null>(null);
  const [isLoadingIndexStatus, setIsLoadingIndexStatus] = useState<boolean>(true);
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  const isCurrentDocProcessing = Boolean(doc?.id && processingDocIds[doc.id]);
  const currentStage = doc?.id ? processingDocIds[doc.id] : undefined;

  useEffect(() => {
    if (!doc?.id) return;

    let isMounted = true;

    // Fetch full extracted text
    documentApi
      .get(doc.id)
      .then((detail) => {
        if (isMounted) {
          setFullText(detail.extracted_text || null);
          setIsLoadingText(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setFullText(null);
          setIsLoadingText(false);
        }
      });

    // Fetch live ChromaDB index status
    documentApi
      .getIndexStatus(doc.id)
      .then((statusRes) => {
        if (isMounted) {
          setIndexStatus(statusRes);
          setIsLoadingIndexStatus(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setIndexStatus(null);
          setIsLoadingIndexStatus(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [doc?.id]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && doc) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [doc, onClose]);

  if (!doc) return null;

  const handleCopyHash = () => {
    navigator.clipboard.writeText(doc.hash);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
  };

  const handleReindex = async () => {
    if (!doc?.id || isCurrentDocProcessing) return;
    setPipelineError(null);
    try {
      await processDocumentPipeline(doc.id, true);
      const updatedStatus = await documentApi.getIndexStatus(doc.id);
      setIndexStatus(updatedStatus);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Indexing failed';
      setPipelineError(msg);
    }
  };

  const displayedPreviewText =
    fullText ||
    doc.rawTextPreview ||
    `No raw preview available for ${doc.name}. Document is fully indexed in the vector database.`;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="preview-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in-0 duration-150"
    >
      <div
        className="w-full max-w-2xl rounded-xl bg-[#14171E] border border-[#1E232D] shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1E232D] px-6 py-4 bg-[#14171E]">
          <div className="flex items-center gap-3 min-w-0">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6] shrink-0">
              <FileText className="h-5 w-5" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h3 id="preview-modal-title" className="text-sm font-semibold text-white truncate">
                {doc.name}
              </h3>
              <div className="flex items-center gap-2 text-[11px] text-[#717886] font-mono mt-0.5">
                <span className="text-[#8C93A3] uppercase">{doc.type}</span>
                <span>·</span>
                <span>{doc.size}</span>
                <span>·</span>
                <span>{doc.uploadedAt}</span>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            aria-label="Close document preview"
            className="text-[#717886] hover:text-white p-1.5 rounded-lg hover:bg-[#1E232D] transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Metadata Cards Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg">
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#525A6A] uppercase">
                <Layers className="h-3 w-3 text-[#3B82F6]" />
                <span>Chunks</span>
              </div>
              <div className="text-sm font-semibold text-white font-mono mt-1">
                {doc.chunksCount}
              </div>
              <span className="text-[10px] text-[#717886]">
                {indexStatus ? `${indexStatus.indexed_chunks} vectors in ChromaDB` : '384-dim vectors'}
              </span>
            </div>

            <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg">
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#525A6A] uppercase">
                <FileCode className="h-3 w-3 text-[#10B981]" />
                <span>Token Volume</span>
              </div>
              <div className="text-sm font-semibold text-white font-mono mt-1">
                {doc.tokenCount.toLocaleString()}
              </div>
              <span className="text-[10px] text-[#717886]">CL100K tokenized</span>
            </div>

            <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg">
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#525A6A] uppercase">
                <Calendar className="h-3 w-3 text-[#F59E0B]" />
                <span>Page Count</span>
              </div>
              <div className="text-sm font-semibold text-white font-mono mt-1">
                {doc.pageCount} pages
              </div>
              <span className="text-[10px] text-[#717886]">Parsed & verified</span>
            </div>

            <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg">
              <div className="flex items-center gap-1.5 text-[10px] font-mono text-[#525A6A] uppercase">
                <HardDrive className="h-3 w-3 text-[#38BDF8]" />
                <span>Domain</span>
              </div>
              <div className="text-sm font-semibold text-white mt-1">
                {doc.tag}
              </div>
              {indexStatus?.is_indexed ? (
                <span className="text-[10px] text-[#10B981] flex items-center gap-1 mt-0.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-[#10B981]" />
                  Indexed
                </span>
              ) : isCurrentDocProcessing ? (
                <span className="text-[10px] text-[#3B82F6] flex items-center gap-1 mt-0.5">
                  <Loader2 className="h-2.5 w-2.5 animate-spin text-[#3B82F6]" />
                  Processing
                </span>
              ) : (
                <span className="text-[10px] text-amber-400 flex items-center gap-1 mt-0.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                  Unindexed
                </span>
              )}
            </div>
          </div>

          {/* ChromaDB Vector Store & Model Verification */}
          <div className="bg-[#0E1015] border border-[#1E232D] p-3.5 rounded-lg space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-[#3B82F6]" />
                <span className="text-xs font-semibold text-white">ChromaDB Vector Indexing</span>
              </div>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                  indexStatus?.is_indexed
                    ? 'text-[#10B981] bg-[#10B981]/10 border-[#10B981]/30'
                    : isCurrentDocProcessing
                    ? 'text-[#60A5FA] bg-[#2563EB]/10 border-[#2563EB]/30'
                    : 'text-amber-400 bg-amber-400/10 border-amber-400/30'
                }`}
              >
                {isLoadingIndexStatus
                  ? 'CHECKING...'
                  : isCurrentDocProcessing
                  ? (currentStage ? currentStage.toUpperCase() : 'PROCESSING')
                  : indexStatus?.is_indexed
                  ? 'INDEXED'
                  : 'UNINDEXED'}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] font-mono text-[#8C93A3] pt-1">
              <div>
                <span className="text-[#525A6A] block text-[10px] uppercase">Collection</span>
                <span className="text-white truncate block">
                  {indexStatus?.collection_name || 'sift_document_embeddings'}
                </span>
              </div>
              <div>
                <span className="text-[#525A6A] block text-[10px] uppercase">Embedding Model</span>
                <span className="text-white truncate block">
                  {indexStatus?.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2'}
                </span>
              </div>
              <div>
                <span className="text-[#525A6A] block text-[10px] uppercase">Indexed Vectors</span>
                <span className="text-white block font-mono">
                  {indexStatus?.indexed_chunks ?? 0} / {indexStatus?.total_chunks ?? doc.chunksCount}
                </span>
              </div>
            </div>

            {pipelineError && (
              <div className="p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-[11px]">
                {pipelineError}
              </div>
            )}
          </div>

          {/* Document Summary / Description */}
          <div>
            <span className="text-[10px] font-mono text-[#525A6A] uppercase tracking-wider block mb-1">
              Document Abstract
            </span>
            <p className="text-xs text-[#8C93A3] leading-relaxed bg-[#0E1015] border border-[#1E232D] p-3.5 rounded-lg">
              {doc.description || `Ingested ${doc.type.toUpperCase()} document indexed in active ChromaDB collection.`}
            </p>
          </div>

          {/* Extracted Raw Content Preview */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-mono text-[#525A6A] uppercase tracking-wider">
                Parsed Text Preview
              </span>
              {isLoadingText && (
                <span className="text-[10px] font-mono text-[#3B82F6] flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading full text...
                </span>
              )}
            </div>
            <div className="bg-[#0E1015] border border-[#1E232D] p-4 rounded-lg font-mono text-xs text-[#E8EAF0] leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap">
              {displayedPreviewText}
            </div>
          </div>

          {/* Cryptographic Hash */}
          <div>
            <span className="text-[10px] font-mono text-[#525A6A] uppercase tracking-wider block mb-1">
              Cryptographic Provenance
            </span>
            <div className="flex items-center justify-between bg-[#0E1015] border border-[#1E232D] px-3 py-2 rounded-lg text-xs font-mono text-[#717886]">
              <span className="truncate mr-2">{doc.hash}</span>
              <button
                onClick={handleCopyHash}
                className="text-[#3B82F6] hover:text-[#60A5FA] shrink-0 flex items-center gap-1 text-[11px]"
              >
                {copiedHash ? (
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
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-[#1E232D] px-6 py-3.5 bg-[#14171E] flex items-center justify-between gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-xs text-[#8C93A3] hover:text-white hover:bg-[#1E232D] transition-colors"
          >
            Close Preview
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={handleReindex}
              disabled={isCurrentDocProcessing}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#182338] hover:bg-[#23354E] border border-[#23354E] text-xs font-medium text-[#60A5FA] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isCurrentDocProcessing ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>
                    {currentStage === 'chunking'
                      ? 'Chunking...'
                      : currentStage === 'embedding'
                      ? 'Embedding...'
                      : currentStage === 'indexing'
                      ? 'Indexing in ChromaDB...'
                      : 'Processing...'}
                  </span>
                </>
              ) : (
                <>
                  <Database className="h-3.5 w-3.5" />
                  <span>{indexStatus?.is_indexed ? 'Re-index Document' : 'Index in ChromaDB'}</span>
                </>
              )}
            </button>

            <button
              onClick={() => {
                onOpenChat(doc);
                onClose();
              }}
              className="flex items-center gap-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] px-4 py-2 text-xs font-medium text-white shadow-sm transition-colors"
            >
              <MessageSquare className="h-3.5 w-3.5" />
              <span>Open in Document Chat</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
