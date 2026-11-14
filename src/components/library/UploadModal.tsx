'use client';

import React, { useState, useRef } from 'react';
import {
  X,
  UploadCloud,
  FileText,
  Trash2,
  CheckCircle2,
  Loader2,
  AlertCircle,
  ArrowRight,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { DocumentTag } from '@/types';

interface StagedFile {
  id: string;
  file: File;
  name: string;
  sizeFormatted: string;
  sizeBytes: number;
  type: 'pdf' | 'txt';
  progress: number;
  stage?: 'OCR & Layout' | 'Chunking' | 'Embedding' | 'Indexing';
  status: 'staged' | 'uploading' | 'processing' | 'indexed' | 'failed';
  tag: DocumentTag;
  errorMessage?: string;
}

export const UploadModal: React.FC = () => {
  const { isUploadModalOpen, setIsUploadModalOpen, uploadDocument } = useDocumentContext();

  const [dragActive, setDragActive] = useState(false);
  const [stagedFiles, setStagedFiles] = useState<StagedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedTag, setSelectedTag] = useState<DocumentTag>('Technical');
  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isUploadModalOpen && !isUploading) {
        setIsUploadModalOpen(false);
        setStagedFiles([]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isUploadModalOpen, isUploading, setIsUploadModalOpen]);

  if (!isUploadModalOpen) return null;

  const formatFileSize = (bytes: number): string => {
    if (bytes >= 1048576) {
      return (bytes / 1048576).toFixed(1) + ' MB';
    }
    return Math.max(1, Math.round(bytes / 1024)) + ' KB';
  };

  const processFiles = (files: FileList | File[]) => {
    const newStaged: StagedFile[] = [];

    Array.from(files).forEach((f) => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      
      // Reject unsupported file formats
      if (ext !== 'pdf' && ext !== 'txt') {
        newStaged.push({
          id: `stage-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          file: f,
          name: f.name,
          sizeFormatted: formatFileSize(f.size),
          sizeBytes: f.size,
          type: 'pdf',
          progress: 0,
          status: 'failed',
          tag: selectedTag,
          errorMessage: `Unsupported format (.${ext || 'unknown'}). SIFT currently supports PDF and TXT.`,
        });
        return;
      }

      // Reject oversized files (> 10 MB default backend limit)
      if (f.size > 10 * 1024 * 1024) {
        newStaged.push({
          id: `stage-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          file: f,
          name: f.name,
          sizeFormatted: formatFileSize(f.size),
          sizeBytes: f.size,
          type: ext as 'pdf' | 'txt',
          progress: 0,
          status: 'failed',
          tag: selectedTag,
          errorMessage: `File size (${(f.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 10 MB limit.`,
        });
        return;
      }

      newStaged.push({
        id: `stage-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        file: f,
        name: f.name,
        sizeFormatted: formatFileSize(f.size),
        sizeBytes: f.size,
        type: ext as 'pdf' | 'txt',
        progress: 0,
        status: 'staged',
        tag: selectedTag,
      });
    });

    setStagedFiles((prev) => [...prev, ...newStaged]);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
      e.target.value = '';
    }
  };

  const handleRemoveStaged = (id: string) => {
    setStagedFiles((prev) => prev.filter((item) => item.id !== id));
  };

  const handleStartIngestion = async () => {
    const filesToUpload = stagedFiles.filter((s) => s.status === 'staged');
    if (filesToUpload.length === 0 || isUploading) return;
    setIsUploading(true);

    let hasFailures = false;

    for (const staged of filesToUpload) {
      setStagedFiles((prev) =>
        prev.map((s) => (s.id === staged.id ? { ...s, status: 'uploading', progress: 20 } : s))
      );

      try {
        await uploadDocument({
          file: staged.file,
          tag: staged.tag,
          onProgress: (stage, progress) => {
            setStagedFiles((prev) =>
              prev.map((s) =>
                s.id === staged.id
                  ? {
                      ...s,
                      stage,
                      status: progress >= 100 ? 'indexed' : 'processing',
                      progress,
                    }
                  : s
              )
            );
          },
        });

        setStagedFiles((prev) =>
          prev.map((s) =>
            s.id === staged.id ? { ...s, status: 'indexed', progress: 100 } : s
          )
        );
      } catch (err: unknown) {
        hasFailures = true;
        const errMsg = err instanceof Error ? err.message : 'Ingestion failed';
        setStagedFiles((prev) =>
          prev.map((s) =>
            s.id === staged.id
              ? { ...s, status: 'failed', progress: 0, errorMessage: errMsg }
              : s
          )
        );
      }
    }

    setIsUploading(false);

    // If all files ingested without errors, close modal after short delay
    if (!hasFailures) {
      setTimeout(() => {
        setIsUploadModalOpen(false);
        setStagedFiles([]);
      }, 1000);
    }
  };

  const availableTags: DocumentTag[] = [
    'Technical',
    'Financial',
    'Legal',
    'Security',
    'Research',
    'Compliance',
  ];

  const hasStagedFilesToUpload = stagedFiles.some((s) => s.status === 'staged');

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in-0 duration-150"
    >
      <div
        className="w-full max-w-xl rounded-xl bg-[#14171E] border border-[#1E232D] shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-[#1E232D] px-6 py-4 bg-[#14171E]">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6]">
              <UploadCloud className="h-4 w-4" aria-hidden="true" />
            </div>
            <div>
              <h3 id="upload-modal-title" className="text-sm font-semibold text-white">
                Upload Documents to SIFT
              </h3>
              <p className="text-[11px] text-[#717886]">
                Supported formats: PDF, TXT (up to 10 MB per file)
              </p>
            </div>
          </div>
          <button
            onClick={() => {
              if (!isUploading) {
                setIsUploadModalOpen(false);
                setStagedFiles([]);
              }
            }}
            disabled={isUploading}
            aria-label="Close upload dialog"
            className="text-[#717886] hover:text-white p-1 rounded-lg hover:bg-[#1E232D] disabled:opacity-50"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Real Drag & Drop Zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-7 text-center cursor-pointer transition-all ${
              dragActive
                ? 'border-[#3B82F6] bg-[#3B82F6]/5 scale-[0.99]'
                : 'border-[#232834] bg-[#0E1015] hover:border-[#384152] hover:bg-[#11141C]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt"
              className="hidden"
              onChange={handleFileInputChange}
            />

            <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-[#182338] border border-[#23354E] text-[#3B82F6] mb-3">
              <UploadCloud className="h-5 w-5" />
            </div>

            <p className="text-xs font-medium text-white font-sans">
              Drag & drop PDF or TXT files here, or{' '}
              <span className="text-[#3B82F6] underline underline-offset-2">browse files</span>
            </p>
            <p className="text-[11px] text-[#717886] mt-1 font-sans">
              Files are automatically parsed, chunked, and embedded into persistent vector storage
            </p>
          </div>

          {/* Domain Tag Selection for batch */}
          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-[#717886] mb-1.5">
              Target Knowledge Domain
            </label>
            <div className="flex flex-wrap gap-1.5">
              {availableTags.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setSelectedTag(t)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                    selectedTag === t
                      ? 'bg-[#2563EB] text-white shadow-xs'
                      : 'bg-[#181D26] text-[#8C93A3] border border-[#232834] hover:text-white'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Staged Files List */}
          {stagedFiles.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-[#8C93A3]">
                <span className="font-medium text-white">
                  Files Ready for Ingestion ({stagedFiles.length})
                </span>
                <span className="text-[11px] font-mono text-[#717886]">
                  Total: {(stagedFiles.reduce((acc, f) => acc + f.sizeBytes, 0) / 1048576).toFixed(2)} MB
                </span>
              </div>

              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {stagedFiles.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 rounded-lg bg-[#0E1015] border border-[#1E232D] flex flex-col gap-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <FileText className="h-4 w-4 text-[#3B82F6] shrink-0" />
                        <span className="text-xs font-medium text-[#E8EAF0] truncate">
                          {item.name}
                        </span>
                        <span className="text-[10px] font-mono text-[#8C93A3] uppercase bg-[#181D26] px-1.5 py-0.2 rounded border border-[#232834]">
                          {item.type}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <span className="text-[11px] font-mono text-[#717886]">
                          {item.sizeFormatted}
                        </span>

                        {/* Status indicators */}
                        {item.status === 'staged' && (
                          <button
                            onClick={() => handleRemoveStaged(item.id)}
                            disabled={isUploading}
                            className="text-[#717886] hover:text-red-400 p-1"
                            title="Remove file"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                        {item.status === 'uploading' && (
                          <span className="text-[10px] font-mono text-[#3B82F6] flex items-center gap-1">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Uploading
                          </span>
                        )}
                        {item.status === 'processing' && (
                          <span className="text-[10px] font-mono text-[#F59E0B] flex items-center gap-1">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            {item.stage || 'Processing'}
                          </span>
                        )}
                        {item.status === 'indexed' && (
                          <span className="text-[10px] font-mono text-[#10B981] flex items-center gap-1">
                            <CheckCircle2 className="h-3.5 w-3.5 text-[#10B981]" />
                            Indexed
                          </span>
                        )}
                        {item.status === 'failed' && (
                          <span className="text-[10px] font-mono text-red-400 flex items-center gap-1">
                            <AlertCircle className="h-3.5 w-3.5 text-red-400" />
                            Failed
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Error message */}
                    {item.errorMessage && (
                      <p className="text-[11px] text-red-400 font-sans flex items-center gap-1">
                        <span>{item.errorMessage}</span>
                      </p>
                    )}

                    {/* Progress Bar when uploading */}
                    {item.progress > 0 && (
                      <div className="h-1 w-full bg-[#181D26] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#2563EB] rounded-full transition-all duration-300"
                          style={{ width: `${item.progress}%` }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-[#1E232D] px-6 py-3.5 bg-[#14171E] flex items-center justify-between">
          <button
            type="button"
            disabled={isUploading}
            onClick={() => {
              setIsUploadModalOpen(false);
              setStagedFiles([]);
            }}
            className="px-4 py-2 rounded-lg text-xs text-[#8C93A3] hover:text-white hover:bg-[#1E232D] transition-colors disabled:opacity-50"
          >
            Cancel
          </button>

          <button
            type="button"
            disabled={!hasStagedFilesToUpload || isUploading}
            onClick={handleStartIngestion}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-medium text-white transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isUploading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Ingesting to Vector Store...</span>
              </>
            ) : (
              <>
                <span>Start Ingestion</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
