'use client';

import React from 'react';
import {
  FileText,
  Clock,
  ArrowRight,
  Eye,
  MoreVertical,
  CheckCircle2,
  Activity,
  Layers,
  Zap,
  ShieldCheck,
  MessageSquare,
  Loader2,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { RECENT_QUERIES } from '@/lib/mockData';

export const DashboardView: React.FC = () => {
  const {
    documents,
    activities,
    quickAskDocument,
    setActiveTab,
    sendChatMessage,
    setIsUploadModalOpen,
  } = useDocumentContext();

  const handleQueryClick = (queryText: string) => {
    setActiveTab('chat');
    sendChatMessage(queryText);
  };

  const totalChunks = documents.reduce((acc, d) => acc + d.chunksCount, 0);
  const totalSizeBytes = documents.reduce((acc, d) => acc + (d.sizeBytes || 0), 0);
  const totalStorageMB = (totalSizeBytes / (1024 * 1024)).toFixed(1);

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 space-y-6 bg-[#0E1015]">
      {/* 1. Header Row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-sans">
            Dashboard
          </h1>
          <p className="text-xs text-[#8C93A3] mt-1 font-sans">
            Your document intelligence workspace
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-full bg-[#121E19] border border-[#183928] px-3 py-1 text-xs text-[#10B981] font-medium font-sans">
            <span className="h-2 w-2 rounded-full bg-[#10B981]" aria-hidden="true" />
            <span>Ready</span>
          </div>
          <span className="text-xs text-[#717886] font-sans">
            Active SIFT Workspace
          </span>
        </div>
      </div>

      {/* 2. Top Metric Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Total Documents */}
        <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between hover:border-[#2A3140] transition-colors relative overflow-hidden">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6]">
              <FileText className="h-4 w-4" />
            </div>
            <span className="text-xs text-[#8C93A3] font-sans">Total Documents</span>
          </div>

          <div className="mt-4 flex items-baseline justify-between">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-bold text-white font-sans">
                {documents.length}
              </span>
              <span className="text-xs font-medium text-[#10B981] flex items-center">
                live
              </span>
            </div>

            {/* Blue sparkline */}
            <svg width="84" height="28" viewBox="0 0 84 28" fill="none" className="text-[#3B82F6] shrink-0">
              <path
                d="M2 22C12 22 18 12 28 12C38 12 44 24 54 20C64 16 70 4 82 4"
                stroke="#3B82F6"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Metric 2: Total Chunks */}
        <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between hover:border-[#2A3140] transition-colors relative overflow-hidden">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6]">
              <Layers className="h-4 w-4" />
            </div>
            <span className="text-xs text-[#8C93A3] font-sans">Total Chunks</span>
          </div>

          <div className="mt-4 flex items-baseline justify-between">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-bold text-white font-sans">
                {totalChunks.toLocaleString()}
              </span>
              <span className="text-xs font-medium text-[#10B981] flex items-center">
                indexed
              </span>
            </div>

            {/* Blue sparkline */}
            <svg width="84" height="28" viewBox="0 0 84 28" fill="none" className="text-[#3B82F6] shrink-0">
              <path
                d="M2 24C12 24 20 18 30 18C40 18 46 22 56 16C66 10 72 6 82 8"
                stroke="#3B82F6"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Metric 3: Avg Response Time */}
        <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between hover:border-[#2A3140] transition-colors relative overflow-hidden">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6]">
              <Zap className="h-4 w-4" />
            </div>
            <span className="text-xs text-[#8C93A3] font-sans">Avg Response Time</span>
          </div>

          <div className="mt-4 flex items-baseline justify-between">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-bold text-white font-sans">1.3s</span>
              <span className="text-xs font-medium text-[#38BDF8] flex items-center">
                optimal
              </span>
            </div>

            {/* Cyan sparkline */}
            <svg width="84" height="28" viewBox="0 0 84 28" fill="none" className="text-[#38BDF8] shrink-0">
              <path
                d="M2 6C14 6 22 20 34 20C46 20 54 8 66 14C74 18 78 22 82 22"
                stroke="#38BDF8"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>

        {/* Metric 4: Uptime */}
        <div className="rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between hover:border-[#2A3140] transition-colors relative overflow-hidden">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#14261F] border border-[#1D3B2E] text-[#10B981]">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <span className="text-xs text-[#8C93A3] font-sans">Uptime</span>
          </div>

          <div className="mt-4 flex items-baseline justify-between">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-bold text-white font-sans">99.9%</span>
            </div>

            {/* Green sparkline */}
            <svg width="84" height="28" viewBox="0 0 84 28" fill="none" className="text-[#10B981] shrink-0">
              <path
                d="M2 20C12 20 18 10 30 14C42 18 50 8 62 10C72 12 76 6 82 8"
                stroke="#10B981"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* 3. Middle Section: Recent Documents & Ingestion Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Recent Documents Table (8 cols) */}
        <div className="lg:col-span-8 rounded-xl bg-[#14171E] border border-[#1E232D] p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#182338] text-[#3B82F6]">
                <Clock className="h-3.5 w-3.5" />
              </div>
              <h2 className="text-sm font-semibold text-white font-sans">
                Recent Documents
              </h2>
            </div>

            <button
              onClick={() => setActiveTab('library')}
              className="text-xs text-[#3B82F6] hover:text-[#60A5FA] flex items-center gap-1 font-medium transition-colors"
            >
              <span>View all</span>
              <ArrowRight className="h-3 w-3" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-[#717886] font-medium border-b border-[#1E232D]">
                  <th className="pb-3 font-normal">Name</th>
                  <th className="pb-3 font-normal">Type</th>
                  <th className="pb-3 font-normal">Uploaded</th>
                  <th className="pb-3 font-normal">Status</th>
                  <th className="pb-3 font-normal text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1E232D]/60">
                {documents.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-[#717886]">
                      <div className="flex flex-col items-center justify-center gap-2 max-w-xs mx-auto">
                        <FileText className="h-6 w-6 text-[#525A6A]" aria-hidden="true" />
                        <span className="text-xs">No documents ingested yet. Upload your first PDF or TXT to get started.</span>
                        <button
                          onClick={() => setIsUploadModalOpen(true)}
                          className="mt-1 px-3 py-1.5 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-medium text-white transition-colors"
                        >
                          Upload Document
                        </button>
                      </div>
                    </td>
                  </tr>
                ) : (
                  documents.slice(0, 5).map((doc) => (
                    <tr key={doc.id} className="hover:bg-[#181C26] transition-colors group">
                      <td className="py-3 pr-2">
                        <div className="flex items-center gap-2.5">
                          <FileText className="h-4 w-4 text-[#717886] shrink-0 group-hover:text-[#3B82F6] transition-colors" aria-hidden="true" />
                          <span className="font-medium text-[#E8EAF0] truncate max-w-xs">
                            {doc.name}
                          </span>
                        </div>
                      </td>

                      <td className="py-3 px-2">
                        <span className="text-[10px] font-mono text-[#8C93A3] bg-[#1C212C] px-2 py-0.5 rounded border border-[#282F3E] uppercase">
                          {doc.type}
                        </span>
                      </td>

                      <td className="py-3 px-2 text-[#717886] text-xs">
                        {doc.uploadedAt}
                      </td>

                      <td className="py-3 px-2">
                        <span className="inline-flex items-center gap-1.5 text-xs text-[#10B981] font-medium">
                          <span className="h-1.5 w-1.5 rounded-full bg-[#10B981]" aria-hidden="true" />
                          Processed
                        </span>
                      </td>

                      <td className="py-3 pl-2 text-right">
                        <div className="flex items-center justify-end gap-1.5 text-[#717886]">
                          <button
                            onClick={() => quickAskDocument(doc)}
                            className="p-1 rounded-md hover:bg-[#232834] hover:text-[#E8EAF0] transition-colors"
                            aria-label={`View and query ${doc.name}`}
                            title="View / Ask"
                          >
                            <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>
                          <button
                            onClick={() => quickAskDocument(doc)}
                            className="p-1 rounded-md hover:bg-[#232834] hover:text-[#E8EAF0] transition-colors"
                            aria-label={`More options for ${doc.name}`}
                            title="More options"
                          >
                            <MoreVertical className="h-3.5 w-3.5" aria-hidden="true" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Ingestion Activity Card (4 cols) */}
        <div className="lg:col-span-4 rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2.5">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#182338] text-[#3B82F6]">
                  <Activity className="h-3.5 w-3.5" />
                </div>
                <h2 className="text-sm font-semibold text-white font-sans">
                  Ingestion Activity
                </h2>
              </div>

              <div className="flex items-center gap-1.5 text-xs text-[#10B981] font-medium">
                <span className="h-2 w-2 rounded-full bg-[#10B981] animate-pulse" />
                <span>Live</span>
              </div>
            </div>

            {/* Activity Stream */}
            <div className="space-y-4 pt-1">
              {activities.length === 0 ? (
                <p className="text-xs text-[#717886] py-4 text-center">
                  No recent ingestion jobs recorded.
                </p>
              ) : (
                activities.slice(0, 3).map((act) => (
                  <div key={act.id} className="flex items-start gap-3">
                    <div className="flex h-5 w-5 items-center justify-center rounded-full bg-[#10B981]/15 text-[#10B981] shrink-0 mt-0.5">
                      {act.status === 'in_progress' ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-[#3B82F6]" />
                      ) : (
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      )}
                    </div>
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-[#E8EAF0] truncate">
                        {act.docName}
                      </p>
                      <p className="text-[11px] text-[#717886]">
                        {act.status === 'in_progress' ? `${act.stage}... • ${act.progress}%` : `Processing completed • ${act.timestamp}`}
                      </p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Bottom Wave & Total Processed */}
          <div className="pt-6 mt-4 border-t border-[#1E232D]/70 flex items-center justify-between">
            <svg width="140" height="24" viewBox="0 0 140 24" fill="none" className="text-[#3B82F6]">
              <path
                d="M2 14C12 14 18 4 28 8C38 12 44 20 54 14C64 8 70 18 80 12C90 6 98 16 108 12C118 8 126 18 138 6"
                stroke="#3B82F6"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
            <div className="text-right">
              <span className="text-xs font-bold text-white font-sans">{totalStorageMB} MB</span>
              <p className="text-[10px] text-[#717886]">Total Processed</p>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Bottom Section: Recent Queries & Feature Callout Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left: Recent Queries (6 cols) */}
        <div className="lg:col-span-6 rounded-xl bg-[#14171E] border border-[#1E232D] p-5">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[#182338] text-[#3B82F6]">
                <MessageSquare className="h-3.5 w-3.5" />
              </div>
              <h2 className="text-sm font-semibold text-white font-sans">
                Recent Queries
              </h2>
            </div>

            <button
              onClick={() => setActiveTab('chat')}
              className="text-xs text-[#3B82F6] hover:text-[#60A5FA] flex items-center gap-1 font-medium transition-colors"
            >
              <span>Explore Chat</span>
              <ArrowRight className="h-3 w-3" />
            </button>
          </div>

          <div className="space-y-2">
            {RECENT_QUERIES.map((q) => (
              <button
                key={q.id}
                onClick={() => handleQueryClick(q.query)}
                className="w-full flex items-center justify-between p-2.5 rounded-lg bg-[#0E1015] hover:bg-[#181D26] border border-[#1E232D] text-left text-xs transition-colors group"
              >
                <div className="min-w-0 pr-2">
                  <p className="text-[#E8EAF0] group-hover:text-white truncate font-medium">
                    {q.query}
                  </p>
                  <p className="text-[10px] text-[#717886] font-mono mt-0.5">
                    {q.targetDoc} · Score: {q.score} · {q.timestamp}
                  </p>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#525A6A] group-hover:text-[#3B82F6] group-hover:translate-x-0.5 transition-all shrink-0" />
              </button>
            ))}
          </div>
        </div>

        {/* Right: Architecture & Security Card (6 cols) */}
        <div className="lg:col-span-6 rounded-xl bg-[#14171E] border border-[#1E232D] p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono uppercase tracking-wider text-[#3B82F6] font-semibold">
                Architecture & Privacy
              </span>
              <span className="text-[10px] font-mono text-[#10B981] bg-[#10B981]/10 px-2 py-0.5 rounded border border-[#10B981]/20">
                Local-First
              </span>
            </div>

            <h3 className="text-sm font-bold text-white mb-1.5 font-sans">
              Local Sentence Transformers & Persistent Vector Engine
            </h3>
            <p className="text-xs text-[#8C93A3] leading-relaxed font-sans">
              All document embeddings are generated locally on CPU using 384-dimensional dense vectors
              and stored in a persistent ChromaDB vector cluster. No proprietary document data is shared with external services during ingestion.
            </p>
          </div>

          <div className="pt-4 mt-4 border-t border-[#1E232D] flex items-center justify-between text-[11px] font-mono text-[#717886]">
            <span>Model: all-MiniLM-L6-v2 (384d)</span>
            <button
              onClick={() => setActiveTab('search')}
              className="text-[#3B82F6] hover:underline flex items-center gap-1 font-sans"
            >
              <span>Test Search Engine</span>
              <ArrowRight className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
