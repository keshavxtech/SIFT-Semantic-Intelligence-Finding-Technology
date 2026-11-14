'use client';

import React, { useState } from 'react';
import {
  Search,
  MessageSquare,
  Files,
  LayoutDashboard,
  Upload,
  ArrowRight,
  X,
  FileText,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';

export const CommandPalette: React.FC = () => {
  const {
    isCommandPaletteOpen,
    setIsCommandPaletteOpen,
    setActiveTab,
    documents,
    quickAskDocument,
    setIsUploadModalOpen,
    performSearch,
  } = useDocumentContext();

  const [input, setInput] = useState('');

  const handleClose = () => {
    setInput('');
    setIsCommandPaletteOpen(false);
  };

  if (!isCommandPaletteOpen) return null;

  const filteredDocs = documents.filter((d) =>
    d.name.toLowerCase().includes(input.toLowerCase()) ||
    d.tag.toLowerCase().includes(input.toLowerCase())
  );

  return (
    <div
      onClick={handleClose}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-sm animate-in fade-in-0 duration-150"
    >
      <div
        className="w-full max-w-xl rounded-lg bg-[#14171D] border border-[#292E38] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 border-b border-[#292E38] px-4 py-3 bg-[#191D25]">
          <Search className="h-4 w-4 text-[#9299A7]" aria-hidden="true" />
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Search documents, commands, or ask questions..."
            aria-label="Search documents or commands"
            autoFocus
            className="flex-1 bg-transparent text-sm text-[#E8EAF0] placeholder-[#656C7D] outline-none"
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                handleClose();
              }
              if (e.key === 'Enter' && input.trim()) {
                performSearch(input.trim());
                setActiveTab('search');
                handleClose();
              }
            }}
          />
          <button
            onClick={handleClose}
            aria-label="Close command palette"
            className="text-[#9299A7] hover:text-[#E8EAF0] p-1 rounded-lg hover:bg-[#1E232D]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Command Groups */}
        <div className="max-h-96 overflow-y-auto p-2 space-y-3">
          {/* Direct Actions */}
          <div>
            <div className="px-2 py-1 text-[10px] font-mono uppercase tracking-wider text-[#656C7D]">
              Navigation & Actions
            </div>
            <div className="space-y-0.5 mt-1">
              <button
                onClick={() => {
                  setActiveTab('chat');
                  setIsCommandPaletteOpen(false);
                }}
                className="flex items-center justify-between w-full px-2.5 py-2 rounded-md hover:bg-[#1E232D] text-left text-xs text-[#E8EAF0] group"
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-3.5 w-3.5 text-[#3B82F6]" />
                  <span>Open Document Chat Workspace</span>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#656C7D] opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>

              <button
                onClick={() => {
                  setActiveTab('library');
                  setIsCommandPaletteOpen(false);
                }}
                className="flex items-center justify-between w-full px-2.5 py-2 rounded-md hover:bg-[#1E232D] text-left text-xs text-[#E8EAF0] group"
              >
                <div className="flex items-center gap-2">
                  <Files className="h-3.5 w-3.5 text-[#10B981]" />
                  <span>Browse Document Library</span>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#656C7D] opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>

              <button
                onClick={() => {
                  setActiveTab('dashboard');
                  setIsCommandPaletteOpen(false);
                }}
                className="flex items-center justify-between w-full px-2.5 py-2 rounded-md hover:bg-[#1E232D] text-left text-xs text-[#E8EAF0] group"
              >
                <div className="flex items-center gap-2">
                  <LayoutDashboard className="h-3.5 w-3.5 text-[#9299A7]" />
                  <span>Workspace Telemetry & Activity</span>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#656C7D] opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>

              <button
                onClick={() => {
                  setIsCommandPaletteOpen(false);
                  setIsUploadModalOpen(true);
                }}
                className="flex items-center justify-between w-full px-2.5 py-2 rounded-md hover:bg-[#1E232D] text-left text-xs text-[#E8EAF0] group"
              >
                <div className="flex items-center gap-2">
                  <Upload className="h-3.5 w-3.5 text-[#F59E0B]" />
                  <span>Ingest New Document (PDF, DOCX, MD)</span>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-[#656C7D] opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
            </div>
          </div>

          {/* Scoped Documents */}
          <div>
            <div className="px-2 py-1 text-[10px] font-mono uppercase tracking-wider text-[#656C7D]">
              Indexed Documents
            </div>
            <div className="space-y-0.5 mt-1">
              {filteredDocs.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => {
                    quickAskDocument(doc);
                    setIsCommandPaletteOpen(false);
                  }}
                  className="flex items-center justify-between w-full px-2.5 py-2 rounded-md hover:bg-[#1E232D] text-left text-xs text-[#E8EAF0] group"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="h-3.5 w-3.5 text-[#9299A7] shrink-0" />
                    <span className="truncate">{doc.name}</span>
                    <span className="text-[10px] font-mono text-[#656C7D] shrink-0">
                      [{doc.tag}]
                    </span>
                  </div>
                  <span className="text-[10px] text-[#3B82F6] opacity-0 group-hover:opacity-100 transition-opacity">
                    Ask Document
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer shortcuts */}
        <div className="border-t border-[#292E38] px-4 py-2 bg-[#101216] flex items-center justify-between text-[11px] text-[#656C7D] font-mono">
          <span>Navigation: ↑ ↓ · Selection: Enter</span>
          <span>Close: Esc</span>
        </div>
      </div>
    </div>
  );
};
