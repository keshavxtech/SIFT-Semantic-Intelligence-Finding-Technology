'use client';

import React from 'react';
import {
  Search,
  Upload,
  Command,
  ChevronDown,
  LayoutGrid,
  Menu,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';

interface TopHeaderProps {
  onToggleMobileNav?: () => void;
}

export const TopHeader: React.FC<TopHeaderProps> = ({ onToggleMobileNav }) => {
  const {
    setIsUploadModalOpen,
    setIsCommandPaletteOpen,
  } = useDocumentContext();

  return (
    <header className="flex h-16 w-full items-center justify-between border-b border-[#1E232D] bg-[#0E1015] px-3 sm:px-6 shrink-0 gap-2">
      {/* Left: Workspace Dropdown */}
      <div className="flex items-center gap-2 sm:gap-3 min-w-0">
        {onToggleMobileNav && (
          <button
            onClick={onToggleMobileNav}
            aria-label="Toggle navigation menu"
            className="md:hidden text-[#8C93A3] hover:text-[#E8EAF0] p-1.5 rounded-lg hover:bg-[#161A22] transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>
        )}

        <button
          aria-label="Workspace selector"
          className="flex items-center gap-2 rounded-lg bg-[#14171E] hover:bg-[#1A1F29] border border-[#232834] px-2.5 sm:px-3 py-1.5 text-xs font-medium text-[#E8EAF0] transition-colors shrink-0"
        >
          <LayoutGrid className="h-3.5 w-3.5 text-[#8C93A3]" aria-hidden="true" />
          <span className="truncate max-w-[110px] sm:max-w-none">Default Workspace</span>
          <ChevronDown className="h-3.5 w-3.5 text-[#717886] ml-0.5 sm:ml-1 shrink-0" aria-hidden="true" />
        </button>
      </div>

      {/* Right: Search Pill & Upload Button */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Search Input Bar with ⌘K */}
        <button
          onClick={() => setIsCommandPaletteOpen(true)}
          aria-label="Open command palette (Ctrl+K)"
          className="flex items-center gap-2 rounded-lg bg-[#14171E] hover:bg-[#181D26] border border-[#232834] px-2.5 sm:px-3.5 py-1.5 text-xs text-[#717886] hover:text-[#8C93A3] transition-colors w-auto max-w-[150px] sm:max-w-xs sm:w-72 lg:w-80 justify-between shrink min-w-0"
        >
          <div className="flex items-center gap-2 min-w-0">
            <Search className="h-3.5 w-3.5 text-[#717886] shrink-0" aria-hidden="true" />
            <span className="truncate hidden sm:inline">Search documents, ask questions...</span>
            <span className="truncate sm:hidden text-xs">Search...</span>
          </div>
          <kbd className="hidden sm:inline-flex items-center gap-0.5 rounded bg-[#1C212C] px-1.5 py-0.5 text-[10px] font-mono text-[#8C93A3] border border-[#2A3140] shrink-0">
            <Command className="h-2.5 w-2.5" /> K
          </kbd>
        </button>

        {/* Upload Document Primary Button */}
        <button
          onClick={() => setIsUploadModalOpen(true)}
          aria-label="Upload Document"
          className="flex items-center gap-1.5 sm:gap-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] px-3 sm:px-3.5 py-1.5 text-xs font-medium text-white shadow-sm transition-colors shrink-0"
        >
          <Upload className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="hidden sm:inline">Upload Document</span>
          <span className="sm:hidden text-xs">Upload</span>
        </button>
      </div>
    </header>
  );
};
