'use client';

import React, { useState } from 'react';
import { DocumentProvider, useDocumentContext } from '@/context/DocumentContext';
import { AppSidebar } from '@/components/layout/AppSidebar';
import { TopHeader } from '@/components/layout/TopHeader';
import { DashboardView } from '@/components/dashboard/DashboardView';
import { DocumentLibraryView } from '@/components/library/DocumentLibraryView';
import { DocumentChatView } from '@/components/chat/DocumentChatView';
import { SearchRetrievalView } from '@/components/search/SearchRetrievalView';
import { UploadModal } from '@/components/common/UploadModal';
import { CommandPalette } from '@/components/common/CommandPalette';

function AppContent() {
  const { activeTab } = useDocumentContext();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  // Close mobile nav on escape
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileNavOpen) {
        setMobileNavOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileNavOpen]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#101216] text-[#E8EAF0]">
      {/* Sidebar - Desktop & Tablet */}
      <div
        className={`${
          mobileNavOpen
            ? 'fixed inset-y-0 left-0 z-40 flex shadow-2xl animate-in slide-in-from-left-2 duration-150'
            : 'hidden md:flex'
        }`}
      >
        <AppSidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
          onCloseMobile={() => setMobileNavOpen(false)}
          isMobileDrawer={mobileNavOpen}
        />
      </div>

      {/* Mobile Backdrop */}
      {mobileNavOpen && (
        <div
          onClick={() => setMobileNavOpen(false)}
          className="fixed inset-0 z-30 bg-black/70 backdrop-blur-xs md:hidden"
          aria-hidden="true"
        />
      )}

      {/* Main Workspace Frame */}
      <div className="flex flex-1 flex-col h-full min-w-0 overflow-hidden">
        {/* Top Header */}
        <TopHeader onToggleMobileNav={() => setMobileNavOpen(!mobileNavOpen)} />

        {/* Dynamic View Router */}
        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 flex flex-col h-[calc(100vh-4rem)] min-w-0 overflow-hidden focus:outline-hidden"
        >
          {activeTab === 'dashboard' && <DashboardView />}
          {activeTab === 'library' && <DocumentLibraryView />}
          {activeTab === 'chat' && <DocumentChatView />}
          {activeTab === 'search' && <SearchRetrievalView />}
        </main>
      </div>

      {/* Global Overlays */}
      <UploadModal />
      <CommandPalette />
    </div>
  );
}

export default function Home() {
  return (
    <DocumentProvider>
      <AppContent />
    </DocumentProvider>
  );
}
