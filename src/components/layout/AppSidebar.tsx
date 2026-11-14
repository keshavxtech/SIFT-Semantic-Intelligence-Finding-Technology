'use client';

import React from 'react';
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  Search,
  Settings,
  Puzzle,
  Info,
  ChevronRight,
  PanelLeftClose,
  PanelLeft,
  X,
} from 'lucide-react';
import { useDocumentContext } from '@/context/DocumentContext';
import { ActiveTab } from '@/types';

interface AppSidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
  onCloseMobile?: () => void;
  isMobileDrawer?: boolean;
}

export const AppSidebar: React.FC<AppSidebarProps> = ({
  collapsed,
  onToggleCollapse,
  onCloseMobile,
  isMobileDrawer = false,
}) => {
  const { activeTab, setActiveTab, documents } = useDocumentContext();

  const navItems: { id: ActiveTab; label: string; icon: React.ElementType; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'library', label: 'Document Library', icon: FileText, badge: `${documents.length}` },
    { id: 'chat', label: 'Document Chat', icon: MessageSquare },
    { id: 'search', label: 'Search & Retrieval', icon: Search },
  ];

  const settingsItems = [
    { label: 'Workspace Settings', icon: Settings },
    { label: 'Integrations', icon: Puzzle },
    { label: 'About', icon: Info },
  ];

  const handleNavClick = (tabId: ActiveTab) => {
    setActiveTab(tabId);
    if (onCloseMobile) {
      onCloseMobile();
    }
  };

  return (
    <aside
      aria-label="Sidebar navigation"
      className={`relative flex flex-col border-r border-[#1E232D] bg-[#0E1015] transition-all duration-200 select-none ${
        collapsed && !isMobileDrawer ? 'w-16' : 'w-64'
      } shrink-0 h-full`}
    >
      {/* Brand Header */}
      <div className="p-4 pb-2 border-b border-[#1E232D]/40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            {/* Custom SIFT Diamond Rhombus Logo */}
            <div className="flex h-7 w-7 items-center justify-center shrink-0">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="text-[#3B82F6]"
                aria-hidden="true"
              >
                <path
                  d="M12 2L2 12L12 22L22 12L12 2Z"
                  stroke="#3B82F6"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="fill-[#3B82F6]/15"
                />
                <path
                  d="M12 6L7 12L12 18L17 12L12 6Z"
                  fill="#3B82F6"
                  opacity="0.9"
                />
              </svg>
            </div>
            {(!collapsed || isMobileDrawer) && (
              <span className="text-lg font-bold tracking-tight text-white font-sans">
                SIFT
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            {isMobileDrawer ? (
              <button
                onClick={onCloseMobile}
                className="text-[#808898] hover:text-white p-1 rounded-lg hover:bg-[#191D25] transition-colors"
                aria-label="Close sidebar navigation"
              >
                <X className="h-4 w-4" />
              </button>
            ) : (
              <button
                onClick={onToggleCollapse}
                className="text-[#808898] hover:text-white p-1 rounded-lg hover:bg-[#191D25] transition-colors"
                aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
              </button>
            )}
          </div>
        </div>

        {(!collapsed || isMobileDrawer) && (
          <p className="mt-2 text-[11px] text-[#717886] leading-tight font-sans">
            Find what matters.<br />
            Understand everything.
          </p>
        )}
      </div>

      {/* Main Navigation Items */}
      <nav aria-label="Main Navigation" className="flex-1 px-3 py-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              aria-current={isActive ? 'page' : undefined}
              className={`flex items-center gap-3 w-full rounded-lg text-xs font-medium transition-all ${
                collapsed && !isMobileDrawer ? 'justify-center h-10 px-0' : 'h-9 px-3'
              } ${
                isActive
                  ? 'bg-[#151D2C] text-white border border-[#23354E] shadow-xs'
                  : 'text-[#8C93A3] hover:text-[#E8EAF0] hover:bg-[#14171D]'
              }`}
              title={collapsed && !isMobileDrawer ? item.label : undefined}
            >
              <Icon
                className={`h-4 w-4 shrink-0 ${
                  isActive ? 'text-[#3B82F6]' : 'text-[#8C93A3]'
                }`}
                aria-hidden="true"
              />
              {(!collapsed || isMobileDrawer) && (
                <span className="flex-1 text-left truncate">{item.label}</span>
              )}
              {(!collapsed || isMobileDrawer) && item.badge && (
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-[#181C24] text-[#8C93A3] border border-[#232834]">
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}

        {/* SETTINGS SECTION */}
        {(!collapsed || isMobileDrawer) && (
          <div className="pt-6 pb-1">
            <div className="px-3 pb-2 text-[10px] font-mono uppercase tracking-wider text-[#525A6A]">
              SETTINGS
            </div>
            {settingsItems.map((s, idx) => {
              const SIcon = s.icon;
              return (
                <button
                  key={idx}
                  className="flex items-center gap-3 w-full h-8 px-3 rounded-lg text-xs font-normal text-[#8C93A3] hover:text-[#E8EAF0] hover:bg-[#14171D] transition-colors"
                >
                  <SIcon className="h-4 w-4 shrink-0 text-[#717886]" aria-hidden="true" />
                  <span className="flex-1 text-left truncate">{s.label}</span>
                </button>
              );
            })}
          </div>
        )}
      </nav>

      {/* FOOTER: System Healthy & User Card */}
      <div className="p-3 border-t border-[#1E232D] space-y-2">
        {/* System Healthy Card with Heartbeat Wave */}
        <div className="rounded-lg bg-[#12151B] p-2.5 border border-[#1E232D] flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className="h-2 w-2 rounded-full bg-[#10B981] shrink-0" aria-hidden="true" />
            {(!collapsed || isMobileDrawer) && (
              <div className="min-w-0 flex flex-col">
                <span className="text-xs font-medium text-[#E8EAF0] truncate">
                  System Healthy
                </span>
                <span className="text-[10px] text-[#717886] truncate">
                  All services operational
                </span>
              </div>
            )}
          </div>

          {(!collapsed || isMobileDrawer) && (
            <svg
              width="34"
              height="16"
              viewBox="0 0 34 16"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="text-[#10B981] shrink-0"
              aria-hidden="true"
            >
              <path
                d="M1 8H8L11 2L15 14L19 5L22 10L25 8H33"
                stroke="#10B981"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>

        {/* User Profile Card */}
        <div
          tabIndex={0}
          role="button"
          aria-label="User profile Keshav Kumar, Workspace Owner"
          className="rounded-lg bg-[#12151B] p-2 border border-[#1E232D] flex items-center justify-between hover:bg-[#161A22] transition-colors cursor-pointer group"
        >
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="h-7 w-7 rounded-full bg-[#2563EB] flex items-center justify-center text-xs font-medium text-white shrink-0">
              K
            </div>
            {(!collapsed || isMobileDrawer) && (
              <div className="min-w-0 flex flex-col">
                <span className="text-xs font-medium text-[#E8EAF0] truncate">
                  Keshav Kumar
                </span>
                <span className="text-[10px] text-[#717886] truncate">
                  Workspace Owner
                </span>
              </div>
            )}
          </div>
          {(!collapsed || isMobileDrawer) && (
            <ChevronRight className="h-3.5 w-3.5 text-[#717886] group-hover:text-white transition-colors" />
          )}
        </div>
      </div>
    </aside>
  );
};
