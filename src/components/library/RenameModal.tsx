'use client';

import React, { useState } from 'react';
import { X, Edit2, AlertCircle } from 'lucide-react';
import { DocumentItem } from '@/types';

interface RenameModalProps {
  doc: DocumentItem | null;
  onClose: () => void;
  onRename: (id: string, newName: string) => Promise<void> | void;
}

export const RenameModal: React.FC<RenameModalProps> = ({
  doc,
  onClose,
  onRename,
}) => {
  const [newName, setNewName] = useState(doc?.name || '');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && doc && !isSubmitting) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [doc, isSubmitting, onClose]);

  if (!doc) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) {
      setError('Document name cannot be empty');
      return;
    }
    if (newName.trim() === doc.name) {
      onClose();
      return;
    }

    try {
      setIsSubmitting(true);
      await onRename(doc.id, newName.trim());
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to rename document';
      setError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="rename-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in-0 duration-150"
    >
      <div
        className="w-full max-w-md rounded-xl bg-[#14171E] border border-[#1E232D] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-[#1E232D] px-5 py-4 bg-[#14171E]">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#182338] border border-[#23354E] text-[#3B82F6]">
              <Edit2 className="h-4 w-4" aria-hidden="true" />
            </div>
            <div>
              <h3 id="rename-modal-title" className="text-sm font-semibold text-white">Rename Document</h3>
              <p className="text-[11px] text-[#717886]">Update the title in the knowledge index</p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close rename dialog"
            className="text-[#717886] hover:text-white p-1 rounded-lg hover:bg-[#1E232D]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#8C93A3] mb-1.5 font-sans">
              Document Title
            </label>
            <input
              type="text"
              value={newName}
              onChange={(e) => {
                setNewName(e.target.value);
                if (error) setError('');
              }}
              autoFocus
              className="w-full rounded-lg bg-[#0E1015] border border-[#232834] px-3.5 py-2 text-xs text-white placeholder-[#525A6A] outline-none focus:border-[#3B82F6] transition-colors"
            />
            {error && (
              <p className="text-[11px] text-red-400 mt-1.5 flex items-center gap-1">
                <AlertCircle className="h-3 w-3" />
                <span>{error}</span>
              </p>
            )}
          </div>

          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 rounded-lg text-xs text-[#8C93A3] hover:text-white hover:bg-[#1E232D] transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-4 py-2 rounded-lg bg-[#2563EB] hover:bg-[#1D4ED8] text-xs font-medium text-white transition-colors disabled:opacity-50"
            >
              {isSubmitting ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
