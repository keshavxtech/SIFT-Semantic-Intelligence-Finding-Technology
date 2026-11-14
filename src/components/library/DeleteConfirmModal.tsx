'use client';

import React, { useState } from 'react';
import { X, Trash2, AlertTriangle, Loader2 } from 'lucide-react';
import { DocumentItem } from '@/types';

interface DeleteConfirmModalProps {
  doc: DocumentItem | null;
  onClose: () => void;
  onConfirm: (id: string) => Promise<void> | void;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
  doc,
  onClose,
  onConfirm,
}) => {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && doc && !isDeleting) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [doc, isDeleting, onClose]);

  if (!doc) return null;

  const handleDelete = async () => {
    try {
      setIsDeleting(true);
      setError(null);
      await onConfirm(doc.id);
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete document';
      setError(msg);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in-0 duration-150"
    >
      <div
        className="w-full max-w-md rounded-xl bg-[#14171E] border border-[#1E232D] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-[#1E232D] px-5 py-4 bg-[#14171E]">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
              <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            </div>
            <div>
              <h3 id="delete-modal-title" className="text-sm font-semibold text-white">Delete Document</h3>
              <p className="text-[11px] text-[#717886]">Irreversible corpus action</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isDeleting}
            aria-label="Close delete confirmation"
            className="text-[#717886] hover:text-white p-1 rounded-lg hover:bg-[#1E232D] disabled:opacity-50"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-5 space-y-3">
          <p className="text-xs text-[#8C93A3] leading-relaxed">
            Are you sure you want to permanently delete{' '}
            <strong className="text-white font-medium">{doc.name}</strong> from SIFT?
          </p>

          <div className="bg-[#0E1015] border border-[#1E232D] p-3 rounded-lg text-xs space-y-1 text-[#717886]">
            <div className="flex justify-between">
              <span>Associated Chunks:</span>
              <span className="font-mono text-white">{doc.chunksCount} vectors</span>
            </div>
            <div className="flex justify-between">
              <span>Token Footprint:</span>
              <span className="font-mono text-white">{doc.tokenCount.toLocaleString()} tokens</span>
            </div>
            <div className="flex justify-between">
              <span>Storage Size:</span>
              <span className="font-mono text-white">{doc.size}</span>
            </div>
          </div>

          <p className="text-[11px] text-[#717886]">
            All neural indexes and cached retrieval spans associated with this document will be purged from the active vector cluster.
          </p>

          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 p-2 rounded-lg">
              {error}
            </p>
          )}
        </div>

        <div className="border-t border-[#1E232D] px-5 py-3.5 bg-[#14171E] flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            disabled={isDeleting}
            className="px-4 py-2 rounded-lg text-xs text-[#8C93A3] hover:text-white hover:bg-[#1E232D] transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-xs font-medium text-white transition-colors shadow-sm disabled:opacity-50"
          >
            {isDeleting ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Deleting...</span>
              </>
            ) : (
              <>
                <Trash2 className="h-3.5 w-3.5" />
                <span>Delete Document</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
