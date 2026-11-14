/**
 * SIFT — Central API Client Module
 * 
 * Re-exports the client, services, domain types, and UI adapters.
 */

export * from './types';
export * from './client';
export * from './documents';
export * from './search';
export * from './chat';
export * from './health';
export * from './adapters';

import { apiClient } from './client';
import { documentApi } from './documents';
import { searchApi } from './search';
import { chatApi } from './chat';
import { healthApi } from './health';

export const api = {
  client: apiClient,
  documents: documentApi,
  search: searchApi,
  chat: chatApi,
  health: healthApi,
};

export default api;
