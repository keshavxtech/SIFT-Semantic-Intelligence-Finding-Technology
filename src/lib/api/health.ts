/**
 * SIFT — System Health & Status API Service
 * 
 * Provides methods for inspecting backend service health, storage utilization,
 * and deployment environment status.
 */

import { ApiClient, apiClient, RequestOptions } from './client';
import { HealthResponse, RootStatusResponse } from './types';

export class HealthApiService {
  private client: ApiClient;

  constructor(client: ApiClient = apiClient) {
    this.client = client;
  }

  /**
   * Retrieves operational health, document index count, and disk storage metrics.
   */
  public checkHealth(options?: RequestOptions): Promise<HealthResponse> {
    return this.client.get<HealthResponse>('health', options);
  }

  /**
   * Retrieves root API status and service identification.
   */
  public getRootStatus(options?: RequestOptions): Promise<RootStatusResponse> {
    // Calls the root / endpoint on the backend
    const rootClient = new ApiClient(
      this.client['baseUrl'].replace(/\/api\/?$/, '')
    );
    return rootClient.get<RootStatusResponse>('', options);
  }
}

export const healthApi = new HealthApiService();
