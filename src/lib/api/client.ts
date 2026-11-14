/**
 * SIFT — Centralized Typed API Client
 * 
 * Provides unified request execution, base URL resolution, status-aware error
 * handling, multipart form data support, and AbortSignal cancellation.
 */

import { ApiErrorResponse, ValidationErrorItem } from './types';

export interface RequestOptions extends Omit<RequestInit, 'body'> {
  params?: Record<string, string | number | boolean | undefined | null>;
  timeoutMs?: number;
}

export class ApiError extends Error {
  public readonly statusCode: number;
  public readonly details: Record<string, unknown>;
  public readonly path?: string;
  public readonly isNetworkError: boolean;
  public readonly raw?: unknown;

  constructor(options: {
    message: string;
    statusCode?: number;
    details?: Record<string, unknown>;
    path?: string;
    isNetworkError?: boolean;
    raw?: unknown;
  }) {
    super(options.message);
    this.name = 'ApiError';
    this.statusCode = options.statusCode ?? 500;
    this.details = options.details ?? {};
    this.path = options.path;
    this.isNetworkError = options.isNetworkError ?? false;
    this.raw = options.raw;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Resolves the base API URL from environment variables or defaults to local backend.
 * Safely strips trailing slashes and ensures routing through the '/api' prefix.
 */
export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const raw = envUrl || 'http://127.0.0.1:8000';
  const clean = raw.replace(/\/+$/, '');
  return clean.endsWith('/api') ? clean : `${clean}/api`;
}

/**
 * Builds a query string safely filtering out null and undefined values.
 */
function buildQueryString(params?: Record<string, string | number | boolean | undefined | null>): string {
  if (!params) return '';
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value));
    }
  }
  const str = searchParams.toString();
  return str ? `?${str}` : '';
}

/**
 * Derives a human-readable, safe error message based on response payload and HTTP status.
 */
function formatErrorMessage(status: number, data?: unknown): { message: string; details: Record<string, unknown>; path?: string } {
  const fallbackMessages: Record<number, string> = {
    400: 'Invalid request parameters.',
    404: 'The requested resource was not found.',
    413: 'File size exceeds maximum allowed upload limit (10 MB).',
    422: 'Unable to process the document content or request data.',
    429: 'Too many requests. Please wait a moment before trying again.',
    500: 'An internal backend error occurred.',
    502: 'The AI generation provider encountered an upstream error.',
    503: 'The AI service is temporarily unavailable.',
    504: 'The request timed out while generating a response.',
  };

  if (typeof data === 'object' && data !== null) {
    const errorObj = data as ApiErrorResponse;

    // SIFT custom exception format: { error: true, message: "...", details: {...}, path: "..." }
    if (typeof errorObj.message === 'string' && errorObj.message.trim()) {
      return {
        message: errorObj.message.trim(),
        details: (errorObj.details as Record<string, unknown>) || {},
        path: errorObj.path,
      };
    }

    // FastAPI validation error format: { detail: "..." | [ { loc, msg, type } ] }
    if (typeof errorObj.detail === 'string' && errorObj.detail.trim()) {
      return {
        message: errorObj.detail.trim(),
        details: {},
        path: errorObj.path,
      };
    }

    if (Array.isArray(errorObj.detail) && errorObj.detail.length > 0) {
      const msgs = (errorObj.detail as ValidationErrorItem[])
        .map((item) => item.msg || 'Invalid field')
        .filter(Boolean);
      return {
        message: msgs.join(', '),
        details: { validationErrors: errorObj.detail },
        path: errorObj.path,
      };
    }
  }

  return {
    message: fallbackMessages[status] || `Request failed with status code ${status}.`,
    details: {},
  };
}

export class ApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl ?? getApiBaseUrl();
  }

  /**
   * Primary HTTP request dispatcher.
   */
  public async request<T>(
    endpoint: string,
    options: RequestOptions & { body?: unknown } = {}
  ): Promise<T> {
    const { params, timeoutMs, body, headers: customHeaders, ...fetchInit } = options;
    const cleanEndpoint = endpoint.replace(/^\/+/, '');
    const queryString = buildQueryString(params);
    const fullUrl = `${this.baseUrl}/${cleanEndpoint}${queryString}`;

    const headers = new Headers(customHeaders);
    const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;

    // Set JSON content-type only when NOT uploading multipart FormData
    if (!isFormData) {
      if (!headers.has('Content-Type') && body !== undefined) {
        headers.set('Content-Type', 'application/json');
      }
      if (!headers.has('Accept')) {
        headers.set('Accept', 'application/json');
      }
    }

    // Handle timeout with AbortController
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let effectiveSignal = fetchInit.signal;

    if (timeoutMs && timeoutMs > 0) {
      const controller = new AbortController();
      if (fetchInit.signal) {
        fetchInit.signal.addEventListener('abort', () => controller.abort());
      }
      timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      effectiveSignal = controller.signal;
    }

    try {
      const response = await fetch(fullUrl, {
        ...fetchInit,
        headers,
        body: isFormData
          ? (body as FormData)
          : body !== undefined
          ? JSON.stringify(body)
          : undefined,
        signal: effectiveSignal,
      });

      if (timeoutId) clearTimeout(timeoutId);

      // Handle 204 No Content
      if (response.status === 204) {
        return undefined as unknown as T;
      }

      const contentType = response.headers.get('content-type') || '';
      const isJson = contentType.includes('application/json');
      const rawData = isJson ? await response.json() : await response.text();

      if (!response.ok) {
        const { message, details, path } = formatErrorMessage(response.status, rawData);
        throw new ApiError({
          message,
          statusCode: response.status,
          details,
          path: path || cleanEndpoint,
          raw: rawData,
        });
      }

      return rawData as T;
    } catch (err: unknown) {
      if (timeoutId) clearTimeout(timeoutId);

      if (err instanceof ApiError) {
        throw err;
      }

      const errorName = (err as { name?: string })?.name;
      if (errorName === 'AbortError') {
        throw new ApiError({
          message: 'The request timed out or was cancelled by the client.',
          statusCode: 408,
          isNetworkError: true,
          raw: err,
        });
      }

      // Network unreachable or CORS failure
      throw new ApiError({
        message:
          'Unable to communicate with the SIFT backend server. Please verify the service is running at ' +
          this.baseUrl,
        statusCode: 0,
        isNetworkError: true,
        raw: err,
      });
    }
  }

  public get<T>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  public post<T>(endpoint: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'POST', body });
  }

  public patch<T>(endpoint: string, body?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'PATCH', body });
  }

  public delete<T = void>(endpoint: string, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  public upload<T>(endpoint: string, formData: FormData, options?: RequestOptions): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'POST', body: formData });
  }
}

// Singleton default client instance
export const apiClient = new ApiClient();
