// Generic HTTP client shared by all API modules.
// Centralises the base URL, JSON handling, auth headers and token refresh.
import {
  clearTokens,
  emitLogout,
  getAccessToken,
  getRefreshToken,
  setAccessToken,
} from "../auth/tokens";

const BASE_URL = import.meta.env.VITE_API_BASE_URL;

/**
 * Parse a fetch Response, raising an Error carrying the DRF field errors
 * (so forms can surface per-field validation messages).
 */
async function handle(response) {
  if (response.status === 204) return null;

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error("Request failed");
    error.status = response.status;
    error.fields = data && typeof data === "object" ? data : {};
    throw error;
  }
  return data;
}

function buildUrl(path, params) {
  let url = `${BASE_URL}${path}`;
  if (params) {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value !== "" && value != null),
    ).toString();
    if (query) url += `?${query}`;
  }
  return url;
}

function buildHeaders(body, auth) {
  const headers = {};
  if (body) headers["Content-Type"] = "application/json";
  if (auth) {
    const token = getAccessToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

// --- Token refresh (single in-flight request shared by concurrent callers) ---
let refreshPromise = null;

function refreshAccessToken() {
  if (refreshPromise) return refreshPromise;

  const refresh = getRefreshToken();
  if (!refresh) return Promise.resolve(false);

  refreshPromise = fetch(buildUrl("/auth/refresh/"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh }),
  })
    .then(async (res) => {
      if (!res.ok) return false;
      const data = await res.json();
      setAccessToken(data.access);
      return true;
    })
    .catch(() => false)
    .finally(() => {
      refreshPromise = null;
    });

  return refreshPromise;
}

/**
 * Core request helper.
 * @param {string} path     Relative to BASE_URL (e.g. "/users/").
 * @param {object} options  { method, params, body, auth }
 *   `auth` (default true) attaches the bearer token and enables 401-refresh.
 */
async function request(path, { method = "GET", params, body, auth = true } = {}) {
  const url = buildUrl(path, params);
  const init = {
    method,
    headers: buildHeaders(body, auth),
    body: body ? JSON.stringify(body) : undefined,
  };

  let response = await fetch(url, init);

  // Access token expired → try a one-time refresh, then replay the request.
  if (response.status === 401 && auth) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      init.headers = buildHeaders(body, auth);
      response = await fetch(url, init);
    } else {
      clearTokens();
      emitLogout();
    }
  }

  return handle(response);
}

export const get = (path, params) => request(path, { method: "GET", params });
export const post = (path, body, options) => request(path, { method: "POST", body, ...options });
export const put = (path, body) => request(path, { method: "PUT", body });
export const patch = (path, body) => request(path, { method: "PATCH", body });
export const del = (path) => request(path, { method: "DELETE" });
