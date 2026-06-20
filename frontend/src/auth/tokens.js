// Centralised access/refresh token storage (localStorage).
const ACCESS_KEY = "pdfqb.access";
const REFRESH_KEY = "pdfqb.refresh";

export const getAccessToken = () => localStorage.getItem(ACCESS_KEY);
export const getRefreshToken = () => localStorage.getItem(REFRESH_KEY);

export function setTokens({ access, refresh }) {
  if (access) localStorage.setItem(ACCESS_KEY, access);
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
}

export function setAccessToken(access) {
  if (access) localStorage.setItem(ACCESS_KEY, access);
}

export function clearTokens() {
  localStorage.removeItem(ACCESS_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

export const hasTokens = () => Boolean(getAccessToken());

// Broadcast a forced logout (e.g. refresh failed) so the AuthContext can react.
export function emitLogout() {
  window.dispatchEvent(new Event("auth:logout"));
}
