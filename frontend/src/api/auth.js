// Authentication endpoints.
import { get, post } from "./client";
import { clearTokens, getRefreshToken, setTokens } from "../auth/tokens";

/** Exchange credentials (email + password) for an access + refresh token pair. */
export async function login(email, password) {
  // auth:false → don't attach a (stale) bearer token or trigger refresh on 401.
  const tokens = await post("/auth/login/", { email, password }, { auth: false });
  setTokens(tokens);
  return tokens;
}

/** Blacklist the refresh token on the server, then clear local storage. */
export async function logout() {
  const refresh = getRefreshToken();
  try {
    if (refresh) await post("/auth/logout/", { refresh });
  } finally {
    clearTokens();
  }
}

/** Fetch the currently authenticated user's profile. */
export const me = () => get("/auth/me/");
