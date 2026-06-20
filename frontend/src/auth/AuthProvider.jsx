import { useCallback, useEffect, useMemo, useState } from "react";

import { AuthContext } from "./context";
import { clearTokens, hasTokens } from "./tokens";
import { login as apiLogin, logout as apiLogout, me as apiMe } from "../api/auth";

export default function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // On first load, validate any stored token by fetching the profile.
  useEffect(() => {
    let active = true;
    (async () => {
      if (hasTokens()) {
        try {
          const profile = await apiMe();
          if (active) setUser(profile);
        } catch {
          clearTokens();
        }
      }
      if (active) setLoading(false);
    })();
    return () => {
      active = false;
    };
  }, []);

  // React to a forced logout broadcast by the API client (refresh failed).
  useEffect(() => {
    const onLogout = () => setUser(null);
    window.addEventListener("auth:logout", onLogout);
    return () => window.removeEventListener("auth:logout", onLogout);
  }, []);

  const login = useCallback(async (email, password) => {
    await apiLogin(email, password);
    setUser(await apiMe());
  }, []);

  const logout = useCallback(async () => {
    await apiLogout();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, isAuthenticated: Boolean(user), loading, login, logout }),
    [user, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
