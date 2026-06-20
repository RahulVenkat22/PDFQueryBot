import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./useAuth";

/** Guards nested routes — redirects to /login when not authenticated. */
export default function ProtectedRoute() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) return <div className="auth-loading">Loading…</div>;
  if (!isAuthenticated) return <Navigate to="/login" replace />;

  return <Outlet />;
}
