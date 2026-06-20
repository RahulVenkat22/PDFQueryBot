import { createContext } from "react";

// Holds { user, isAuthenticated, loading, login, logout }.
export const AuthContext = createContext(null);
