import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { getToken, setToken as persistToken } from "../api/client";

interface AuthContextValue {
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setTokenState] = useState<string | null>(() => getToken());

  const value = useMemo<AuthContextValue>(
    () => ({
      isAuthenticated: token !== null,
      login: (newToken: string) => {
        persistToken(newToken);
        setTokenState(newToken);
      },
      logout: () => {
        persistToken(null);
        setTokenState(null);
      },
    }),
    [token]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
