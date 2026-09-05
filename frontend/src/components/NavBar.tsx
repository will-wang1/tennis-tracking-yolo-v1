import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useActiveJobs } from "../hooks/useActiveJobs";
import Logo from "./Logo";

export default function NavBar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const activeJobs = useActiveJobs(isAuthenticated);

  return (
    <nav className="navbar">
      <Link className="brand" to={isAuthenticated ? "/app" : "/"}>
        <Logo size={20} />
        <span className="brand-wordmark">
          Tennis<span className="accent">VA</span>
        </span>
      </Link>
      {isAuthenticated ? (
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {activeJobs.length > 0 && (
            <Link
              to="/app"
              title="Analysis running in the background"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 7,
                padding: "5px 12px",
                borderRadius: "var(--radius-pill)",
                background: "var(--fill-accent-wash)",
                border: "1px solid var(--border-accent-soft)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--text-accent)",
                textDecoration: "none",
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: "var(--tva-accent)",
                  animation: "tvaPulse 2s ease-in-out infinite",
                }}
              />
              {activeJobs.length} analysing
            </Link>
          )}
          <button
            className="btn-ghost"
            onClick={() => {
              logout();
              navigate("/login");
            }}
          >
            Sign out
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", gap: 8 }}>
          <Link className="btn-ghost" to="/login">
            Log in
          </Link>
          <Link className="btn btn-secondary" to="/register">
            Register
          </Link>
        </div>
      )}
    </nav>
  );
}
