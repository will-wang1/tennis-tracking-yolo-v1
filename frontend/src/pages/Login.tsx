import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../context/AuthContext";
import Logo from "../components/Logo";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { access_token } = await api.login(email, password);
      login(access_token);
      navigate("/app");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div style={{ maxWidth: 380, margin: "64px auto 0", padding: "0 20px" }}>
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}>
          <Logo size={40} />
        </div>
        <div className="brand-wordmark" style={{ fontSize: 28 }}>
          Tennis<span className="accent">VA</span>
        </div>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "var(--tracking-eyebrow)",
            textTransform: "uppercase",
            color: "var(--text-muted)",
            marginTop: 6,
          }}
        >
          Match video analysis
        </div>
      </div>

      <div className="card">
        <h3 style={{ fontFamily: "var(--font-sans)", fontWeight: 600, fontSize: 16 }}>
          Sign in to your account
        </h3>
        <form onSubmit={handleSubmit} style={{ marginTop: 16 }}>
          <div className="form-field">
            <label htmlFor="email">Email</label>
            <input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <div className="form-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {error && <p className="error-text">{error}</p>}
          <button className="btn btn-primary" style={{ width: "100%" }} type="submit" disabled={submitting}>
            {submitting ? "Logging in..." : "Log in"}
          </button>
        </form>
        <p style={{ marginTop: 20, textAlign: "center", fontSize: 13, color: "var(--text-muted)" }}>
          Don't have an account? <Link to="/register">Register</Link>
        </p>
      </div>
    </div>
  );
}
