import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import Logo from "./Logo";

export default function NavBar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <Link className="brand" to="/">
        <Logo size={20} />
        <span className="brand-wordmark">
          Tennis<span className="accent">VA</span>
        </span>
      </Link>
      {isAuthenticated && (
        <button
          className="btn-ghost"
          onClick={() => {
            logout();
            navigate("/login");
          }}
        >
          Sign out
        </button>
      )}
    </nav>
  );
}
