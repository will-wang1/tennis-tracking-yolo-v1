import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function NavBar() {
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <nav className="navbar">
      <Link className="brand" to="/">
        🎾 Tennis Video Analysis
      </Link>
      {isAuthenticated && (
        <button
          onClick={() => {
            logout();
            navigate("/login");
          }}
        >
          Log out
        </button>
      )}
    </nav>
  );
}
