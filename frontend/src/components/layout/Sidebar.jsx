import { NavLink } from "react-router-dom";

import { useAuth } from "../../auth/useAuth";

// Icons kept inline (no extra dependency). currentColor lets CSS drive the colour.
function UsersIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

// Navigation items — add more here as new sections are built.
const NAV_ITEMS = [{ to: "/", label: "Users", icon: <UsersIcon /> }];

export default function Sidebar({ open, onToggle }) {
  const { user, logout } = useAuth();

  return (
    <aside className={`sidebar ${open ? "open" : "closed"}`}>
      <div className="sidebar-top">
        {open && <span className="sidebar-brand">PDFQueryBot</span>}
        <button className="sidebar-toggle" onClick={onToggle} aria-label="Toggle sidebar">
          <MenuIcon />
        </button>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink key={item.to} to={item.to} end className="sidebar-link" title={item.label}>
            <span className="sidebar-icon">{item.icon}</span>
            {open && <span className="sidebar-label">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        {open && user && <span className="sidebar-user">{user.email || user.username}</span>}
        <button className="sidebar-link logout" onClick={logout} title="Logout">
          <span className="sidebar-icon">
            <LogoutIcon />
          </span>
          {open && <span className="sidebar-label">Logout</span>}
        </button>
      </div>
    </aside>
  );
}
