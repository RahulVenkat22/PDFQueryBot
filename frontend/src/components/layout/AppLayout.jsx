import { useState } from "react";
import { Outlet } from "react-router-dom";

import "./layout.css";
import Sidebar from "./Sidebar";

const STORAGE_KEY = "pdfqb.sidebar";

/** App shell: collapsible sidebar + routed page content. */
export default function AppLayout() {
  const [open, setOpen] = useState(() => localStorage.getItem(STORAGE_KEY) !== "closed");

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev;
      localStorage.setItem(STORAGE_KEY, next ? "open" : "closed");
      return next;
    });
  };

  return (
    <div className={`app-layout ${open ? "sidebar-open" : "sidebar-collapsed"}`}>
      <Sidebar open={open} onToggle={toggle} />
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  );
}
