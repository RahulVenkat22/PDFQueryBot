import { useCallback, useEffect, useState } from "react";

import "./UserPage.css";
import UserForm from "./UserForm";
import UserList from "./UserList";
import { Banner, Button, ConfirmDialog, Input } from "../../components/common";
import { createUser, deleteUser, listUsers, updateUser } from "../../api/users";
import { useAuth } from "../../auth/useAuth";

export default function UserPage() {
  const { user, logout } = useAuth();
  const [users, setUsers] = useState([]);
  const [search, setSearch] = useState("");
  const [editing, setEditing] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [formErrors, setFormErrors] = useState({});
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async (term = "") => {
    setLoading(true);
    try {
      setUsers(await listUsers(term));
    } catch {
      setMessage({ type: "error", text: "Could not load users. Is the backend running?" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Load the initial list once on mount (legitimate data-fetching effect).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh();
  }, [refresh]);

  // Normalise a DRF error map ({field: [msg]}) into {field: msg}.
  const flattenErrors = (fields = {}) =>
    Object.fromEntries(
      Object.entries(fields).map(([key, val]) => [key, Array.isArray(val) ? val.join(" ") : String(val)]),
    );

  const handleSubmit = async (values) => {
    setBusy(true);
    setFormErrors({});
    try {
      if (editing) {
        await updateUser(editing.id, values);
        setMessage({ type: "success", text: "User updated." });
        setEditing(null);
      } else {
        await createUser(values);
        setMessage({ type: "success", text: "User created." });
      }
      await refresh(search);
    } catch (err) {
      setFormErrors(flattenErrors(err.fields));
      setMessage({ type: "error", text: "Please fix the errors and try again." });
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setBusy(true);
    try {
      await deleteUser(pendingDelete.id);
      if (editing?.id === pendingDelete.id) setEditing(null);
      setMessage({ type: "success", text: "User deleted." });
      await refresh(search);
    } catch {
      setMessage({ type: "error", text: "Could not delete user." });
    } finally {
      setBusy(false);
      setPendingDelete(null);
    }
  };

  const handleSearch = (event) => {
    event.preventDefault();
    refresh(search);
  };

  return (
    <main className="user-page">
      <header className="page-header">
        <div>
          <h1>PDFQueryBot — Users</h1>
          <p>Manage application users (create, read, update, delete).</p>
        </div>
        <div className="user-menu">
          {user && <span className="who">{user.username}</span>}
          <Button variant="secondary" onClick={logout}>
            Logout
          </Button>
        </div>
      </header>

      <Banner type={message?.type} onDismiss={() => setMessage(null)}>
        {message?.text}
      </Banner>

      <section className="panel">
        <UserForm
          key={editing ? editing.id : "new"}
          initial={editing}
          errors={formErrors}
          busy={busy}
          onSubmit={handleSubmit}
          onCancel={() => {
            setEditing(null);
            setFormErrors({});
          }}
        />
      </section>

      <section className="panel">
        <form className="toolbar" onSubmit={handleSearch}>
          <Input
            placeholder="Search by name, email or phone…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <Button type="submit" disabled={loading}>
            Search
          </Button>
        </form>

        {loading ? (
          <p className="table-empty">Loading…</p>
        ) : (
          <UserList
            users={users}
            busy={busy}
            onEdit={(user) => {
              setEditing(user);
              setFormErrors({});
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            onDelete={(user) => setPendingDelete(user)}
          />
        )}
      </section>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete user"
        message={pendingDelete && `Delete ${pendingDelete.full_name}? This cannot be undone.`}
        confirmLabel="Delete"
        busy={busy}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </main>
  );
}
