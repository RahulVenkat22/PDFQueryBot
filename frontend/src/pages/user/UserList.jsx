import { Button, Table } from "../../components/common";

/** Table of users with edit / delete actions, built on the generic Table. */
export default function UserList({ users, onEdit, onDelete, busy }) {
  const columns = [
    { key: "id", header: "ID" },
    { key: "name", header: "Name", render: (u) => u.full_name },
    { key: "mail_id", header: "Email" },
    { key: "phone_number", header: "Phone" },
    { key: "gender", header: "Gender", className: "capitalize" },
    {
      key: "actions",
      header: "",
      className: "row-actions",
      render: (u) => (
        <>
          <Button variant="link" onClick={() => onEdit(u)} disabled={busy}>
            Edit
          </Button>
          <Button variant="link" className="danger" onClick={() => onDelete(u)} disabled={busy}>
            Delete
          </Button>
        </>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      data={users}
      empty="No users yet. Add one using the form above."
    />
  );
}
