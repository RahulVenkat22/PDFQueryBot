import { useState } from "react";

import { Button, Input, Select } from "../../components/common";

const EMPTY = {
  first_name: "",
  last_name: "",
  mail_id: "",
  phone_number: "",
  gender: "male",
};

const GENDERS = [
  { value: "male", label: "Male" },
  { value: "female", label: "Female" },
  { value: "other", label: "Other" },
];

/**
 * Create / edit form. When `initial` is provided the form is in "edit" mode,
 * otherwise it creates a new user. `errors` is the DRF field-error map.
 */
export default function UserForm({ initial, errors = {}, onSubmit, onCancel, busy }) {
  // Initialised once per mount; App passes a `key` so the form remounts
  // (and resets) whenever the edited user changes.
  const [values, setValues] = useState(initial ? { ...EMPTY, ...initial } : EMPTY);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(values);
  };

  return (
    <form className="user-form" onSubmit={handleSubmit}>
      <h2>{initial ? "Edit user" : "Add user"}</h2>

      <div className="grid">
        <Input
          label="First name"
          name="first_name"
          value={values.first_name}
          onChange={handleChange}
          error={errors.first_name}
          required
        />
        <Input
          label="Last name"
          name="last_name"
          value={values.last_name}
          onChange={handleChange}
          error={errors.last_name}
          required
        />
        <Input
          label="Email"
          type="email"
          name="mail_id"
          value={values.mail_id}
          onChange={handleChange}
          error={errors.mail_id}
          required
        />
        <Input
          label="Phone number"
          name="phone_number"
          value={values.phone_number}
          onChange={handleChange}
          error={errors.phone_number}
          required
        />
        <Select
          label="Gender"
          name="gender"
          value={values.gender}
          onChange={handleChange}
          options={GENDERS}
          error={errors.gender}
        />
      </div>

      <div className="actions">
        <Button type="submit" disabled={busy}>
          {initial ? "Update" : "Create"}
        </Button>
        {initial && (
          <Button variant="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
        )}
      </div>
    </form>
  );
}
