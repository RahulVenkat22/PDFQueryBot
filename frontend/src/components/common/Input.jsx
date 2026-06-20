/**
 * Labelled text input with an optional validation error message.
 * Any extra props (name, type, value, onChange, required…) pass through.
 */
export default function Input({ label, error, className = "", ...rest }) {
  return (
    <label className={`field ${className}`.trim()}>
      {label && <span className="field-label">{label}</span>}
      <input className={error ? "has-error" : ""} {...rest} />
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}
