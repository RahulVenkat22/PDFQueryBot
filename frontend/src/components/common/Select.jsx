/**
 * Labelled select with an optional validation error message.
 *
 * @param {{value: string, label: string}[]} options
 */
export default function Select({ label, error, options = [], className = "", ...rest }) {
  return (
    <label className={`field ${className}`.trim()}>
      {label && <span className="field-label">{label}</span>}
      <select className={error ? "has-error" : ""} {...rest}>
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}
