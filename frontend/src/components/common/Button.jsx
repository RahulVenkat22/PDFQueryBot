/**
 * Reusable button.
 *
 * @param {"primary"|"secondary"|"danger"|"link"} [variant="primary"]
 */
export default function Button({ variant = "primary", type = "button", className = "", children, ...rest }) {
  return (
    <button type={type} className={`btn btn-${variant} ${className}`.trim()} {...rest}>
      {children}
    </button>
  );
}
