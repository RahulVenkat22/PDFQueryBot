/**
 * Dismissible status banner.
 *
 * @param {"success"|"error"|"info"} [type="info"]
 */
export default function Banner({ type = "info", children, onDismiss }) {
  if (!children) return null;

  return (
    <div className={`banner banner-${type}`} onClick={onDismiss} role="status">
      {children}
    </div>
  );
}
