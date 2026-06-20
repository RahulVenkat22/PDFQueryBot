/**
 * Generic, column-driven data table.
 *
 * @param {{key: string, header: string, render?: (row) => any, className?: string}[]} columns
 * @param {object[]} data
 * @param {(row) => string|number} rowKey  Unique key extractor (defaults to `row.id`).
 * @param {string} [empty]  Message shown when there are no rows.
 */
export default function Table({ columns, data, rowKey = (row) => row.id, empty = "No records found." }) {
  if (!data.length) {
    return <p className="table-empty">{empty}</p>;
  }

  return (
    <table className="table">
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={col.key} className={col.className}>
              {col.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row) => (
          <tr key={rowKey(row)}>
            {columns.map((col) => (
              <td key={col.key} className={col.className}>
                {col.render ? col.render(row) : row[col.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
