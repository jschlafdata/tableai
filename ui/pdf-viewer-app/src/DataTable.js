import React from 'react';

function DataTable({ data, columns, totalsRow }) {
  if (!data || data.length === 0) {
    return <div>No table data available.</div>;
  }

  return (
    <div className="data-table-container" style={{ marginTop: '20px', overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #ccc' }}>
        <thead>
          <tr>
            {columns.map((col, idx) => (
              <th key={idx} style={{ padding: '8px 12px', backgroundColor: '#f2f2f2', border: '1px solid #ccc', textAlign: 'left' }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIdx) => (
            <tr key={rowIdx}>
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} style={{ padding: '8px 12px', border: '1px solid #ccc' }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        {totalsRow && (
          <tfoot>
            <tr style={{ backgroundColor: '#f9f9f9', fontWeight: 'bold' }}>
              {totalsRow.map((cell, idx) => (
                <td key={idx} style={{ padding: '8px 12px', border: '1px solid #ccc' }}>
                  {cell}
                </td>
              ))}
            </tr>
          </tfoot>
        )}
      </table>
    </div>
  );
}

export default DataTable;