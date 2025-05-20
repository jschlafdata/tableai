// /utils/getColorForTable.js
const tableColors = [
    'hsl(210, 80%, 60%)',
    'hsl(180, 70%, 50%)',
    'hsl(150, 65%, 50%)',
    'hsl(270, 60%, 60%)',
    'hsl(240, 70%, 65%)',
    'hsl(330, 70%, 60%)',
    'hsl(30, 80%, 55%)'
  ];
  
  /**
   * Returns a color for a given tableIndex, maintaining consistency across your UI
   */
  export function getColorForTable(tableIndex) {
    // "Table 1" => color[0], "Table 2" => color[1], etc.
    const idx = Math.max(0, (tableIndex - 1) % tableColors.length);
    return tableColors[idx];
  }