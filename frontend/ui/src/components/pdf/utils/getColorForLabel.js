// Standalone function for use without React context
export function createLabelColorGenerator(initialColors = {}) {
    const cache = { ...initialColors };
    
    return (label) => {
      if (cache[label]) return cache[label];
      
      let hash = 0;
      for (let i = 0; i < label.length; i++) {
        hash = label.charCodeAt(i) + ((hash << 5) - hash);
      }
      
      const hue = Math.abs(hash) % 360;
      const saturation = 60 + (Math.abs(hash) % 30);
      const lightness = 45 + (Math.abs(hash * 7) % 30);
      const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      
      cache[label] = color;
      return color;
    };
  }

  export default {
    createLabelColorGenerator
  };