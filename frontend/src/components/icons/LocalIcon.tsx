// src/components/icons/LocalIcon.tsx
import React from "react";
import type { CSSProperties, ImgHTMLAttributes } from "react";

interface LocalIconProps extends Omit<ImgHTMLAttributes<HTMLImageElement>, "style"> {
  src: string;
  alt: string;
  active?: boolean;
  style?: CSSProperties;
  size?: number;
}

/**
 * A component for displaying local image icons with consistent styling
 */
const LocalIcon: React.FC<LocalIconProps> = ({ 
  src, 
  alt, 
  active = false, 
  style,
  size = 20,
  ...props 
}) => {
  const defaultStyles: CSSProperties = {
    width: `${size}px`,
    height: `${size}px`,
    objectFit: "contain",
    opacity: active ? 1 : 0.7,
    transition: "opacity 0.2s ease",
    ...style,
  };

  return (
    <img
      src={src}
      alt={alt}
      style={defaultStyles}
      {...props}
    />
  );
};

export default LocalIcon;