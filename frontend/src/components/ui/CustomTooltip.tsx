// src/components/ui/CustomTooltip.tsx
import { useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { 
  Info, 
  Lightbulb, 
  AlertTriangle, 
  AlertCircle, 
  FileText 
} from "lucide-react";

// Types
type TooltipVariant = "note" | "tip" | "info" | "warning" | "danger";

interface CustomTooltipProps {
  variant?: TooltipVariant;
  title?: string;
  children: ReactNode;
  placement?: "top" | "bottom" | "left" | "right";
}

interface VariantStyle {
  icon: ReactNode;
  color: string;
  background: string;
  border: string;
}

const VARIANT_STYLES: Record<TooltipVariant, VariantStyle> = {
  note: {
    icon: <FileText size={14} />,
    color: "#444",
    background: "#f9f9f9",
    border: "1px solid #ddd",
  },
  tip: {
    icon: <Lightbulb size={14} />,
    color: "#1a4731",
    background: "#e6f4ea",
    border: "1px solid #c8e6c9",
  },
  info: {
    icon: <Info size={14} />,
    color: "#0d3b66",
    background: "#e1f5fe",
    border: "1px solid #b3e5fc",
  },
  warning: {
    icon: <AlertTriangle size={14} />,
    color: "#7a4f01",
    background: "#fff8e1",
    border: "1px solid #ffe082",
  },
  danger: {
    icon: <AlertCircle size={14} />,
    color: "#761b18",
    background: "#fdecea",
    border: "1px solid #f5c6cb",
  },
};

const CustomTooltip: React.FC<CustomTooltipProps> = ({ 
  variant = "info", 
  title = "", 
  children,
  placement = "top" 
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const style = VARIANT_STYLES[variant];

  const handleMouseEnter = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    
    let x = rect.left + rect.width / 2;
    let y = rect.top;
    
    switch (placement) {
      case "bottom":
        y = rect.bottom;
        break;
      case "left":
        x = rect.left;
        y = rect.top + rect.height / 2;
        break;
      case "right":
        x = rect.right;
        y = rect.top + rect.height / 2;
        break;
      default: // top
        y = rect.top;
    }
    
    setPosition({ x, y });
    setIsVisible(true);
  };

  const handleMouseLeave = () => {
    setIsVisible(false);
  };

  const getTooltipPosition = (): CSSProperties => {
    const offset = 8;
    let transform = "";
    let top = position.y;
    let left = position.x;

    switch (placement) {
      case "bottom":
        top = position.y + offset;
        transform = "translateX(-50%)";
        break;
      case "left":
        left = position.x - offset;
        transform = "translate(-100%, -50%)";
        break;
      case "right":
        left = position.x + offset;
        transform = "translateY(-50%)";
        break;
      default: // top
        top = position.y - offset;
        transform = "translate(-50%, -100%)";
    }

    return {
      position: "fixed",
      top: `${top}px`,
      left: `${left}px`,
      transform,
      zIndex: 9999,
    };
  };

  const tooltipStyles: Record<string, CSSProperties> = {
    wrapper: {
      position: "relative",
      display: "inline-block",
    },
    tooltip: {
      ...getTooltipPosition(),
      backgroundColor: style.background,
      color: style.color,
      border: style.border,
      borderRadius: "6px",
      padding: "8px 12px",
      maxWidth: "280px",
      boxShadow: "0 2px 8px rgba(0, 0, 0, 0.15)",
      opacity: isVisible ? 1 : 0,
      visibility: isVisible ? "visible" : "hidden",
      transition: "opacity 0.2s, visibility 0.2s",
      pointerEvents: "none",
    },
    header: {
      display: "flex",
      alignItems: "center",
      gap: "6px",
      marginBottom: "4px",
    },
    label: {
      fontSize: "11px",
      fontWeight: 700,
      textTransform: "uppercase",
      letterSpacing: "0.5px",
    },
    content: {
      fontSize: "13px",
      lineHeight: 1.4,
    },
    arrow: {
      position: "absolute",
      width: "8px",
      height: "8px",
      backgroundColor: style.background,
      border: style.border,
      borderRight: "none",
      borderTop: "none",
      ...(placement === "top" && {
        bottom: "-5px",
        left: "50%",
        transform: "translateX(-50%) rotate(-45deg)",
      }),
      ...(placement === "bottom" && {
        top: "-5px",
        left: "50%",
        transform: "translateX(-50%) rotate(135deg)",
      }),
      ...(placement === "left" && {
        right: "-5px",
        top: "50%",
        transform: "translateY(-50%) rotate(45deg)",
      }),
      ...(placement === "right" && {
        left: "-5px",
        top: "50%",
        transform: "translateY(-50%) rotate(-135deg)",
      }),
    },
  };

  return (
    <>
      <div
        style={tooltipStyles.wrapper}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {children}
      </div>
      
      {/* Portal-like tooltip rendering */}
      {typeof document !== "undefined" && (
        <div style={tooltipStyles.tooltip}>
          <div style={tooltipStyles.arrow} />
          <div style={tooltipStyles.header}>
            {style.icon}
            <span style={tooltipStyles.label}>{variant}</span>
          </div>
          <div style={tooltipStyles.content}>{title}</div>
        </div>
      )}
    </>
  );
};

export default CustomTooltip;