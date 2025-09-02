// src/components/HomePage.tsx - Using simplified PageLayout
import React from "react";
import { useNavigate } from "react-router-dom";
import type { CSSProperties } from "react";
import type { User } from "../types/user";
import PageLayout from "./layouts/PageLayout";
import UserDropdown from "./auth/UserDropdown";

interface QuickAction {
  title: string;
  description: string;
  icon: string;
  color?: string;
  path: string;
}

interface QuickActionCardProps extends QuickAction {
  onClick: () => void;
}

const QuickActionCard: React.FC<QuickActionCardProps> = ({ 
  title, 
  description, 
  icon, 
  onClick, 
  color = "#2563eb" 
}) => {
  const [hover, setHover] = React.useState(false);

  const styles: Record<string, CSSProperties> = {
    card: {
      backgroundColor: "white",
      borderRadius: "12px",
      padding: "24px",
      boxShadow: hover 
        ? "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
        : "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
      height: "100%",
      cursor: "pointer",
      transition: "transform 0.2s, box-shadow 0.2s",
      textAlign: "center",
      transform: hover ? "translateY(-2px)" : "translateY(0)",
    },
    iconContainer: {
      backgroundColor: color,
      borderRadius: "50%",
      width: "56px",
      height: "56px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "white",
      fontSize: "24px",
      margin: "0 auto 16px",
    },
    title: {
      fontSize: "1.125rem",
      fontWeight: 600,
      color: "#111827",
      marginBottom: "8px",
    },
    description: {
      fontSize: "14px",
      color: "#6b7280",
    },
  };

  return (
    <div 
      style={styles.card}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div style={styles.iconContainer}>{icon}</div>
      <div style={styles.title}>{title}</div>
      <div style={styles.description}>{description}</div>
    </div>
  );
};

interface HomePageProps {
  user: User | null;
}

const HomePage: React.FC<HomePageProps> = ({ user }) => {
  const navigate = useNavigate();

  const quickActions: QuickAction[] = [
    {
      title: "Document Processing",
      description: "Process and extract data from PDFs",
      icon: "📄",
      color: "#2563eb",
      path: "/processing",
    },
    {
      title: "Cloud Storage",
      description: "Manage your cloud files",
      icon: "☁️",
      color: "#7c3aed",
      path: "/cloud_storage/documents",
    },
    {
      title: "Integrations",
      description: "Connect your cloud storage",
      icon: "🔗",
      color: "#059669",
      path: "/profile/integrations",
    },
    {
      title: "Change Password",
      description: "Update your account security",
      icon: "🔒",
      color: "#dc2626",
      path: "/profile/info",
    },
  ];

  const styles: Record<string, CSSProperties> = {
    welcomeMessage: {
      marginBottom: "32px",
    },
    subtitle: {
      fontSize: "1rem",
      color: "#6b7280",
      margin: 0,
    },
    section: {
      marginBottom: "24px",
    },
    sectionTitle: {
      fontSize: "1.5rem",
      fontWeight: "bold",
      color: "#111827",
      margin: "0 0 16px 0",
    },
    grid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
      gap: "16px",
    },
  };

  return (
    <PageLayout 
      title={`Welcome back${user?.full_name ? `, ${user.full_name}` : ""}! 👋`}
      headerRight={<UserDropdown user={user} />}
    >
      <div style={styles.welcomeMessage}>
        <p style={styles.subtitle}>Here's your personal dashboard overview.</p>
      </div>

      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Quick Actions</h2>
        <div style={styles.grid}>
          {quickActions.map((action, index) => (
            <QuickActionCard
              key={index}
              {...action}
              onClick={() => navigate(action.path)}
            />
          ))}
        </div>
      </div>
    </PageLayout>
  );
};

export default HomePage;