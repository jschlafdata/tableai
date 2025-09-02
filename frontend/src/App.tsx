
// src/App.tsx
import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import AuthCard from "./components/auth/AuthCard";
import HomePage from "./components/HomePage";
import ClassifyPage from "./components/classify/ClassifyPage";
import PdfViewer from "./components/pdf/PdfViewer";
import { me, setToken } from "./api";
import type { User } from "./types/user";
import IntegrationsPage from "./components/integrations/IntegrationsPage";
import CloudIntegrationsPage from "./components/integrations/CloudIntegrationsPage";
import UnifiedFileViewer from "./components/integrations/UnifiedFileViewer";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated
    me()
      .then((userData) => {
        console.log("User authenticated:", userData);
        setUser(userData);
      })
      .catch((err) => {
        console.log("Not authenticated:", err);
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleAuthSuccess = (authenticatedUser: User) => {
    console.log("Auth success:", authenticatedUser);
    setUser(authenticatedUser);
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
  };

  if (loading) {
    return (
      <div style={{ 
        display: "flex", 
        justifyContent: "center", 
        alignItems: "center", 
        height: "100vh",
        background: "linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%)"
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{
            width: "48px",
            height: "48px",
            border: "4px solid #e5e7eb",
            borderTop: "4px solid #2563eb",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
            margin: "0 auto 16px"
          }} />
          <div style={{ color: "#6b7280" }}>Loading...</div>
        </div>
        <style>
          {`@keyframes spin { to { transform: rotate(360deg); } }`}
        </style>
      </div>
    );
  }

  // If not authenticated, show auth card
  if (!user) {
    return <AuthCard initialMode="login" onSuccess={handleAuthSuccess} showLinks />;
  }

  // If authenticated, show the app with routing
  return (
    <Router>
      <Routes>
        {/* Home/Dashboard */}
        <Route path="/" element={<HomePage user={user} />} />
        {/* Document Processing - uses the ClassifyPage component */}
        <Route path="/processing/classifications" element={<ClassifyPage user={user} />} />
        <Route path="/processing/pdf_viewer" element={<PdfViewer user={user} />} />
        <Route path="/integrations-page" element={<IntegrationsPage user={user} />} />
        <Route path="/integrations" element={<CloudIntegrationsPage user={user} />} />
        <Route path="/cloud_storage/documents" element={<UnifiedFileViewer user={user} />} />
        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}