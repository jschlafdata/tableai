
// src/App.tsx
import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import AuthCard from "./components/auth/AuthCard";
import HomePage from "./components/HomePage";
import ClassifyPage from "./components/classify/ClassifyPage";
import PdfViewer from "./components/pdf/PdfViewer";
import { me, setToken } from "./api";
import type { User } from "./types/user";

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
        <Route path="/processing" element={<ClassifyPage user={user} />} />
        
        {/* PDF Viewer for S3 documents */}
        <Route path="/processing/pdf/s3_viewer" element={<PdfViewer user={user} />} />
        <Route path="/processing/pdf_viewer" element={<PdfViewer user={user} />} />
        
        {/* Classifications page */}
        <Route path="/processing/classifications" element={<ClassifyPage user={user} />} />
        
        {/* Cloud Storage */}
        <Route path="/cloud_storage/*" element={
          <div style={{ padding: "20px" }}>
            <h1>Cloud Storage</h1>
            <p>Cloud storage component to be implemented</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        
        {/* Profile pages */}
        <Route path="/profile" element={
          <div style={{ padding: "20px" }}>
            <h1>Profile</h1>
            <p>Profile component to be implemented</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        <Route path="/profile/info" element={
          <div style={{ padding: "20px" }}>
            <h1>Profile Info</h1>
            <p>Change your password and account settings</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        <Route path="/profile/integrations" element={
          <div style={{ padding: "20px" }}>
            <h1>Integrations</h1>
            <p>Connect your cloud storage accounts</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        <Route path="/profile/settings" element={
          <div style={{ padding: "20px" }}>
            <h1>Settings</h1>
            <p>Settings component to be implemented</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        <Route path="/profile/security" element={
          <div style={{ padding: "20px" }}>
            <h1>Security</h1>
            <p>Security settings to be implemented</p>
            <a href="/">Back to Home</a>
          </div>
        } />
        
        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}