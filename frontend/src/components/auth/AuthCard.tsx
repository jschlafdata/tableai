import React, { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { Eye, EyeOff, User as UserIcon, Lock, Mail } from "lucide-react";
import { login, register, me } from "../../api";
import type { User } from "../../types/user";

export type AuthMode = "login" | "signup";

interface AuthCardProps {
  /** "login" or "signup" */
  initialMode?: AuthMode;
  /** called when auth succeeds; receive /users/me payload */
  onSuccess?: (user: User) => void;
  /** show quick links to /docs and /admin below the card */
  showLinks?: boolean;
}

const styles: Record<string, CSSProperties> = {
  container: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "1rem",
  },
  card: {
    background: "white",
    borderRadius: "1rem",
    boxShadow:
      "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
    padding: "2rem",
    width: "100%",
    maxWidth: "28rem",
  },
  header: { textAlign: "center", marginBottom: "2rem" },
  iconContainer: {
    width: "4rem",
    height: "4rem",
    background: "#dbeafe",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    margin: "0 auto 1rem",
  },
  title: {
    fontSize: "1.875rem",
    fontWeight: "bold",
    color: "#111827",
    margin: "0 0 0.5rem 0",
  },
  subtitle: { color: "#6b7280", margin: 0 },
  error: {
    background: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#b91c1c",
    padding: "0.75rem 1rem",
    borderRadius: "0.5rem",
    marginBottom: "1.5rem",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  success: {
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    color: "#166534",
    padding: "0.75rem 1rem",
    borderRadius: "0.5rem",
    marginBottom: "1.5rem",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  formGroup: { marginBottom: "1.5rem" },
  label: {
    display: "block",
    fontSize: "0.875rem",
    fontWeight: 500,
    color: "#374151",
    marginBottom: "0.5rem",
  },
  inputContainer: { position: "relative" },
  input: {
    width: "100%",
    paddingLeft: "2.5rem",
    paddingRight: "1rem",
    paddingTop: "0.75rem",
    paddingBottom: "0.75rem",
    border: "1px solid #d1d5db",
    borderRadius: "0.5rem",
    fontSize: "1rem",
    outline: "none",
    transition: "all 0.2s",
    boxSizing: "border-box",
  },
  inputWithIcon: { paddingRight: "3rem" },
  inputError: {
    borderColor: "#ef4444",
    boxShadow: "0 0 0 3px rgba(239, 68, 68, 0.1)",
  },
  icon: {
    position: "absolute",
    left: "0.75rem",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#9ca3af",
    width: "1.25rem",
    height: "1.25rem",
  },
  eyeIcon: {
    position: "absolute",
    right: "0.75rem",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#9ca3af",
    width: "1.25rem",
    height: "1.25rem",
    cursor: "pointer",
    background: "none",
    border: "none",
    padding: 0,
  },
  button: {
    width: "100%",
    background: "#2563eb",
    color: "white",
    padding: "0.75rem 1rem",
    borderRadius: "0.5rem",
    border: "none",
    fontSize: "1rem",
    fontWeight: 500,
    cursor: "pointer",
    transition: "background-color 0.2s",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "0.5rem",
  },
  buttonDisabled: { opacity: 0.5, cursor: "not-allowed" },
  textButton: {
    background: "none",
    border: "none",
    color: "#2563eb",
    cursor: "pointer",
    fontSize: "1rem",
    fontWeight: 500,
    marginTop: "1.5rem",
  },
  passwordStrength: { fontSize: "0.75rem", marginTop: "0.25rem", color: "#6b7280" },
  links: { display: "flex", gap: 12, justifyContent: "center", marginTop: 16 },
};

type FormData = {
  email: string;
  password: string;
  confirm_password: string;
  name: string;
};

const initialForm: FormData = {
  email: "",
  password: "",
  confirm_password: "",
  name: "",
};

export default function AuthCard({
  initialMode = "login",
  onSuccess,
  showLinks = true,
}: AuthCardProps) {
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [form, setForm] = useState<FormData>(initialForm);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string>("");

  // If already authenticated, resolve right away.
  useEffect(() => {
    me().then((u) => onSuccess?.(u)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const passwordsMatch = useMemo(
    () => !!form.password && !!form.confirm_password && form.password === form.confirm_password,
    [form.password, form.confirm_password]
  );
  const passwordsDontMatch = useMemo(
    () => !!form.confirm_password && form.password !== form.confirm_password,
    [form.password, form.confirm_password]
  );

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
    setMessage("");
  }

  // keep these simple for TS: set specific CSS props instead of Object.assign
  function handleFocus(e: React.FocusEvent<HTMLInputElement>) {
    e.currentTarget.style.borderColor = "#3b82f6";
    e.currentTarget.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.1)";
  }
  function handleBlur(e: React.FocusEvent<HTMLInputElement>) {
    // red border only if confirm doesn't match; else reset to default
    e.currentTarget.style.borderColor = passwordsDontMatch ? "#ef4444" : "#d1d5db";
    e.currentTarget.style.boxShadow = "none";
  }

  function validateRegistration(): boolean {
    if (!form.name.trim()) {
      setMessage("Name is required");
      return false;
    }
    if (!form.email.trim()) {
      setMessage("Email is required");
      return false;
    }
    // backend uses min_length=6
    if (form.password.length < 6) {
      setMessage("Password must be at least 6 characters");
      return false;
    }
    if (form.password !== form.confirm_password) {
      setMessage("Passwords do not match");
      return false;
    }
    return true;
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setMessage("");
    try {
      if (mode === "login") {
        await login(form.email, form.password);
      } else {
        if (!validateRegistration()) return;
        // register returns a token; user is logged in immediately
        await register({ email: form.email, password: form.password, full_name: form.name });
      }
      const u = await me();
      onSuccess?.(u);
      setForm(initialForm);
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        (mode === "login" ? "Login failed" : "Registration failed");
      setMessage(detail);
    } finally {
      setLoading(false);
    }
  }

  function toggleMode() {
    setMode((m) => (m === "login" ? "signup" : "login"));
    setMessage("");
    setForm(initialForm);
  }

  const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <div style={styles.iconContainer}>
            <Lock style={{ width: "2rem", height: "2rem", color: "#2563eb" }} />
          </div>
          <h2 style={styles.title}>{mode === "login" ? "Welcome Back" : "Create Account"}</h2>
          <p style={styles.subtitle}>
            {mode === "login" ? "Sign in to your account" : "Sign up to get started"}
          </p>
        </div>

        {!!message && (
          <div style={message.toLowerCase().includes("success") ? styles.success : styles.error}>
            {message.toLowerCase().includes("success") ? "✅" : "⚠️"} {message}
          </div>
        )}

        <form onSubmit={onSubmit}>
          {mode === "signup" && (
            <div style={styles.formGroup}>
              <label style={styles.label}>Full Name</label>
              <div style={styles.inputContainer}>
                <UserIcon style={styles.icon} />
                <input
                  type="text"
                  name="name"
                  value={form.name}
                  onChange={onInputChange}
                  style={styles.input}
                  placeholder="Enter your full name"
                  required={mode === "signup"}
                  onFocus={handleFocus}
                  onBlur={handleBlur}
                />
              </div>
            </div>
          )}

          <div style={styles.formGroup}>
            <label style={styles.label}>Email Address</label>
            <div style={styles.inputContainer}>
              <Mail style={styles.icon} />
              <input
                type="email"
                name="email"
                value={form.email}
                onChange={onInputChange}
                style={styles.input}
                placeholder="Enter your email"
                required
                onFocus={handleFocus}
                onBlur={handleBlur}
              />
            </div>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Password</label>
            <div style={styles.inputContainer}>
              <Lock style={styles.icon} />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={form.password}
                onChange={onInputChange}
                style={{ ...styles.input, ...styles.inputWithIcon }}
                placeholder="Enter your password"
                required
                onFocus={handleFocus}
                onBlur={handleBlur}
              />
              <button
                type="button"
                onClick={() => setShowPassword((s) => !s)}
                style={styles.eyeIcon}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff /> : <Eye />}
              </button>
            </div>
            {mode === "signup" && form.password && (
              <div style={styles.passwordStrength}>
                {form.password.length < 6
                  ? "⚠️ Password must be at least 6 characters"
                  : "✅ Password length OK"}
              </div>
            )}
          </div>

          {mode === "signup" && (
            <div style={styles.formGroup}>
              <label style={styles.label}>Confirm Password</label>
              <div style={styles.inputContainer}>
                <Lock style={styles.icon} />
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  name="confirm_password"
                  value={form.confirm_password}
                  onChange={onInputChange}
                  style={{
                    ...styles.input,
                    ...styles.inputWithIcon,
                    ...(passwordsDontMatch ? styles.inputError : {}),
                  }}
                  placeholder="Confirm your password"
                  required={mode === "signup"}
                  onFocus={handleFocus}
                  onBlur={handleBlur}
                  aria-invalid={passwordsDontMatch || undefined}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((s) => !s)}
                  style={styles.eyeIcon}
                  aria-label={showConfirmPassword ? "Hide confirm password" : "Show confirm password"}
                >
                  {showConfirmPassword ? <EyeOff /> : <Eye />}
                </button>
              </div>
              {form.confirm_password && (
                <div style={styles.passwordStrength}>
                  {passwordsMatch ? "✅ Passwords match" : "⚠️ Passwords do not match"}
                </div>
              )}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
            onMouseOver={(e) => {
              if (!loading) (e.currentTarget as HTMLButtonElement).style.background = "#1d4ed8";
            }}
            onMouseOut={(e) => {
              if (!loading) (e.currentTarget as HTMLButtonElement).style.background = "#2563eb";
            }}
          >
            {loading ? (
              <>
                <div
                  style={{
                    width: "1rem",
                    height: "1rem",
                    border: "2px solid transparent",
                    borderTop: "2px solid currentColor",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                  }}
                />
                Processing...
              </>
            ) : mode === "login" ? (
              "Sign In"
            ) : (
              "Create Account"
            )}
          </button>
        </form>

        <div style={{ textAlign: "center" }}>
          <button
            type="button"
            onClick={toggleMode}
            style={styles.textButton}
            onMouseOver={(e) => (e.currentTarget.style.color = "#1d4ed8")}
            onMouseOut={(e) => (e.currentTarget.style.color = "#2563eb")}
          >
            {mode === "login"
              ? "Don't have an account? Sign up"
              : "Already have an account? Sign in"}
          </button>
        </div>

        {showLinks && (
          <>
            <hr style={{ margin: "24px 0" }} />
            <div style={styles.links}>
              <a href={`${apiUrl}/docs`} target="_blank" rel="noreferrer">OpenAPI /docs</a>
              <a href={`${apiUrl}/admin`} target="_blank" rel="noreferrer">Admin</a>
            </div>
          </>
        )}

        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    </div>
  );
}
