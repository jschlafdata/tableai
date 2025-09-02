import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

// Token helpers
export function setToken(token: string | null) {
  if (token) {
    localStorage.setItem("token", token);
    api.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    localStorage.removeItem("token");
    delete api.defaults.headers.common.Authorization;
  }
}
const saved = localStorage.getItem("token");
if (saved) setToken(saved);

// Auth calls
export async function login(email: string, password: string) {
  const form = new URLSearchParams();
  form.append("username", email); // OAuth2PasswordRequestForm expects 'username' by spec
  form.append("password", password);
  const { data } = await api.post("/auth/login", form, {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  setToken(data.access_token);
  return data;
}

export async function register(payload: { email: string; password: string; full_name?: string }) {
  const { data } = await api.post("/auth/register", payload);
  setToken(data.access_token);
  return data;
}

export async function me() {
  const { data } = await api.get("/users/me");
  return data;
}

export async function updateUser(id: number, patch: Record<string, unknown>) {
  const { data } = await api.patch(`/users/${id}`, patch);
  return data;
}
