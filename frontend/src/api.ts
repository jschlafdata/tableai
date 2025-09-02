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


export async function startDropboxSync(baseUrl: string, body: {
  root_path?: string;
  bucket?: string;
  prefix?: string;
  force?: boolean;
}) {
  const resp = await fetch(`${baseUrl}/integrations/dropbox/sync`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      root_path: body.root_path ?? "/",
      bucket: body.bucket,
      prefix: body.prefix,
      force: body.force ?? false,
    }),
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json() as Promise<{ task_id: string }>;
}

export async function getTaskStatus(baseUrl: string, task_id: string) {
  // Works for both /integrations/dropbox/sync/status and /classifier/task/status
  const sync = await fetch(`${baseUrl}/integrations/dropbox/sync/status?task_id=${encodeURIComponent(task_id)}`);
  if (sync.ok) return sync.json();
  const cls = await fetch(`${baseUrl}/classifier/task/status?task_id=${encodeURIComponent(task_id)}`);
  if (cls.ok) return cls.json();
  throw new Error("Task not found");
}

export async function classifyS3Now(baseUrl: string, opts: {
  bucket?: string;
  prefix?: string;
  min_cluster?: number;
  upload_yaml_to_s3?: boolean;
}) {
  const u = new URL(`${baseUrl}/classifier/classify-s3`);
  if (opts.bucket) u.searchParams.set("bucket", opts.bucket);
  if (opts.prefix) u.searchParams.set("prefix", opts.prefix);
  if (opts.min_cluster !== undefined) u.searchParams.set("min_cluster", String(opts.min_cluster));
  if (opts.upload_yaml_to_s3 !== undefined) u.searchParams.set("upload_yaml_to_s3", String(opts.upload_yaml_to_s3));
  const resp = await fetch(u.toString(), { method: "POST" });
  if (!resp.ok) throw new Error(await resp.text());
  const blob = await resp.blob();
  return blob; // clusters.yaml blob (download or parse)
}

export async function classifyS3Async(baseUrl: string, opts: {
  bucket?: string;
  prefix?: string;
  min_cluster?: number;
  upload_yaml_to_s3?: boolean;
}) {
  const u = new URL(`${baseUrl}/classifier/classify-s3/async`);
  if (opts.bucket) u.searchParams.set("bucket", opts.bucket);
  if (opts.prefix) u.searchParams.set("prefix", opts.prefix);
  if (opts.min_cluster !== undefined) u.searchParams.set("min_cluster", String(opts.min_cluster));
  if (opts.upload_yaml_to_s3 !== undefined) u.searchParams.set("upload_yaml_to_s3", String(opts.upload_yaml_to_s3));
  const resp = await fetch(u.toString(), { method: "POST" });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json() as Promise<{ task_id: string }>;
}
