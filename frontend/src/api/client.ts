import type {
  Calibration,
  CalibrationFrame,
  Job,
  JobOptions,
  JobResult,
  PublicConfig,
  Video,
} from "./types";

const BASE_URL = "/api";
const TOKEN_STORAGE_KEY = "tennis_tracking_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token);
  else localStorage.removeItem(TOKEN_STORAGE_KEY);
}

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (!(init.body instanceof FormData) && init.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON - keep statusText
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  register: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  publicConfig: () => request<PublicConfig>("/config"),

  listVideos: () => request<Video[]>("/videos"),

  getVideo: (videoId: string) => request<Video>(`/videos/${videoId}`),

  uploadVideo: (file: File, onProgress?: (fraction: number) => void) =>
    new Promise<Video>((resolve, reject) => {
      const formData = new FormData();
      formData.append("file", file);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE_URL}/videos`);
      const token = getToken();
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          let detail = xhr.statusText;
          try {
            detail = JSON.parse(xhr.responseText).detail ?? detail;
          } catch {
            // ignore
          }
          reject(new ApiError(xhr.status, detail));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
      xhr.send(formData);
    }),

  createCalibrationFrame: (videoId: string) =>
    request<CalibrationFrame>(`/videos/${videoId}/calibration-frame`, { method: "POST" }),

  createCalibration: (
    videoId: string,
    points: {
      baseline_left: [number, number];
      baseline_right: [number, number];
      service_left: [number, number];
      service_right: [number, number];
    },
    courtType: "singles" | "doubles"
  ) =>
    request<Calibration>(`/videos/${videoId}/calibration`, {
      method: "POST",
      body: JSON.stringify({
        baseline_left: { x: points.baseline_left[0], y: points.baseline_left[1] },
        baseline_right: { x: points.baseline_right[0], y: points.baseline_right[1] },
        service_left: { x: points.service_left[0], y: points.service_left[1] },
        service_right: { x: points.service_right[0], y: points.service_right[1] },
        court_type: courtType,
      }),
    }),

  createJob: (videoId: string, options: JobOptions, calibrationId?: string) =>
    request<Job>(`/videos/${videoId}/jobs`, {
      method: "POST",
      body: JSON.stringify({ ...options, calibration_id: calibrationId ?? null }),
    }),

  getJob: (jobId: string) => request<Job>(`/jobs/${jobId}`),

  getJobResult: (jobId: string) => request<JobResult>(`/jobs/${jobId}/result`),
};

export { ApiError };
