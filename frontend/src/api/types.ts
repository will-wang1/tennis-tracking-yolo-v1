export interface Video {
  id: string;
  filename: string;
  duration_s: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  status: string;
  created_at: string;
}

export interface CalibrationFrame {
  frame_url: string;
  width: number;
  height: number;
}

export interface CalibrationPoint {
  x: number;
  y: number;
}

export interface Calibration {
  id: string;
  video_id: string;
  court_type: "singles" | "doubles";
  pixel_points: Record<string, [number, number]>;
  created_at: string;
}

export interface JobOptions {
  bounce: boolean;
  speed: boolean;
  sidebar: boolean;
  minimap: boolean;
}

export interface Job {
  id: string;
  video_id: string;
  options: JobOptions;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ShotSpeedStat {
  start_frame: number;
  end_frame: number;
  peak_speed: number;
  unit: "km/h" | "px/s";
  method: string;
}

export interface RallyStat {
  start_frame: number;
  end_frame: number;
  duration_s: number;
  shot_count: number;
  bounce_count: number;
  unattributed_count: number;
  peak_speed: number | null;
  peak_speed_unit: string | null;
}

export interface MatchStats {
  rally_count: number;
  rallies: RallyStat[];
  total_bounces: number;
  total_contacts: number;
  total_unattributed: number;
  shot_speeds: ShotSpeedStat[];
  bounce_locations: [number, number][];
  near_shot_counts: Record<string, number> | null;
  far_shot_counts: Record<string, number> | null;
}

export interface JobResult {
  id: string;
  status: string;
  video_url: string | null;
  stats: MatchStats | null;
}

export interface PublicConfig {
  minimap_available: boolean;
  invite_required: boolean;
}
