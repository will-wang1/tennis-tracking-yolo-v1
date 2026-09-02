import type { JobOptions } from "../api/types";

interface Props {
  value: JobOptions;
  onChange: (value: JobOptions) => void;
  minimapAvailable: boolean;
}

export default function FeatureToggles({ value, onChange, minimapAvailable }: Props) {
  function set<K extends keyof JobOptions>(key: K, checked: boolean) {
    onChange({ ...value, [key]: checked });
  }

  return (
    <div>
      <div className="toggle-row">
        <input
          type="checkbox"
          id="toggle-bounce"
          checked={value.bounce}
          onChange={(e) => set("bounce", e.target.checked)}
        />
        <label htmlFor="toggle-bounce">
          Bounce detection <small>- marks where the ball lands</small>
        </label>
      </div>
      <div className="toggle-row">
        <input
          type="checkbox"
          id="toggle-speed"
          checked={value.speed}
          onChange={(e) => set("speed", e.target.checked)}
        />
        <label htmlFor="toggle-speed">
          Shot speed <small>- px/s, or km/h if you calibrate the court</small>
        </label>
      </div>
      <div className="toggle-row">
        <input
          type="checkbox"
          id="toggle-sidebar"
          checked={value.sidebar}
          onChange={(e) => set("sidebar", e.target.checked)}
        />
        <label htmlFor="toggle-sidebar">
          Sidebar panel <small>- live stroke/speed readout composited on the video</small>
        </label>
      </div>
      <div className="toggle-row">
        <input
          type="checkbox"
          id="toggle-minimap"
          checked={value.minimap}
          disabled={!minimapAvailable}
          onChange={(e) => set("minimap", e.target.checked)}
        />
        <label htmlFor="toggle-minimap">
          Minimap overlay{" "}
          <small>
            {minimapAvailable
              ? "- bird's-eye view of ball/player positions"
              : "- unavailable on this server (no court-keypoint model configured)"}
          </small>
        </label>
      </div>
    </div>
  );
}
