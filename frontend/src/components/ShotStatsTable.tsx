import type { ShotSpeedStat } from "../api/types";

export default function ShotStatsTable({ shots }: { shots: ShotSpeedStat[] }) {
  if (shots.length === 0) return <p style={{ color: "var(--text-muted)" }}>No shots recorded.</p>;

  return (
    <div style={{ overflowX: "auto" }}>
      <table className="stats-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Frames</th>
            <th>Peak speed</th>
            <th>Method</th>
          </tr>
        </thead>
        <tbody>
          {shots.map((shot, index) => (
            <tr key={`${shot.start_frame}-${shot.end_frame}`}>
              <td>{index + 1}</td>
              <td>
                {shot.start_frame}–{shot.end_frame}
              </td>
              <td className="numeric">
                {shot.peak_speed.toFixed(0)} {shot.unit}
              </td>
              <td>{shot.method}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
