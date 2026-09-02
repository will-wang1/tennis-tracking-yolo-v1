import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ShotSpeedStat } from "../api/types";

export default function ShotSpeedChart({ shots }: { shots: ShotSpeedStat[] }) {
  if (shots.length === 0) return null;

  const unit = shots[0].unit;
  const data = shots.map((shot, index) => ({
    name: `#${index + 1}`,
    speed: Math.round(shot.peak_speed),
  }));

  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#263241" />
          <XAxis dataKey="name" stroke="#8b96a3" fontSize={12} />
          <YAxis stroke="#8b96a3" fontSize={12} unit={` ${unit}`} />
          <Tooltip
            contentStyle={{ background: "#16212e", border: "1px solid #263241", color: "#e6edf3" }}
            formatter={(value: number) => [`${value} ${unit}`, "Peak speed"]}
          />
          <Bar dataKey="speed" fill="#4fd1a5" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
