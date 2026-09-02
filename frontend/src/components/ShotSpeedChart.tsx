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
          <CartesianGrid strokeDasharray="3 3" stroke="#182230" />
          <XAxis dataKey="name" stroke="#566878" fontSize={11} fontFamily="JetBrains Mono, monospace" />
          <YAxis stroke="#566878" fontSize={11} fontFamily="JetBrains Mono, monospace" unit={` ${unit}`} />
          <Tooltip
            contentStyle={{
              background: "#0d1520",
              border: "1px solid #182230",
              borderRadius: 8,
              color: "#dce4ed",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 12,
            }}
            cursor={{ fill: "rgba(0,212,164,0.08)" }}
            formatter={(value: number) => [`${value} ${unit}`, "Peak speed"]}
          />
          <Bar dataKey="speed" fill="#00d4a4" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
