import { useEffect, useRef, useState } from "react";

export interface CalibrationPoints {
  baseline_left: [number, number];
  baseline_right: [number, number];
  service_left: [number, number];
  service_right: [number, number];
}

const STEPS: { key: keyof CalibrationPoints; label: string }[] = [
  { key: "baseline_left", label: "Near baseline, LEFT corner" },
  { key: "baseline_right", label: "Near baseline, RIGHT corner" },
  { key: "service_left", label: "Near service line, LEFT corner" },
  { key: "service_right", label: "Near service line, RIGHT corner" },
];

interface Props {
  frameUrl: string;
  frameWidth: number;
  frameHeight: number;
  onComplete: (points: CalibrationPoints) => void;
}

export default function CalibrationCanvas({ frameUrl, frameWidth, frameHeight, onComplete }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<Partial<CalibrationPoints>>({});
  const [displayScale, setDisplayScale] = useState(1);

  useEffect(() => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.src = frameUrl;
    image.onload = () => {
      imageRef.current = image;
      draw();
    };
  }, [frameUrl]);

  function draw() {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const scale = canvas.width / frameWidth;
    Object.entries(points).forEach(([key, point]) => {
      if (!point) return;
      const [x, y] = point as [number, number];
      ctx.beginPath();
      ctx.arc(x * scale, y * scale, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#00d4a4";
      ctx.fill();
      ctx.strokeStyle = "#080c10";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#dce4ed";
      ctx.font = "12px 'Inter', sans-serif";
      ctx.fillText(key.replace("_", " "), x * scale + 10, y * scale - 10);
    });
  }

  useEffect(draw, [points]);

  const nextStep = STEPS.find((step) => points[step.key] === undefined);

  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!nextStep) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scale = frameWidth / rect.width;
    const x = (event.clientX - rect.left) * scale;
    const y = (event.clientY - rect.top) * scale;
    const updated = { ...points, [nextStep.key]: [x, y] as [number, number] };
    setPoints(updated);
    if (Object.keys(updated).length === STEPS.length) {
      onComplete(updated as CalibrationPoints);
    }
  }

  function reset() {
    setPoints({});
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maxWidth = 800;
    const scale = Math.min(1, maxWidth / frameWidth);
    canvas.width = frameWidth * scale;
    canvas.height = frameHeight * scale;
    setDisplayScale(scale);
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameWidth, frameHeight]);

  return (
    <div>
      <p>
        {nextStep
          ? `Click: ${nextStep.label}`
          : "All 4 points placed - review the markers, or reset to try again."}
      </p>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          cursor: nextStep ? "crosshair" : "default",
          maxWidth: "100%",
          border: "1px solid var(--border-default)",
          borderRadius: "var(--radius-lg)",
        }}
      />
      <div style={{ marginTop: 10 }}>
        <button className="btn btn-secondary" onClick={reset}>
          Reset points
        </button>
      </div>
      <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
        Displayed at {(displayScale * 100).toFixed(0)}% - clicks are mapped back to the original{" "}
        {frameWidth}×{frameHeight} frame.
      </p>
    </div>
  );
}
