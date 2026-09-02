// Full tennis court, drawn to scale in meters and mapped to pixels: doubles
// envelope 10.97 x 23.77m (see src/analysis/court_calibration.py -
// FULL_COURT_REFERENCE_POINTS), origin at the near baseline-left corner,
// matching the world coordinates CourtCalibration.pixel_to_world produces.
const COURT_WIDTH_M = 10.97;
const COURT_LENGTH_M = 23.77;
const SINGLES_INSET_M = (COURT_WIDTH_M - 8.23) / 2;
const SERVICE_LINE_M = 5.485;
const NET_Y_M = COURT_LENGTH_M / 2;

const PADDING_PX = 16;
const COURT_WIDTH_PX = 260;
const SCALE = COURT_WIDTH_PX / COURT_WIDTH_M;
const COURT_LENGTH_PX = COURT_LENGTH_M * SCALE;
const SVG_WIDTH = COURT_WIDTH_PX + PADDING_PX * 2;
const SVG_HEIGHT = COURT_LENGTH_PX + PADDING_PX * 2;

function toSvg(xMeters: number, yMeters: number): [number, number] {
  return [PADDING_PX + xMeters * SCALE, PADDING_PX + yMeters * SCALE];
}

function Line({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) {
  const [px1, py1] = toSvg(x1, y1);
  const [px2, py2] = toSvg(x2, y2);
  return <line x1={px1} y1={py1} x2={px2} y2={py2} stroke="#3a4553" strokeWidth={2} />;
}

export default function BounceHeatmap({ locations }: { locations: [number, number][] }) {
  if (locations.length === 0) {
    return <p>No bounce locations available - run with bounce detection and a court calibration.</p>;
  }

  return (
    <svg width={SVG_WIDTH} height={SVG_HEIGHT} style={{ background: "#0f1720", borderRadius: 8 }}>
      {/* doubles sidelines + baselines */}
      <Line x1={0} y1={0} x2={COURT_WIDTH_M} y2={0} />
      <Line x1={0} y1={COURT_LENGTH_M} x2={COURT_WIDTH_M} y2={COURT_LENGTH_M} />
      <Line x1={0} y1={0} x2={0} y2={COURT_LENGTH_M} />
      <Line x1={COURT_WIDTH_M} y1={0} x2={COURT_WIDTH_M} y2={COURT_LENGTH_M} />
      {/* singles sidelines */}
      <Line x1={SINGLES_INSET_M} y1={0} x2={SINGLES_INSET_M} y2={COURT_LENGTH_M} />
      <Line x1={COURT_WIDTH_M - SINGLES_INSET_M} y1={0} x2={COURT_WIDTH_M - SINGLES_INSET_M} y2={COURT_LENGTH_M} />
      {/* service lines + center service line */}
      <Line x1={SINGLES_INSET_M} y1={SERVICE_LINE_M} x2={COURT_WIDTH_M - SINGLES_INSET_M} y2={SERVICE_LINE_M} />
      <Line
        x1={SINGLES_INSET_M}
        y1={COURT_LENGTH_M - SERVICE_LINE_M}
        x2={COURT_WIDTH_M - SINGLES_INSET_M}
        y2={COURT_LENGTH_M - SERVICE_LINE_M}
      />
      <Line x1={COURT_WIDTH_M / 2} y1={SERVICE_LINE_M} x2={COURT_WIDTH_M / 2} y2={COURT_LENGTH_M - SERVICE_LINE_M} />
      {/* net */}
      <Line x1={0} y1={NET_Y_M} x2={COURT_WIDTH_M} y2={NET_Y_M} />

      {locations.map(([x, y], index) => {
        const [px, py] = toSvg(x, y);
        return <circle key={index} cx={px} cy={py} r={5} fill="#ef6461" fillOpacity={0.75} stroke="#0f1720" />;
      })}
    </svg>
  );
}
