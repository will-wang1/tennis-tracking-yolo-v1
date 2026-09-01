"""Recover the ball's HEIGHT above the court, in metres.

Every bounce detector in this project before this one had to work with
pixel y, or with the court homography's ground projection, and both of
them conflate two completely different things: how high the ball is off the
ground, and how far down the court it is. That conflation is the single
reason bounce/contact/net-crossing could never be told apart reliably -
see `parabolic_bounce_detector.py`, whose physics are sound but whose
"vertical restitution" is only really vertical when the ball is close to
the camera.

The way out is that the ground-plane homography's error IS the height
signal. A homography maps the court plane, so it is exact for a ball ON the
ground and increasingly wrong for a ball above it: the camera ray through
an airborne ball hits the court somewhere BEYOND where the ball actually
is, pushed directly away from the camera. Measured on this project's
footage, a ball at the top of its arc projects several metres past the far
baseline - impossible as a position, but a direct measure of height.

Geometrically, with the camera at height `Cz` above the court and standing
over the ground point `nadir`, a ball at true ground position P and height
h projects to

    G = nadir + (P - nadir) * Cz / (Cz - h)

Written with w = 1 - h/Cz that rearranges to `w * (G - nadir) = P - nadir`,
and over a single free flight two more facts pin it down: P moves in a
straight line at constant speed, and h is a parabola whose curvature is
GRAVITY - not a free parameter. Because h is quadratic in time, so is w,
and its quadratic coefficient is exactly `g / (2 * Cz)`, a known constant.
That makes the whole reconstruction a plain linear least-squares fit for
six unknowns (the ground line's position and velocity, plus w's constant
and linear terms) - no iterative solver, no per-frame depth guess. See
`fit_height_curve`.

The camera pose this needs comes from `pose_from_upright_objects`, which
solves it from the players - people of known height standing on the court.
On this project's footage that agrees with an independent decomposition of
the homography to within 0.1m of camera height, so the pose is sound.

NOT CURRENTLY USED BY THE PIPELINE, and the reason is worth recording,
because the maths here is exact and it is tempting to assume it must work.
Only gravity fixes the height scale, and gravity's contribution to the fit
grows with the SQUARE of the window length: across 0.3s it moves `w` by
about 0.007, far less than the measurement noise, so the fit is free to
slide the ball anywhere along the camera ray. Measured against a synthetic
camera with a clean flight, recovering height to better than ~0.2m needs
BOTH of:

    - a window of >= 0.75s of uninterrupted flight (0.3s gave 10m errors), and
    - ground-projection noise <= ~0.05m.

The window length is what this footage fails on, and by a wide margin.
Rallies rarely leave 0.75s of flight between one impact and the next, and
roughly a fifth of frames have no ball detection at all, which breaks the
long stretches further. Run against the real clip, only 2 of 17 flight
segments produced physical heights.

RETESTED on better footage, since the diagnosis above says the fix is a
higher detection rate rather than better code, and the zverev clip has one:
93.7% of frames detected against video_input2's 83%, and 13 of its 33
flights clear the 0.75s bar rather than 6 of 24. Ten of those thirteen
return heights inside a physical range, with least-squares residuals of
0.001 to 0.006 - and they are still wrong. The check that shows it is to
ask the height at an instant where it is already known: a flight that ends
in a detected bounce must reach the ground there. It does not. Measured at
eight such bounces on the zverev clip the reconstruction puts the ball
between 0.2m and 3.8m up at the moment it lands, mean error 2.2m; on the US
Open clip, four bounces, mean error 2.9m, one of them 2.1m BELOW the court.

That is the predicted failure rather than a new one - the fit slides the
ball along the camera ray and reports an excellent residual for a wrong
answer - but it is worth recording how convincing the wrong answer looks.
Neither a plausible height range nor a tiny residual is evidence of
anything here; only agreement with a height known independently is, and
the bounce is where such a height is free. Any future attempt should be
scored that way from the start.

Measurement precision is close to sufficient, which is worth recording
because it is the intuitive suspect and it is not the culprit. Measured on
this footage: the ball's frame-to-frame jitter is under a pixel per axis,
and one pixel at the far baseline is 0.066m of court (0.028m at the near
baseline), so ground-projection noise is around 0.06m - near the 0.05m
tolerance rather than orders off it. Far-court detections are no noisier
than near-court ones (2.94px vs 3.14px); they simply dominate by volume,
being 80% of all detections from a camera behind the near baseline.

So this is kept for its verified geometry and its camera-pose solver, not
as a working bounce signal. Making it work is mostly about getting longer
uninterrupted flights - a higher detection rate, so fewer dropouts break a
flight up - rather than about better fitting code.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

GRAVITY = 9.81  # m/s^2

@dataclass(frozen=True)
class CameraPose:
    """Where the camera is, in court coordinates (metres, the frame
    `court_calibration.FULL_COURT_REFERENCE_POINTS` defines)."""

    nadir: tuple[float, float]  # ground point directly below the camera
    height: float  # metres above the court plane


@dataclass(frozen=True)
class HeightCurve:
    """One reconstructed free flight: where the ball was, in three
    dimensions, over the window it was fitted to.

    Stored as the fitted `w = 1 - h/Cz` (quadratic in time about `t0`)
    rather than as height directly, because that is the form the linear
    fit produces; `height` converts on demand.
    """

    a: float  # w's constant term
    b: float  # w's linear term, 1/s
    c: float  # w's quadratic term = g / (2 * camera_height), fixed by gravity
    t0: float  # seconds; the instant the fit is written about
    camera_height: float
    residual: float  # scale-normalized least-squares residual, unitless

    def height(self, t: float) -> float:
        """Metres above the court at time `t` (seconds)."""
        s = t - self.t0
        return self.camera_height * (1.0 - (self.a + self.b * s + self.c * s * s))

    def vertical_velocity(self, t: float) -> float:
        """Metres per second, positive upward - a real vertical speed, not a
        screen-space one, so it can be compared against gravity and against
        the other side of an impact on equal terms."""
        s = t - self.t0
        return -self.camera_height * (self.b + 2.0 * self.c * s)


def fit_height_curve(
    times: Sequence[float],
    ground_points: Sequence[tuple[float, float]],
    pose: CameraPose,
    t0: Optional[float] = None,
) -> Optional[HeightCurve]:
    """Reconstruct one free flight from where its samples PROJECT onto the
    court (`ground_points`, what `CourtCalibration.pixel_to_world` returns
    for each ball detection) at `times` in seconds.

    Returns None if there are too few samples to over-determine the fit:
    six unknowns against two equations per sample means four samples is the
    smallest window that constrains rather than merely interpolates.

    `ground_points` must all come from ONE uninterrupted flight - the fit
    assumes a single straight ground path and a single gravity parabola, so
    a window spanning a bounce or a racket contact describes no real
    trajectory and shows up as a large `residual`.
    """
    if len(times) < 4 or pose.height <= 0.0:
        return None

    t0 = float(times[0]) if t0 is None else float(t0)
    c = GRAVITY / (2.0 * pose.height)
    nadir = np.asarray(pose.nadir, dtype=np.float64)

    rows: list[list[float]] = []
    rhs: list[float] = []
    offsets = []
    for t, point in zip(times, ground_points):
        s = float(t) - t0
        offset = np.asarray(point, dtype=np.float64) - nadir
        offsets.append(offset)
        # w(s) * offset == (P - nadir), which is linear in s. Unknowns are
        # w's a and b, and the ground line's position and velocity per axis.
        rows.append([offset[0], s * offset[0], -1.0, -s, 0.0, 0.0])
        rhs.append(-c * s * s * offset[0])
        rows.append([offset[1], s * offset[1], 0.0, 0.0, -1.0, -s])
        rhs.append(-c * s * s * offset[1])

    design = np.array(rows, dtype=np.float64)
    target = np.array(rhs, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)

    # Normalized by how far the samples sit from the nadir, so a residual
    # from the far court and one from right under the camera are comparable
    # - distant samples have large offsets and would otherwise dominate.
    scale = float(np.mean([np.linalg.norm(o) for o in offsets])) or 1.0
    residual = float(np.sqrt(np.mean((design @ solution - target) ** 2))) / scale

    return HeightCurve(
        a=float(solution[0]),
        b=float(solution[1]),
        c=c,
        t0=t0,
        camera_height=pose.height,
        residual=residual,
    )


def initial_pose_from_homography(
    homography: np.ndarray,
    image_width: int,
    image_height: int,
    focal_lengths: Optional[Sequence[float]] = None,
) -> CameraPose:
    """A first estimate of the camera pose, by decomposing the court
    homography under an assumed pinhole model.

    The focal length is unknown, so it is scanned: for the correct one the
    recovered rotation columns come out orthogonal and equal in length, and
    both measures bottom out together, which makes the scan self-checking.
    The principal point is assumed to be the image centre - a good
    assumption for essentially any real camera and a far smaller error
    source than the focal length.

    Useful as an independent cross-check on `pose_from_upright_objects`
    (the two agree to 0.1m of camera height on this project's footage), and
    as a fallback when no player boxes are available.
    """
    ground_to_image = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    if focal_lengths is None:
        focal_lengths = np.arange(400.0, 8000.0, 25.0)

    best_score = np.inf
    best_pose = None
    for focal in focal_lengths:
        intrinsics = np.array(
            [[focal, 0.0, image_width / 2.0], [0.0, focal, image_height / 2.0], [0.0, 0.0, 1.0]]
        )
        mapped = np.linalg.inv(intrinsics) @ ground_to_image
        norm1, norm2 = np.linalg.norm(mapped[:, 0]), np.linalg.norm(mapped[:, 1])
        if norm1 <= 0 or norm2 <= 0:
            continue
        scale = 2.0 / (norm1 + norm2)
        r1, r2, translation = scale * mapped[:, 0], scale * mapped[:, 1], scale * mapped[:, 2]

        score = abs(float(np.dot(r1, r2))) + abs(float(np.linalg.norm(r1) - np.linalg.norm(r2)))
        if score >= best_score:
            continue

        rotation = np.column_stack([r1, r2, np.cross(r1, r2)])
        u, _, vt = np.linalg.svd(rotation)
        center = -(u @ vt).T @ translation
        best_score = score
        best_pose = CameraPose(nadir=(float(center[0]), float(center[1])), height=abs(float(center[2])))

    if best_pose is None:
        raise ValueError("Could not decompose the court homography into a camera pose")
    return best_pose


def pose_from_upright_objects(
    observations: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    object_height: float = 1.75,
    min_projected_length: float = 0.5,
    max_line_distance: float = 4.0,
) -> Optional[CameraPose]:
    """Solve the camera pose from things of known height standing on the
    court - in practice the players, whose feet are on the ground and whose
    heads are about `object_height` above it.

    Each observation is `(base, top)`: where the object's ground contact
    point and its top project onto the court through the homography. The
    base is on the court, so its projection is exact; the top is elevated,
    so its projection is pushed directly away from the camera. That gives
    two independent things:

    - The nadir must lie on the line through both points, for every
      observation. Many observations at different court positions therefore
      pin it down as the common intersection - a plain linear least-squares
      solve, no search involved.
    - The ratio of the two points' distances from the nadir is
      `Cz / (Cz - object_height)`, which inverts to give the camera height
      directly. Taken as a median over observations, so a mis-detected box
      moves nothing.

    Calibrating from the ball's own flights instead was tried and does not
    work: a camera infinitely far away is orthographic, and under an
    orthographic camera every trajectory fits equally well and all height
    information disappears, so that cost has a degenerate direction it runs
    away along - measured, it drifted the nadir 100m off court. A known
    height standing on the court removes the degeneracy outright.

    Returns None if the observations don't constrain a pose - too few, or
    all pointing the same way, which leaves the intersection undetermined.
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    usable: list[tuple[np.ndarray, np.ndarray]] = []
    for base, top in observations:
        base_point = np.asarray(base, dtype=np.float64)
        top_point = np.asarray(top, dtype=np.float64)
        direction = top_point - base_point
        length = float(np.linalg.norm(direction))
        if length < min_projected_length:
            continue  # too short to define a reliable direction
        direction = direction / length
        # The nadir is collinear with base and top: cross product is zero.
        rows.append([direction[1], -direction[0]])
        rhs.append(float(direction[1] * base_point[0] - direction[0] * base_point[1]))
        usable.append((base_point, top_point))

    if len(usable) < 8:
        return None

    design = np.array(rows, dtype=np.float64)
    target = np.array(rhs, dtype=np.float64)
    if np.linalg.matrix_rank(design, tol=1e-6) < 2:
        return None  # every sighting points the same way; no intersection
    nadir, *_ = np.linalg.lstsq(design, target, rcond=None)

    # Drop sightings whose line misses that intersection badly - a bad box,
    # or a player who was mid-jump and whose "feet" weren't on the ground.
    kept = [
        (base, top)
        for (base, top), row, value in zip(usable, rows, rhs)
        if abs(float(np.dot(row, nadir)) - value) <= max_line_distance
    ]
    if len(kept) >= 8:
        design = np.array([[(t - b)[1], -(t - b)[0]] for b, t in kept]) / np.array(
            [[float(np.linalg.norm(t - b))] for b, t in kept]
        )
        target = np.array([float(r[0] * b[0] + r[1] * b[1]) for r, (b, _) in zip(design, kept)])
        nadir, *_ = np.linalg.lstsq(design, target, rcond=None)
        usable = kept

    heights = []
    for base_point, top_point in usable:
        base_distance = float(np.linalg.norm(base_point - nadir))
        top_distance = float(np.linalg.norm(top_point - nadir))
        if base_distance <= 0 or top_distance <= base_distance:
            continue  # the top must project FURTHER from the camera than the base
        ratio = top_distance / base_distance
        heights.append(object_height * ratio / (ratio - 1.0))
    if not heights:
        return None

    return CameraPose(nadir=(float(nadir[0]), float(nadir[1])), height=float(np.median(heights)))


def upright_observations_from_boxes(
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]],
    calibrations_by_frame: dict[int, "object"],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Turn detected person boxes into `(base, top)` court projections for
    `pose_from_upright_objects`, using the bottom-centre of each box as the
    feet and the top-centre as the head."""
    observations = []
    for frame_idx, boxes in boxes_by_frame.items():
        calibration = calibrations_by_frame.get(frame_idx)
        if calibration is None:
            continue
        for x1, y1, x2, y2 in boxes:
            center_x = (x1 + x2) / 2.0
            observations.append(
                (calibration.pixel_to_world(center_x, y2), calibration.pixel_to_world(center_x, y1))
            )
    return observations
