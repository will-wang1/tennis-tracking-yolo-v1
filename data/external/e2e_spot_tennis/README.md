Stroke-labelled tennis points from E2E-Spot (Hong et al., "Spotting Temporally
Precise, Fine-Grained Events in Video", ECCV 2022; github.com/jhong93/spot at
edec4201, `data/tennis/test.json`, BSD-3 - see LICENSE).

`stroke_clips.json` keeps only the points whose events carry a stroke type in
`comment` (forehand_topspin, backhand_topspin, forehand_slice, backhand_slice,
forehand_volley, backhand_volley, overhead, serve): 9 matches, ~9,800 strokes.
Each point is `<match>_<start frame>_<end frame>` in the full-match YouTube
video listed in `videos.csv`; event frames are relative to the start frame,
at the fps in the entry. The videos are not included - see
docs/gpu_stroke_training.html for downloading them.
