# Figure maker: still images of a session's first far cloud + EgoArt prediction

    python3 tools/spot_webapp/figures/serve_figures.py /Users/andyye/dev/egoart-recordings/final
    open http://127.0.0.1:8771

Indexes every `<root>/<name>/*_session/run/*_farpred.data.json` (the first far prediction of each saved web-app session).
The page (three.js, same loader as the web app) shows the cloud in the camera frame with the prediction as real geometry:
trajectory tube with an arrow, contact sphere, hinge point + hinge axis (or slide axis with an arrow for drawers), handle
mask points recoloured, the camera frustum, optionally the photo in it. Orbit to the view you want, set the style (background
white / dark / transparent, point size in mm, tube radii, colours, field of view, depth cut-off) and press **Save PNG + camera**:
renders off-screen at the chosen resolution (default 3840x2160) and writes `<root>/<name>/<name>_far.png` plus
`<name>_camera.json` (camera + style), which is reloaded automatically next time so the figure can be re-rendered identically.
**copy to all sessions** writes the current camera + style to every session for a consistent panel. Points have a world
size, so their apparent size is the same on screen and in the export.
