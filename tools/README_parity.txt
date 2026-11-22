parity_lung.js + planner_page.js: the breathing-TTS page planner (breath_page.html 2026-11-17, plan/effWords/simulate) run in Node on
the cases written by a Python loop over texts x voices x settings; kipp.lung gave identical breaths, classes, sizes and times (336 cases).
Since Kipp 1.3 the planner sizes breaths from fitted models by default (sizing="data"); the parity cases use voice_params(..., sizing="rule"),
which is the page planner unchanged.
