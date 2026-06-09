# salticid-dynamicfocus-dfdd

Implementation of **Depth-from-Defocus (DfDD)** within a virtual *Habronattus pyrrithrix* jumping spider visual system, as described in *Dynamic focusing through retinal movements in Habronattus pyrrithrix jumping spiders*.

---

## Overview

This repository implements a modified version of the **FocalSplit DfDD algorithm** (https://arxiv.org/abs/2504.11202), adapted from a camera-based framework to simulate the optical properties of the jumping spider *Habronattus pyrrithrix*.

Rather than operating on a conventional imaging system, this implementation emulates, the **focal properties of the spider’s principal eyes**, **Retinal tier structure (Tier I & Tier II)** and **viewing-angle-dependent optical effects**


### Given an input image and defined visual parameters, `src/FocalSplit_forHpyrrithrix.py` will:

1. Simulate how a jumping spider views a scene across different **viewing angles**
2. Determine **viewing distance** based on where gaze intersects the ground plane
3. Generate images formed at the focusing planes associated with Tier I and Tier II of the retina
4. Estimate depth using a modified version of:
   - **FocalSplit**
   - **FocalTrack** (optional)


### Features
The code can be adapted to meet different visual systems. The in-axis focusing distance of the lens can be defined through the `focusingDistance` variable. If the visual system exhibits field-based optical aberrations that are not corrected through morphological means, we recommend quantifying the changes in focus and describing them in the `normalized_field_curvature.csv` file.

The `FocalSplit_forHpyrrithrix.py` script will produce direct plots of Mean Aggregate Depth Errors, depicting the errors associated with each viewing angle (and thus viewing distance). Additionally, csv outputs are generated in reports/MAE_outputs which can be interpreted and visualized using the provided `dfd_error_plotter.R` file.
