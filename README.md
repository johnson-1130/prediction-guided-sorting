# Prediction-Guided Sorting through Inversions

Experiment code for my MSc dissertation (School of Computer Science, University of Sheffield, 2026).

## Overview
This project compares two prediction models for adaptive sorting, positional
predictions and dirty comparisons, both empirically and theoretically. The
inversion ratio is used as the common error measure, with Insertion Sort as the
adaptive sorting algorithm.

## Key Results
- Proved that the expected out-degree is an affine function of the flip
  probability and the true rank.
- Observed the phase-transition behavior of dirty comparisons at 50% flip
  probability.

## Repository Structure
- `positional_prediction.py`  Positional prediction model
- `dirty_comparison.py`       Dirty comparison model
- `insertion_sort/`           Insertion Sort using predictions (comparisons and swaps)
- `scale/`                    Inversion ratio as the input size scales
- `single_run/`               Single-run experiments
- `average/`                  Results averaged over repeated runs

Each experiment folder contains its script and the generated figures.

## Running the Experiments
Requires Python 3. Run a script from the repository root, for example:

    python3 insertion_sort/insertion_sort.py

## Author
Hao-Hsiang Chang (張皓翔)
