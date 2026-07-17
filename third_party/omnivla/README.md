Vendored OmniVLA inference code
===============================

This directory contains the minimal OmniVLA inference modules needed by the
FrodoBots rides_11 fine-tuning and analysis scripts in this repository.

Source:
- Repository: https://github.com/NHirose/OmniVLA.git
- Local source revision checked during vendoring: 5182600

Only lightweight source files are vendored here. Model checkpoints, datasets,
training logs, generated maps, and other large artifacts should stay outside
Git and be referenced through config paths.
