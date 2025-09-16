# HistoCloud Command Line (Hiper Gator) Branch

This branch provides instructions and scripts to run **HistoCloud** via the command line on the Hiper Gator HPC cluster.

## Purpose

- Enable command-line execution of HistoCloud workflows.
- Support batch processing and automation on Hiper Gator.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Histo-cloudTN
    ```

2. **Load required modules and dependencies**  
    - Requires container `.sif` for env
    - Or conda env can be made a specified


3. **Run HistoCloud:**
    ```bash
    sbatch run.sh
    ```

## Notes

- Update `run.sh` with your input data and parameters.
- For troubleshooting or custom configurations

## Contact
For questions, open an issue or contact the repository maintainers.