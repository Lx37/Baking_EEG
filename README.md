
# 🧁 Baking_EEG

**Your Recipe for EEG Insights!**

Baking_EEG is a comprehensive, modular, and open-source Python toolkit for the analysis of EEG signals, with a special focus on evoked potential protocols. Designed for both research and clinical applications, it supports a wide range of acquisition systems and provides robust pipelines for preprocessing, decoding, statistics, and visualization.

---

## Key Features

- **Multi-protocol support:** Analyze EEG data from various evoked potential paradigms 
- **Multi-system compatibility:** Works with BrainAmp, EGI, Micromed, and more
- **Flexible pipelines:** Individual and group-level analyses, including ERP, temporal decoding and temporal generalization matrix
- **Statistical analysis:** Intra- and inter-subject statistics, permutation tests, FDR, cluster-based correction
- **Rich visualization:** Automated dashboards and publication-ready plots
- **Reproducibility:** Configurable, version-controlled, and ready for cluster computing (SLURM/submitit)
- **Extensible:** Modular codebase for easy adaptation to new protocols or analysis needs

---

## 📂 Project Structure

```
Baking_EEG/
├── Baking_EEG/                # Core analysis modules (preprocessing, decoding, stats, etc.)
├── base/                      # Base decoding and pipeline utilities
├── config/                    # Configuration files (protocols, classifiers, etc.)
├── examples/                  # Example scripts and analysis workflows
├── results/                   # Output results (organized by protocol, subject, etc.)
├── submitit/                  # SLURM/submitit job submission scripts
├── utils/                     # Utility functions (visualization, loading, etc.)
├── requirements.txt           # Main dependencies
├── README.md                  # This file
└── ...
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher recommended
- See `requirements.txt` for all dependencies

### Quick Start

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Lx37/Baking_EEG.git
    cd Baking_EEG
    ```
2. **(Recommended) Create a virtual environment:**
    ```bash
    python3 -m venv bakingeeg_env
    source bakingeeg_env/bin/activate
    ```
   Or use [uv](https://github.com/astral-sh/uv) for faster installs:
    ```bash
    uv venv bakingeeg_env --python 3.12
    source bakingeeg_env/bin/activate
    uv pip install -r requirements.txt
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Prepare your EEG data
- Organize your raw/preprocessed EEG files according to your acquisition system and protocol.
- Update or check the configuration files in `config/` as needed.

### 2. Run an analysis
- **Single subject decoding:**
    ```bash
    python examples/run_decoding_one_lg.py --subject_id <SUBJECT_ID>
    ```
- **Group analysis (SLURM/submitit):**
    ```bash
    python submitit/submit_1group_lg_all.py
    ```
- **Custom analysis:**
    Explore scripts in `examples/` or build your own using the modular functions.

### 3. Review results
- Results (metrics, plots, logs) are saved in the results directory, organized by protocol and subject.
- Use the visualization utilities in `utils/` for further exploration.

---

## 🧩 Extending Baking_EEG
- Add new protocols by editing or adding config files in `config/`
- Implement new analysis pipelines in Baking_EEG or `base/`
- Contribute new visualization or statistics modules in `utils/`

---

## 🤝 Contributing

Contributions are welcome! Please:
- Fork the repository and create a feature branch
- Submit pull requests with clear descriptions
- Report bugs or request features via GitHub Issues

---

## 📚 Documentation


- Examples and scripts are available in the `examples/` folder. 

---

## 📝 License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.

---

## Contributors

- CNRS, Alexandra Corneyllie, Tom Balay, and all contributors
- Inspired by the open-source neuroscience and Python communities

---

<div align="center">
  <em>Baking_EEG: Turning raw EEG into scientific delicacies!</em>
</div>

---

