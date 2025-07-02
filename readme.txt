# Baking_EEG

Your Recipe for EEG Insights!
and prepping your EEG for Scientific Gourmet.

Baking_EEG is a project designed to analyze EEG signals from different types 
of evoked potential protocols. This project supports multiple acquisition 
systems, including BrainAmp, EGI, and Micromed. It allows for analyses at 
different levels, including individual and group analyses.


## Features

- Analysis of EEG signals from different evoked potential protocols
- Compatibility with multiple acquisition systems (BrainAmp, EGI, Micromed)
- Individual and group analyses

## Prerequisites

- Python 3.x
- Required Python libraries (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/BatteryEEG.git
    ```
2. Navigate to the project directory:
    ```bash
    cd BatteryEEG
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    We recomand to create a specific python environment with **uv**
        1. On macOS and Linux. Open a terminal and do
           `$ curl -LsSf https://astral.sh/uv/install.sh | sh`
        1. On windows. Open a terminal using **CMD** in the windows menu and do
         `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
        2. Exit the terminal and open it again.
        3. Download with right click and save this file corresponding in "Documents" folder:
        * [`requirements.txt`](https://github.com/Lx37/Baking_EEG/blob/main/requirements.txt)
        4. open terminal or **CMD**
        5. `uv venv backingEEG --python 3.12`
        6. For Mac/Linux `source env_formation/bin/activate` (you should have `(backingEEG)` 
        in your terminal) or for Powershell `backingEEG\Scripts\activate`
        7. `uv pip install -r Documents/requirements.txt`

## Usage

1. Prepare your EEG data following the instructions specific to your 
   acquisition system.
2. Run the analysis script:
    ```bash
    python analyse_eeg.py --input path/to/your/data --output path/to/results
    ```
3. Review the results in the specified output directory.

## Contributing

Contributions are welcome! Please submit pull requests and report issues via 
the issue tracker on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for 
details.
