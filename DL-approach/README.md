# Brain-Age prediction using t1-MRI dataset

A simple Jupyter Notebook that demonstrates the core of the deep learning approach of the paper, focusing on the regression branch.

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook
- Required Python libraries: `numpy`, `matplotlib`, `tensorflow`, `scikit-learn`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Euskal-Oxcitas-Biotek/UKB-gen-clocks.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DL-approach
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter-notebook
   ```
2. Create a folder named 'data' and add your data there.
   ```bash
   mkdir data
   ```
   Note: Make sure that the data added matches the configuration files.
3. Navigate to the notebook file (e.g., `demo.ipynb`) and open it.
4. Run the cells step-by-step to explore the examples and outputs.

## Folder Structure

```
DL-apporach/
├──notebooks
	├──demo.ipynb 			    	# Main Jupyter Notebook
├── data/                   			# Folder for datasets
├── src/                 			# Folder for source code
├── train/                   			# Folder for training files
├── configs/                 			# Folder for configuration files
├── README.md               		    	# Project documentation
└── requirements.txt        		   	# List of required Python packages
```

## Disclaimer

The file **[demo.ipynb](./demo.ipynb)** is provided as a demonstration of the methodology. It is executed with limited data and a low number of training epochs for illustrative purposes only. The results may not reflect the performance of the models described in the paper.

If you want to perform real training:
1. Add your own dataset to a **`data/`** folder in the repository.
2. Adapt the number of epochs and other training parameters as needed.
3. Use the script **[train/train.py](./train/train.py)** for full-scale training.

## Contributing

Feel free to fork this project and submit pull requests. For significant changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This research has been conducted using the UK Biobank Resource under application number 88003.

