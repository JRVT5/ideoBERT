# ideoBERT: A Fine-Tuned BERT-Based Approach to News Ideology Classification

This repository contains the code and resources for ideoBERT, a fine-tuned BERT-based model designed for the classification of news articles based on the political leaning of their sources. This project is the final submission for the Natural Language Processing course (CS 457) at Middlebury College.

## Overview

ideoBERT leverages the power of the BERT (Bidirectional Encoder Representations from Transformers) language model to predict the ideological stance of news articles, whether they lean conservative, liberal, or neutral. By fine-tuning BERT on a carefully curated dataset, we aim to achieve high accuracy in identifying the ideological positions of different news sources.

## Contents

The main model is the model.py. The hugging face model is an alternative model to the one used in the paper.

model.sbatch: Sbatch file to train/eval the model on Slurm (Simple Linux Utility for Resource Management) workload manager.  
model.py: Python file containing the default model using autotrainer BERT model.  
train.py: Training function for model.py. Trained through the model.sbatch file  
util.py: Utility functions for training/evaluating the model.  
hugging_model.sbatch: Alternative sbatch file to run the hugging_model.py model on Slurm (Simple Linux Utility for Resource Management) workload manager.  
hugging_model.py: Alternative python file containing hugging face model hyperparameters and data loading functions. Defaults to train and eval mode when running sbatch.  
data/: Data files for training, evaluating and testing the model  
ideoBERT.pdf: The final paper detailing the findings, methodology, and results of the ideoBERT project.  
README.md: This file.  

##Installation

To run ideoBERT, you'll need to have Python 3.x and the following libraries installed:

transformers  
torch  
numpy  
pandas  
argparse  
tqdm  
itertools  
typing  
datasets  

You can install these dependencies using the following command:

```bash
pip install transformers torch numpy pandas argparse tqdm itertools typing datasets
```
## Usage

Clone the repository:

```bash
git clone https://github.com/JRVT5/ideoBERT.git  
cd ideoBERT
```

The dataset should be placed in the data/ folder, and the model is found in model.py in the main folder.

## Data

The dataset used for fine-tuning ideoBERT consists of news article titles labeled by the political ideology of their sources. The dataset has been processed into training, evaluation and testing. It was developed from the NewB data set from Jerry Wei. The original dataset can be found [here](https://github.com/JerryWeiAI/NewB).

## Final Paper and Results

The final paper, ideoBERT.pdf, provides an in-depth analysis of the methodology, experiments, results, and conclusions of this project. It is recommended to read the paper for a comprehensive understanding of the approach and findings.

### Contributing

Contributions are welcome! If you'd like to improve or extend the ideoBERT project, please submit a pull request with your changes or open an issue for discussion.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.

### Acknowledgments

Jerry Wei - [NewB data set](https://github.com/JerryWeiAI/NewB)  
Middlebury College CS 457 - Natural Language Processing: For guiding the study of NLP techniques and providing computer resources.  
The creators of the BERT model and the Hugging Face library for making NLP tools accessible to everyone.  
