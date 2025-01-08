# UK Biobank Generalizable Brain Clocks

UK Biobank MRI Data Can Power the Development of Generalizable Brain Clocks: A Study of Standard ML/DL Methodologies and Performance Analysis on External Databases

## Abstract

In this study, we present a comprehensive pipeline to train and compare a broad spectrum of machine learning and deep learning brain clocks, integrating diverse preprocessing strategies and correction terms. Our analysis also includes established methodologies which have shown success in prior UK Biobank-related studies. For our analysis we used T1-weighted MRI scans and processed de novo all images via FastSurfer, transforming them into a conformed space for deep learning and extracting image-derived phenotypes for our machine learning approaches. We rigorously evaluated these approaches both as robust age predictors for healthy individuals and as potential biomarkers for various neurodegenerative conditions, leveraging data from the UK Biobank, ADNI, and NACC datasets. To this end we designed a statistical framework to assess age prediction performance, the robustness of the prediction across cohort variability (database, machine type and ethnicity) and its potential as a biomarker for neurodegenerative conditions. Results demonstrate that highly accurate brain age models, typically utilising penalised linear machine learning models adjusted with Zhang's methodology, with mean absolute errors under 1 year in external validation, can be achieved while maintaining consistent prediction performance across different age brackets and subgroups (e.g., ethnicity and MRI machine/manufacturer). Additionally, these models show strong potential as biomarkers for neurodegenerative conditions, such as dementia, where brain age prediction achieved an AUROC of up to 0.90 in distinguishing healthy individuals from those with dementia.

---

## Repository Structure

This repository contains the implementation and supporting materials for the methodologies described in the paper. The folder structure corresponds to the primary approaches and their comparisons:

1. **[ML-approach](./ML-approach)**  
   This folder provides materials for the machine learning methodologies discussed in Section **3.2.1 Machine Learning Approaches** of the paper. It includes feature selection processes, preprocessing pipelines, and model training scripts.

2. **[DL-approach](./DL-approach)**  
   This folder contains all resources related to the deep learning methodologies described in Section **3.2.2 Deep Learning Approaches** of the paper, including model architectures and training scripts.
 
3. **[Comparison](./comparison)**  
   This folder focuses on the comparative analysis of the results. More specifically, there is the statistical analysis, as detailed in Section **3.5. Statistical analysis of model performance** of the paper, and the process for the selection of the best models, as detailed in Section **3.6. Model selection based on stepwise pareto analysis** of the paper. It includes evaluation scripts, results visualization, and summary tables.

---

## Acknowledgements

- **Data Source**: This research has been conducted using the UK Biobank Resource under application number 88003.  
- **Rights**: All rights reserved by [Oxcitas Ltd.](https://www.oxcitas.com). 

For further inquiries or collaborations, feel free to contact us.

---

## Citation

If you use this work in your research, please cite our paper:
