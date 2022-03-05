# QSRR_RT_prediction

## Before execute the code:
 - Decompress multivolume compressed file 'Datasets.7z.001'. Rename resulting directory to 'Datasets'.
 - Decompress compressed file 'RAW_data.7z'.



#### 1. Datasets creation and preprocessing
Uses functionality found in 'Preprocess_datasets' class:
 - Reads the xlsx and csv files, describing each dataset, present in 'Datasets' directory
 - Using RDKit calculates the Molecular Descriptors for all compounds in datasets
 - Applies filtering processes to produce the 4 configurations of each dataset
 - Creates separate directories/subdirectories for each dataset/confoguration with corresponding dependent and independent variables files

#### 2. Datasets analysis by selected Machine Learning regression algorithms
Uses functionality found in 'Dataset_ML_analysis' class. Εach class object corresponds to a specific dataset and has functionality to:
 - Perfom analysis of specific dataset configuration/metric using all or part of available algorithms
 - Store analysis results in 'RAW_data' directory, creating all the necessary subfolders corresponding to each dataset/configuration
 - Create summary for all the analysis iterations
 
#### 3. Collection of RAW experimental data for all datasets
Uses functionality found in 'RAW_data' class:
 - Reads all available results files (analysis iterations) for each dataset/configuration
 - Creates a separate excel file for each dataset, containing, for each configuration and each iteration of analysis, the predictions made by each algorithm as well as calculates the values for all available metrics for each prediction.
 - RAW data files are stored in 'Evaluation_Results/Collect_RAW_data' directory

#### 4. Analysis of experimental data
Uses functionality found in 'Data_analysis' class:
 - Computes the best model (algorithm) and the best predictions (iteration) for all datasets
     - Stores results in the 'Evaluation_Results/Best_models_predictions' directory
 - Computes the regression error (algorithms' performance) for all datasets together, for each combination of configuration and metric used
     - Stores results in the 'Evaluation_Results/Regression_errors' directory
 - Applies various statistical methods on algorithms' performance values for all datasets' configurations on each available combination of dataset-algorithm-metric
     - Stores results (files and plots) in the 'Evaluation_Results/Statistical_tests' directory
 - Creates various types of Regression and Residuals plots to evaluate performance for the best algorithm found for each dataset for all combinations of metric-datasets' configurations.
     - Stores results (files and plots) in the 'Evaluation_Results/Regression_Residuals_plots' directory
 - Creates Performance plots, comparing the performance of each algorithm on all datasets for all combinations of metric-datasets' configurations.
     - Stores results (files and plots) in the 'Evaluation_Results/Models_performance_(matrices_plots)' directory
