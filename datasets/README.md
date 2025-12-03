# Datasets Folder

Place the dataset file(s) here before running `main.jl`.

## Required Dataset

- **Fraudulent_E-Commerce_Transaction_Data_merge.csv**: E-Commerce Fraud Detection dataset
  - Expected columns: `Transaction ID`, `Customer ID`, `Transaction Date`, `Transaction Amount`, 
    `Account Age Days`, `Is Fraudulent`, etc.

## Dataset Source

The dataset should be a publicly available fraud detection dataset appropriate for 
multiclass classification (or binary with feature engineering for multiclass).

## Note

CSV files in this folder are excluded from git (see `.gitignore`) to avoid 
committing large data files to the repository.
