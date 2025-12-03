# ============================================================================
#          FRAUD DETECTION - MULTICLASS CLASSIFICATION
#          Main Script - Executes All Experiments
#          Group Project - Machine Learning I
# ============================================================================
#
# Objective: E-commerce fraud detection using multiclass classification
#            (Legitimate, Suspicious, Fraudulent)
#
# Methodology: Three different approaches as required by the assignment:
#
#   APPROACH 1: Real Dataset with 80/20 Split
#       - Standard train/test split: 80% train, 20% test
#       - All features used
#
#   APPROACH 2: Real Dataset with 50/50 Split  
#       - Balanced train/test split: 50% train, 50% test
#       - Tests generalization with less training data
#
#   APPROACH 3: PCA Dimensionality Reduction with 80/20 Split
#       - Principal Component Analysis applied
#       - Reduces features to optimal number of components
#       - Tests if dimensionality reduction improves performance
#
# For each approach:
#   - 4 ML Algorithms tested:
#       * ANNs: 8 topologies (1-2 hidden layers)
#       * SVMs: 8 configurations (different kernels and C values)  
#       * Decision Trees: 6 maximum depths
#       * kNN: 6 different k values
#   - Ensemble Methods: Majority Voting + Weighted Voting
#   - Cross-Validation: 3-fold stratified on training set
#   - Final evaluation on hold-out test set
#
# Code Organization:
#   /project/
#   ├── main.jl                    ← This file (executable from top to bottom)
#   ├── /utils/
#   │   ├── utils.jl              ← Course utilities (includes modelCrossValidation)
#   │   └── preprocessing.jl      ← Custom preprocessing functions
#   └── /datasets/
#       └── Fraudulent_E-Commerce_Transaction_Data_merge.csv
#
# ============================================================================

# Set random seed for reproducibility (REQUIRED by assignment)
using Random
Random.seed!(42)

# ============================================================================
#                    PACKAGE IMPORTS
# ============================================================================

using CSV
using DataFrames
using Statistics
using Dates
using StatsBase
using LinearAlgebra  # For PCA

println("✅ Packages loaded!")

# Load course utilities (contains modelCrossValidation, confusionMatrix, etc.)
include("utils/utils.jl")
println("✅ Course utilities loaded (includes modelCrossValidation, confusionMatrix, etc.)")

# Load custom preprocessing utilities
include("utils/preprocessing.jl")
using .PreprocessingUtils: create_risk_classes, preprocess_multiclass
println("✅ Custom preprocessing utilities loaded!")

# ============================================================================
#              DATA LOADING & 3-CLASS TARGET CREATION
# ============================================================================
#
# Dataset: Fraudulent E-Commerce Transaction Data (1.5M transactions)
#
# Target Creation Strategy:
#   Original: Binary fraud labels (fraud vs non-fraud)
#   Our approach: 3-class risk assessment based on multiple signals:
#     1. Time Risk: Night transactions (0-5am, 11pm)
#     2. Amount Risk: High-value transactions (>90th percentile)
#     3. Account Age Risk: New accounts (<30 days)
#
# Class Mapping:
#   - Class 0 (LEGITTIMO): Low-risk, legitimate transactions
#   - Class 1 (SOSPETTO): Borderline cases requiring manual review
#   - Class 2 (FRAUDOLENTO): High-risk fraudulent transactions
#
# Justification: This approach allows for graduated risk assessment, enabling
# businesses to automatically approve low-risk transactions, flag suspicious
# cases for manual review, and immediately block high-risk frauds.
#
# ============================================================================

const DATA_PATH = "datasets/Fraudulent_E-Commerce_Transaction_Data_merge.csv"
println("\n" * "="^70)
println("📂 LOADING DATA")
println("="^70)

df = CSV.read(DATA_PATH, DataFrame)
target_col = "Is Fraudulent"

println("Original dataset size: $(size(df))")
println("Original fraud distribution:")
println("  Non-fraud: $(sum(df[!, target_col] .== 0))")
println("  Fraud:     $(sum(df[!, target_col] .== 1))")

# Create 3-class target using custom function
println("\n" * "="^70)
println("🎯 CREATING 3-CLASS TARGET")
println("="^70)

df_with_classes = create_risk_classes(df, target_col)

# ============================================================================
#          CLASS BALANCING & TRAIN/TEST SPLIT
# ============================================================================
#
# Challenge: Highly imbalanced dataset (90% Legitimate, 8.6% Suspicious, 1.4% Fraudulent)
#
# Solution: Undersample majority classes to match minority class (20,654 samples per class)
#
# Train/Test Split:
#   - 80% Training (49,569 samples) - used for cross-validation and model selection
#   - 20% Test (12,393 samples) - held out for final evaluation
#
# CRITICAL: Test set is NEVER used during training or model selection to prevent data leakage!
#
# ============================================================================

println("\n" * "="^70)
println("✅ TRAIN/TEST SPLIT (80% Train / 20% Test)")
println("="^70)

# Balance classes
class_0 = df_with_classes[df_with_classes.Risk_Class .== 0, :]
class_1 = df_with_classes[df_with_classes.Risk_Class .== 1, :]
class_2 = df_with_classes[df_with_classes.Risk_Class .== 2, :]

n_min = minimum([size(class_0, 1), size(class_1, 1), size(class_2, 1)])
n_target = min(n_min, 40000)

println("\n🔄 Balancing dataset...")
println("  Samples per class: $n_target")

class_0_sample = class_0[shuffle(1:size(class_0, 1))[1:n_target], :]
class_1_sample = class_1[shuffle(1:size(class_1, 1))[1:n_target], :]
class_2_sample = class_2[shuffle(1:size(class_2, 1))[1:n_target], :]

df_balanced = vcat(class_0_sample, class_1_sample, class_2_sample)
df_balanced = df_balanced[shuffle(1:size(df_balanced, 1)), :]

println("  Balanced dataset size: $(size(df_balanced))")

# Split Train/Test BEFORE preprocessing (critical to avoid data leakage!)
n_total = size(df_balanced, 1)
n_train = floor(Int, n_total * 0.80)
n_test = n_total - n_train

all_indices = shuffle(1:n_total)
train_indices = all_indices[1:n_train]
test_indices = all_indices[n_train+1:end]

df_train = df_balanced[train_indices, :]
df_test = df_balanced[test_indices, :]

println("\n📊 Split Summary:")
println("  Total samples:     $n_total")
println("  Training set:      $n_train (80%)")
println("  Test set:          $n_test (20%)")

# ============================================================================
#                    PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
#
# Steps:
#   1. Time Features: Extract hour, create night flag (hour < 6)
#   2. Feature Engineering:
#      - Amount_per_AccountAge: Transaction amount relative to account maturity
#      - High_Value_Flag: Transactions above 95th percentile
#      - New_Account_Flag: Accounts younger than 30 days
#   3. Missing Value Imputation: Median imputation
#   4. Feature Selection: Drop IDs, addresses, categorical features → 8 numerical features
#   5. Normalization: Min-Max [0,1] using training set parameters only
#
# Final Features (8):
#   - Transaction Amount
#   - Account Age Days
#   - Transaction_Hour
#   - Is_Night
#   - Amount_per_AccountAge
#   - High_Value_Flag
#   - New_Account_Flag
#   - (1 more from preprocessing)
#
# ============================================================================

println("\n🔧 Preprocessing train and test sets...")
df_train_processed = preprocess_multiclass(df_train, target_col)
df_test_processed = preprocess_multiclass(df_test, target_col)

input_cols = setdiff(names(df_train_processed), ["Risk_Class"])
train_inputs = Matrix{Float64}(df_train_processed[:, input_cols])
train_targets = Int.(df_train_processed.Risk_Class)

test_inputs = Matrix{Float64}(df_test_processed[:, input_cols])
test_targets = Int.(df_test_processed.Risk_Class)

println("\n📊 Preprocessed Data:")
println("  Features: $(length(input_cols))")
println("  Train samples: $(size(train_inputs, 1))")
println("  Test samples: $(size(test_inputs, 1))")
println("\n  Feature names: $input_cols")

# Create cross-validation indices (3-fold stratified)
k_folds = 3
cv_indices = crossvalidation(train_targets, k_folds)
println("\n✅ Cross-validation indices created ($k_folds folds, stratified)")

# ============================================================================
#        EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS (ANNs)
# ============================================================================
#
# Configuration:
#   - Topologies tested: 8 architectures (1-2 hidden layers as required)
#   - Activation: σ (sigmoid) for hidden layers, Softmax for output
#   - Optimizer: Adam (learning rate: 0.003)
#   - Loss: Cross-entropy
#   - Regularization: Early stopping (patience: 25 epochs)
#   - Validation: 10% of training set
#   - Executions: 1 per topology (can increase for stability)
#
# Architectures:
#   1. [256] - 1 hidden layer, Large
#   2. [128] - 1 hidden layer, Medium
#   3. [64] - 1 hidden layer, Small
#   4. [32] - 1 hidden layer, Tiny
#   5. [256, 128] - 2 hidden layers, Large
#   6. [128, 64] - 2 hidden layers, Medium
#   7. [64, 32] - 2 hidden layers, Small
#   8. [96, 48] - 2 hidden layers, Alternative
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS")
println("Testing 8 ANN Topologies (1-2 hidden layers)")
println("="^70)

topologies_to_test = [
    [256],            # 1. 1 hidden layer - Large
    [128],            # 2. 1 hidden layer - Medium
    [64],             # 3. 1 hidden layer - Small
    [32],             # 4. 1 hidden layer - Tiny
    [256, 128],       # 5. 2 hidden layers - Large
    [128, 64],        # 6. 2 hidden layers - Medium
    [64, 32],         # 7. 2 hidden layers - Small
    [96, 48]          # 8. 2 hidden layers - Alternative
]

ann_results = []

for (i, topology) in enumerate(topologies_to_test)
    println("\n[$i/8] Testing topology: $topology")
    
    hyperparams = Dict(
        "topology" => topology,
        "learningRate" => 0.003,
        "validationRatio" => 0.1,
        "numExecutions" => 1,
        "maxEpochs" => 800,
        "maxEpochsVal" => 25
    )
    
    # Use modelCrossValidation from utils.jl (course function)
    results = modelCrossValidation(
        :ANN,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(ann_results, (topology, f1_stats[1], results))
end

# Sort by F1 score
sorted_ann_results = sort(ann_results, by=x->x[2], rev=true)

println("\n🏆 ANN Results Ranking (by F1 Score):")
println("-"^70)
for (i, (topo, f1, _)) in enumerate(sorted_ann_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $topo - F1: $(round(f1*100, digits=2))%")
end

best_topology_ann = sorted_ann_results[1][1]
best_f1_ann = sorted_ann_results[1][2]
println("\n✨ Best ANN: $best_topology_ann (CV F1: $(round(best_f1_ann*100, digits=2))%)")

# Train final ANN on full training set and evaluate on test set
println("\n🚀 Training final ANN on full training set...")

# Prepare data
train_targets_onehot = oneHotEncoding(train_targets)
test_targets_onehot = oneHotEncoding(test_targets)

# Normalize
normParams_ann = calculateMinMaxNormalizationParameters(train_inputs)
train_inputs_norm = normalizeMinMax(train_inputs, normParams_ann)
test_inputs_norm = normalizeMinMax(test_inputs, normParams_ann)

# Create validation split (10% of train)
N_train = size(train_inputs_norm, 1)
(train_idx, val_idx) = holdOut(N_train, 0.1)

# Train final model
final_ann, _ = trainClassANN(best_topology_ann,
    (train_inputs_norm[train_idx, :], train_targets_onehot[train_idx, :]),
    validationDataset=(train_inputs_norm[val_idx, :], train_targets_onehot[val_idx, :]),
    testDataset=(test_inputs_norm, test_targets_onehot),
    maxEpochs=800,
    learningRate=0.003,
    maxEpochsVal=25)

# Predict on test set
test_outputs_ann = final_ann(test_inputs_norm')'

# Calculate metrics using confusionMatrix from course utils
cm_results_ann = confusionMatrix(test_outputs_ann, test_targets_onehot; weighted=true)

println("\n📊 ANN TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_ann.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_ann.aggregated.f1*100, digits=2))%")
println("\nConfusion Matrix:")
printConfusionMatrix(test_outputs_ann, test_targets_onehot; weighted=true)

# ============================================================================
#        EXPERIMENT 2: SUPPORT VECTOR MACHINES (SVMs)
# ============================================================================
#
# Configuration:
#   - 8 configurations tested (as required by assignment)
#   - Kernels: Linear, RBF, Polynomial
#   - Hyperparameter C: 0.1, 1.0, 10.0
#   - Gamma (RBF): auto (1/n_features = 0.125)
#   - Degree (Polynomial): 2, 3
#
# Configurations:
#   1-2. Linear (C = 1.0, 10.0)
#   3-5. RBF with different C values (C = 0.1, 1.0, 10.0)
#   6-8. Polynomial with different degrees (deg = 2, 3)
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 2: SUPPORT VECTOR MACHINES")
println("Testing 8 SVM Configurations")
println("="^70)

svm_configs = [
    ("linear", 1.0, 0.125, 3, "Linear C=1.0"),           # 1
    ("linear", 10.0, 0.125, 3, "Linear C=10.0"),         # 2
    ("rbf", 0.1, 0.125, 3, "RBF C=0.1 γ=auto"),          # 3
    ("rbf", 1.0, 0.125, 3, "RBF C=1.0 γ=auto"),          # 4
    ("rbf", 10.0, 0.125, 3, "RBF C=10.0 γ=auto"),        # 5
    ("poly", 1.0, 0.125, 2, "Poly C=1.0 deg=2"),         # 6
    ("poly", 1.0, 0.125, 3, "Poly C=1.0 deg=3"),         # 7
    ("poly", 10.0, 0.125, 2, "Poly C=10.0 deg=2")        # 8
]

svm_results = []

for (i, (kernel, C, gamma, degree, desc)) in enumerate(svm_configs)
    println("\n[$i/8] Testing: $desc")
    
    hyperparams = Dict(
        "kernel" => kernel,
        "C" => C,
        "gamma" => gamma,
        "degree" => degree
    )
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :SVC,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(svm_results, (desc, f1_stats[1], kernel, C, gamma, degree, results))
end

sorted_svm_results = sort(svm_results, by=x->x[2], rev=true)

println("\n🏆 SVM Results Ranking:")
println("-"^70)
for (i, (desc, f1, _, _, _, _, _)) in enumerate(sorted_svm_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $desc - F1: $(round(f1*100, digits=2))%")
end

best_svm = sorted_svm_results[1]
best_desc_svm, best_f1_svm, best_kernel_svm, best_C_svm, best_gamma_svm, best_degree_svm = best_svm[1:6]
println("\n✨ Best SVM: $best_desc_svm (CV F1: $(round(best_f1_svm*100, digits=2))%)")

# Train final SVM and evaluate on test set
println("\n🚀 Training final SVM on full training set...")

# Load MLJ for final model training
using MLJ
SVMClassifier = @load SVC pkg=LIBSVM

# Normalize
train_inputs_norm_svm = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_svm = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

# Convert to strings for MLJ
train_targets_str = string.(train_targets)
test_targets_str = string.(test_targets)
classes = sort(unique(train_targets_str))

# Set kernel
if best_kernel_svm == "linear"
    kernel_func = LIBSVM.Kernel.Linear
elseif best_kernel_svm == "poly"
    kernel_func = LIBSVM.Kernel.Polynomial
else
    kernel_func = LIBSVM.Kernel.RadialBasis
end

model_svm = SVMClassifier(
    kernel=kernel_func,
    cost=best_C_svm,
    gamma=best_gamma_svm,
    degree=Int32(best_degree_svm)
)

mach_svm = machine(model_svm, MLJ.table(train_inputs_norm_svm), categorical(train_targets_str))
MLJ.fit!(mach_svm, verbosity=0)

# Predict
svm_predictions = MLJ.predict(mach_svm, MLJ.table(test_inputs_norm_svm))
cm_results_svm = confusionMatrix(svm_predictions, test_targets_str, classes; weighted=true)

println("\n📊 SVM TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_svm.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_svm.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 3: DECISION TREES
# ============================================================================
#
# Configuration:
#   - 6 maximum depths tested (as required by assignment)
#   - Splitting criterion: Gini impurity
#   - Min samples split: 2
#   - Random seed: 42 (for reproducibility)
#
# Advantages:
#   - Interpretable (can visualize decision rules)
#   - No feature scaling required
#   - Fast training and prediction
#   - Handles non-linear relationships naturally
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 3: DECISION TREES")
println("Testing 6 Maximum Depths")
println("="^70)

tree_depths = [3, 5, 7, 10, 15, -1]  # 6 depths: 3, 5, 7, 10, 15, unlimited
tree_results = []

for (i, max_depth) in enumerate(tree_depths)
    depth_str = max_depth == -1 ? "Unlimited" : string(max_depth)
    println("\n[$i/6] Testing: Depth=$depth_str")
    
    hyperparams = Dict("max_depth" => max_depth)
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :DecisionTreeClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(tree_results, (depth_str, max_depth, f1_stats[1], results))
end

sorted_tree_results = sort(tree_results, by=x->x[3], rev=true)

println("\n🏆 Decision Tree Results Ranking:")
println("-"^70)
for (i, (depth_str, _, f1, _)) in enumerate(sorted_tree_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. Depth=$depth_str - F1: $(round(f1*100, digits=2))%")
end

best_desc_tree, best_max_depth_tree, best_f1_tree = sorted_tree_results[1][1:3]
println("\n✨ Best Tree: Depth=$best_desc_tree (CV F1: $(round(best_f1_tree*100, digits=2))%)")

# Train final Decision Tree
println("\n🚀 Training final Decision Tree on full training set...")

DTClassifier = @load DecisionTreeClassifier pkg=DecisionTree

train_inputs_norm_tree = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_tree = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

train_targets_str_tree = string.(train_targets)
test_targets_str_tree = string.(test_targets)

if best_max_depth_tree == -1
    model_tree = DTClassifier(rng=Random.MersenneTwister(42))
else
    model_tree = DTClassifier(max_depth=best_max_depth_tree, rng=Random.MersenneTwister(42))
end

mach_tree = machine(model_tree, MLJ.table(train_inputs_norm_tree), categorical(train_targets_str_tree))
MLJ.fit!(mach_tree, verbosity=0)

tree_predictions = MLJ.predict(mach_tree, MLJ.table(test_inputs_norm_tree))
tree_predictions_mode = mode.(tree_predictions)

cm_results_tree = confusionMatrix(tree_predictions_mode, test_targets_str_tree, classes; weighted=true)

println("\n📊 DECISION TREE TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_tree.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_tree.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 4: k-NEAREST NEIGHBORS (kNN)
# ============================================================================
#
# Configuration:
#   - 6 k values tested: 1, 3, 5, 7, 10, 15
#   - Distance metric: Euclidean
#   - Voting: Majority voting among k neighbors
#
# Notes:
#   - Feature normalization is CRITICAL for kNN (distance-based)
#   - No explicit training phase (lazy learning)
#   - k=1 is most sensitive to noise
#   - Higher k values create smoother decision boundaries
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 4: k-NEAREST NEIGHBORS")
println("Testing 6 k Values")
println("="^70)

k_values = [1, 3, 5, 7, 10, 15]
knn_results = []

for (i, k) in enumerate(k_values)
    println("\n[$i/6] Testing: k=$k")
    
    hyperparams = Dict("n_neighbors" => k)
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :KNeighborsClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(knn_results, (k, f1_stats[1], results))
end

sorted_knn_results = sort(knn_results, by=x->x[2], rev=true)

println("\n🏆 kNN Results Ranking:")
println("-"^70)
for (i, (k, f1, _)) in enumerate(sorted_knn_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. k=$k - F1: $(round(f1*100, digits=2))%")
end

best_k_knn, best_f1_knn = sorted_knn_results[1][1:2]
println("\n✨ Best kNN: k=$best_k_knn (CV F1: $(round(best_f1_knn*100, digits=2))%)")

# Train final kNN
println("\n🚀 Preparing final kNN...")

kNNClassifier = @load KNNClassifier pkg=NearestNeighborModels

train_inputs_norm_knn = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm_knn = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))

train_targets_str_knn = string.(train_targets)
test_targets_str_knn = string.(test_targets)

model_knn = kNNClassifier(K=best_k_knn)

mach_knn = machine(model_knn, MLJ.table(train_inputs_norm_knn), categorical(train_targets_str_knn))
MLJ.fit!(mach_knn, verbosity=0)

knn_predictions = MLJ.predict(mach_knn, MLJ.table(test_inputs_norm_knn))
knn_predictions_mode = mode.(knn_predictions)

cm_results_knn = confusionMatrix(knn_predictions_mode, test_targets_str_knn, classes; weighted=true)

println("\n📊 kNN TEST SET RESULTS:")
println("="^70)
println("Accuracy:  $(round(cm_results_knn.accuracy*100, digits=2))%")
println("F1 Score:  $(round(cm_results_knn.aggregated.f1*100, digits=2))%")

# ============================================================================
#            EXPERIMENT 5: ENSEMBLE METHODS
# ============================================================================
#
# Strategy: Combine the top 3 individual models to improve robustness
#
# Models Selected:
#   1. Best ANN
#   2. Best Decision Tree
#   3. Best kNN
#
# Ensemble Techniques:
#   1. Majority Voting: Each model votes equally, winner takes all
#   2. Weighted Voting: Models vote proportionally to their CV F1 scores
#
# Expected Benefits:
#   - Reduced variance through model averaging
#   - More robust predictions
#   - Leverage complementary strengths of different algorithms
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 5: ENSEMBLE METHODS")
println("Combining ANN + Decision Tree + kNN")
println("="^70)

# Helper functions for ensemble
function majorityVoting(predictions::Vector{Vector{String}})
    n_samples = length(predictions[1])
    ensemble_predictions = Vector{String}(undef, n_samples)
    
    for i in 1:n_samples
        votes = [pred[i] for pred in predictions]
        ensemble_predictions[i] = mode(votes)
    end
    
    return ensemble_predictions
end

function weightedVoting(predictions::Vector{Vector{String}}, weights::Vector{Float64})
    n_samples = length(predictions[1])
    n_models = length(predictions)
    classes_unique = sort(unique(vcat(predictions...)))
    n_classes = length(classes_unique)
    
    ensemble_predictions = Vector{String}(undef, n_samples)
    
    for i in 1:n_samples
        class_scores = Dict(c => 0.0 for c in classes_unique)
        
        for j in 1:n_models
            class_pred = predictions[j][i]
            class_scores[class_pred] += weights[j]
        end
        
        ensemble_predictions[i] = argmax(class_scores)
    end
    
    return ensemble_predictions
end

# Get test predictions from all 3 models as string vectors
ann_test_pred_str = string.(argmax.(eachrow(test_outputs_ann)) .- 1)
tree_test_pred_str = string.(tree_predictions_mode)
knn_test_pred_str = string.(knn_predictions_mode)

all_predictions = [ann_test_pred_str, tree_test_pred_str, knn_test_pred_str]

# Method 1: Majority Voting
println("\n[1/2] Majority Voting...")
majority_predictions = majorityVoting(all_predictions)
cm_results_majority = confusionMatrix(majority_predictions, test_targets_str, classes; weighted=true)
println("✅ Majority Voting - F1: $(round(cm_results_majority.aggregated.f1*100, digits=2))%")

# Method 2: Weighted Voting
println("\n[2/2] Weighted Voting...")
cv_scores = [best_f1_ann, best_f1_tree, best_f1_knn]
weights = cv_scores ./ sum(cv_scores)

println("  Model weights:")
println("    ANN:           $(round(weights[1]*100, digits=1))%")
println("    Decision Tree: $(round(weights[2]*100, digits=1))%")
println("    kNN:           $(round(weights[3]*100, digits=1))%")

weighted_predictions = weightedVoting(all_predictions, weights)
cm_results_weighted = confusionMatrix(weighted_predictions, test_targets_str, classes; weighted=true)
println("✅ Weighted Voting - F1: $(round(cm_results_weighted.aggregated.f1*100, digits=2))%")

# ============================================================================
#              FINAL RESULTS COMPARISON
# ============================================================================
#
# Comprehensive comparison of all 6 approaches on the hold-out test set.
#
# Evaluation Metrics:
#   - F1 Score: Harmonic mean of precision and recall
#   - Accuracy: Overall correct predictions
#   - Per-Class Metrics: Performance for each risk level
#
# Key Question: Which approach best balances overall performance 
#               with fraud detection capability?
#
# ============================================================================

println("\n" * "="^70)
println("📊 APPROACH 1 RESULTS - 80/20 SPLIT")
println("="^70)

println("\n🏆 TEST SET PERFORMANCE RANKING:")
println("="^70)
println("Rank | Approach                | F1 Score   | Accuracy")
println("-"^70)
println("1.   | ANN ($best_topology_ann)         | $(rpad(round(cm_results_ann.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_ann.accuracy*100, digits=2))%")
println("2.   | Decision Tree (d=$best_desc_tree)   | $(rpad(round(cm_results_tree.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_tree.accuracy*100, digits=2))%")
println("3.   | kNN (k=$best_k_knn)            | $(rpad(round(cm_results_knn.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_knn.accuracy*100, digits=2))%")
println("4.   | SVM ($best_desc_svm)  | $(rpad(round(cm_results_svm.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_svm.accuracy*100, digits=2))%")
println("-"^70)
println("5.   | Ensemble (Majority)     | $(rpad(round(cm_results_majority.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_majority.accuracy*100, digits=2))%")
println("6.   | Ensemble (Weighted)     | $(rpad(round(cm_results_weighted.aggregated.f1*100, digits=2), 7))% | $(round(cm_results_weighted.accuracy*100, digits=2))%")
println("="^70)

# Store Approach 1 results
approach1_results = Dict(
    "split" => "80/20",
    "ann" => (best_topology_ann, cm_results_ann.aggregated.f1, cm_results_ann.accuracy),
    "svm" => (best_desc_svm, cm_results_svm.aggregated.f1, cm_results_svm.accuracy),
    "tree" => (best_desc_tree, cm_results_tree.aggregated.f1, cm_results_tree.accuracy),
    "knn" => (best_k_knn, cm_results_knn.aggregated.f1, cm_results_knn.accuracy),
    "ensemble_majority" => (cm_results_majority.aggregated.f1, cm_results_majority.accuracy),
    "ensemble_weighted" => (cm_results_weighted.aggregated.f1, cm_results_weighted.accuracy)
)

# ############################################################################
#
#                         APPROACH 2: 50/50 SPLIT
#
# ############################################################################
#
# This approach tests the models with a balanced 50/50 train/test split.
# The goal is to test generalization with less training data.
#
# ############################################################################

println("\n")
println("="^80)
println("="^80)
println("                    APPROACH 2: 50/50 TRAIN/TEST SPLIT")
println("="^80)
println("="^80)

# Re-seed for reproducibility
Random.seed!(42)

# Use the same balanced dataset but with 50/50 split
println("\n" * "="^70)
println("✅ APPROACH 2: TRAIN/TEST SPLIT (50% Train / 50% Test)")
println("="^70)

n_train_50 = floor(Int, n_total * 0.50)
n_test_50 = n_total - n_train_50

all_indices_50 = shuffle(1:n_total)
train_indices_50 = all_indices_50[1:n_train_50]
test_indices_50 = all_indices_50[n_train_50+1:end]

df_train_50 = df_balanced[train_indices_50, :]
df_test_50 = df_balanced[test_indices_50, :]

println("\n📊 Split Summary (Approach 2):")
println("  Total samples:     $n_total")
println("  Training set:      $n_train_50 (50%)")
println("  Test set:          $n_test_50 (50%)")

# Preprocess
println("\n🔧 Preprocessing train and test sets (Approach 2)...")
df_train_50_processed = preprocess_multiclass(df_train_50, target_col)
df_test_50_processed = preprocess_multiclass(df_test_50, target_col)

input_cols_50 = setdiff(names(df_train_50_processed), ["Risk_Class"])
train_inputs_50 = Matrix{Float64}(df_train_50_processed[:, input_cols_50])
train_targets_50 = Int.(df_train_50_processed.Risk_Class)

test_inputs_50 = Matrix{Float64}(df_test_50_processed[:, input_cols_50])
test_targets_50 = Int.(df_test_50_processed.Risk_Class)

# Create cross-validation indices
cv_indices_50 = crossvalidation(train_targets_50, k_folds)
println("✅ Cross-validation indices created ($k_folds folds, stratified)")

# ============================================================================
# APPROACH 2: ANN (8 topologies)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 2 - EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS")
println("Testing 8 ANN Topologies")
println("="^70)

ann_results_50 = []
for (i, topology) in enumerate(topologies_to_test)
    println("\n[$i/8] Testing topology: $topology")
    
    hyperparams = Dict(
        "topology" => topology,
        "learningRate" => 0.003,
        "validationRatio" => 0.1,
        "numExecutions" => 1,
        "maxEpochs" => 800,
        "maxEpochsVal" => 25
    )
    
    results = modelCrossValidation(:ANN, hyperparams, (train_inputs_50, train_targets_50), cv_indices_50)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(ann_results_50, (topology, f1_stats[1], results))
end

sorted_ann_results_50 = sort(ann_results_50, by=x->x[2], rev=true)
best_topology_ann_50 = sorted_ann_results_50[1][1]
best_f1_ann_50 = sorted_ann_results_50[1][2]
println("\n✨ Best ANN (Approach 2): $best_topology_ann_50 (CV F1: $(round(best_f1_ann_50*100, digits=2))%)")

# Train final ANN (Approach 2)
train_targets_onehot_50 = oneHotEncoding(train_targets_50)
test_targets_onehot_50 = oneHotEncoding(test_targets_50)
normParams_ann_50 = calculateMinMaxNormalizationParameters(train_inputs_50)
train_inputs_norm_50 = normalizeMinMax(train_inputs_50, normParams_ann_50)
test_inputs_norm_50 = normalizeMinMax(test_inputs_50, normParams_ann_50)

N_train_50_val = size(train_inputs_norm_50, 1)
(train_idx_50, val_idx_50) = holdOut(N_train_50_val, 0.1)

final_ann_50, _ = trainClassANN(best_topology_ann_50,
    (train_inputs_norm_50[train_idx_50, :], train_targets_onehot_50[train_idx_50, :]),
    validationDataset=(train_inputs_norm_50[val_idx_50, :], train_targets_onehot_50[val_idx_50, :]),
    testDataset=(test_inputs_norm_50, test_targets_onehot_50),
    maxEpochs=800, learningRate=0.003, maxEpochsVal=25)

test_outputs_ann_50 = final_ann_50(test_inputs_norm_50')'
cm_results_ann_50 = confusionMatrix(test_outputs_ann_50, test_targets_onehot_50; weighted=true)
println("\n📊 ANN TEST SET RESULTS (Approach 2): F1=$(round(cm_results_ann_50.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_ann_50.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 2: SVM (8 configurations)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 2 - EXPERIMENT 2: SUPPORT VECTOR MACHINES")
println("Testing 8 SVM Configurations")
println("="^70)

svm_results_50 = []
for (i, (kernel, C, gamma, degree, desc)) in enumerate(svm_configs)
    println("\n[$i/8] Testing: $desc")
    hyperparams = Dict("kernel" => kernel, "C" => C, "gamma" => gamma, "degree" => degree)
    results = modelCrossValidation(:SVC, hyperparams, (train_inputs_50, train_targets_50), cv_indices_50)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(svm_results_50, (desc, f1_stats[1], kernel, C, gamma, degree, results))
end

sorted_svm_results_50 = sort(svm_results_50, by=x->x[2], rev=true)
best_svm_50 = sorted_svm_results_50[1]
best_desc_svm_50, best_f1_svm_50, best_kernel_svm_50, best_C_svm_50, best_gamma_svm_50, best_degree_svm_50 = best_svm_50[1:6]
println("\n✨ Best SVM (Approach 2): $best_desc_svm_50 (CV F1: $(round(best_f1_svm_50*100, digits=2))%)")

# Train final SVM (Approach 2)
train_inputs_norm_svm_50 = normalizeMinMax(train_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
test_inputs_norm_svm_50 = normalizeMinMax(test_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
train_targets_str_50 = string.(train_targets_50)
test_targets_str_50 = string.(test_targets_50)
classes_50 = sort(unique(train_targets_str_50))

if best_kernel_svm_50 == "linear"
    kernel_func_50 = LIBSVM.Kernel.Linear
elseif best_kernel_svm_50 == "poly"
    kernel_func_50 = LIBSVM.Kernel.Polynomial
else
    kernel_func_50 = LIBSVM.Kernel.RadialBasis
end

model_svm_50 = SVMClassifier(kernel=kernel_func_50, cost=best_C_svm_50, gamma=best_gamma_svm_50, degree=Int32(best_degree_svm_50))
mach_svm_50 = machine(model_svm_50, MLJ.table(train_inputs_norm_svm_50), categorical(train_targets_str_50))
MLJ.fit!(mach_svm_50, verbosity=0)
svm_predictions_50 = MLJ.predict(mach_svm_50, MLJ.table(test_inputs_norm_svm_50))
cm_results_svm_50 = confusionMatrix(svm_predictions_50, test_targets_str_50, classes_50; weighted=true)
println("📊 SVM TEST SET RESULTS (Approach 2): F1=$(round(cm_results_svm_50.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_svm_50.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 2: Decision Trees (6 depths)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 2 - EXPERIMENT 3: DECISION TREES")
println("Testing 6 Maximum Depths")
println("="^70)

tree_results_50 = []
for (i, max_depth) in enumerate(tree_depths)
    depth_str = max_depth == -1 ? "Unlimited" : string(max_depth)
    println("\n[$i/6] Testing: Depth=$depth_str")
    hyperparams = Dict("max_depth" => max_depth)
    results = modelCrossValidation(:DecisionTreeClassifier, hyperparams, (train_inputs_50, train_targets_50), cv_indices_50)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(tree_results_50, (depth_str, max_depth, f1_stats[1], results))
end

sorted_tree_results_50 = sort(tree_results_50, by=x->x[3], rev=true)
best_desc_tree_50, best_max_depth_tree_50, best_f1_tree_50 = sorted_tree_results_50[1][1:3]
println("\n✨ Best Tree (Approach 2): Depth=$best_desc_tree_50 (CV F1: $(round(best_f1_tree_50*100, digits=2))%)")

# Train final Decision Tree (Approach 2)
train_inputs_norm_tree_50 = normalizeMinMax(train_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
test_inputs_norm_tree_50 = normalizeMinMax(test_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
train_targets_str_tree_50 = string.(train_targets_50)
test_targets_str_tree_50 = string.(test_targets_50)

if best_max_depth_tree_50 == -1
    model_tree_50 = DTClassifier(rng=Random.MersenneTwister(42))
else
    model_tree_50 = DTClassifier(max_depth=best_max_depth_tree_50, rng=Random.MersenneTwister(42))
end
mach_tree_50 = machine(model_tree_50, MLJ.table(train_inputs_norm_tree_50), categorical(train_targets_str_tree_50))
MLJ.fit!(mach_tree_50, verbosity=0)
tree_predictions_50 = MLJ.predict(mach_tree_50, MLJ.table(test_inputs_norm_tree_50))
tree_predictions_mode_50 = mode.(tree_predictions_50)
cm_results_tree_50 = confusionMatrix(tree_predictions_mode_50, test_targets_str_tree_50, classes_50; weighted=true)
println("📊 Decision Tree TEST SET RESULTS (Approach 2): F1=$(round(cm_results_tree_50.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_tree_50.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 2: kNN (6 k values)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 2 - EXPERIMENT 4: k-NEAREST NEIGHBORS")
println("Testing 6 k Values")
println("="^70)

knn_results_50 = []
for (i, k) in enumerate(k_values)
    println("\n[$i/6] Testing: k=$k")
    hyperparams = Dict("n_neighbors" => k)
    results = modelCrossValidation(:KNeighborsClassifier, hyperparams, (train_inputs_50, train_targets_50), cv_indices_50)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(knn_results_50, (k, f1_stats[1], results))
end

sorted_knn_results_50 = sort(knn_results_50, by=x->x[2], rev=true)
best_k_knn_50, best_f1_knn_50 = sorted_knn_results_50[1][1:2]
println("\n✨ Best kNN (Approach 2): k=$best_k_knn_50 (CV F1: $(round(best_f1_knn_50*100, digits=2))%)")

# Train final kNN (Approach 2)
train_inputs_norm_knn_50 = normalizeMinMax(train_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
test_inputs_norm_knn_50 = normalizeMinMax(test_inputs_50, calculateMinMaxNormalizationParameters(train_inputs_50))
train_targets_str_knn_50 = string.(train_targets_50)
test_targets_str_knn_50 = string.(test_targets_50)

model_knn_50 = kNNClassifier(K=best_k_knn_50)
mach_knn_50 = machine(model_knn_50, MLJ.table(train_inputs_norm_knn_50), categorical(train_targets_str_knn_50))
MLJ.fit!(mach_knn_50, verbosity=0)
knn_predictions_50 = MLJ.predict(mach_knn_50, MLJ.table(test_inputs_norm_knn_50))
knn_predictions_mode_50 = mode.(knn_predictions_50)
cm_results_knn_50 = confusionMatrix(knn_predictions_mode_50, test_targets_str_knn_50, classes_50; weighted=true)
println("📊 kNN TEST SET RESULTS (Approach 2): F1=$(round(cm_results_knn_50.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_knn_50.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 2: Ensemble Methods
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 2 - EXPERIMENT 5: ENSEMBLE METHODS")
println("Combining ANN + Decision Tree + kNN")
println("="^70)

ann_test_pred_str_50 = string.(argmax.(eachrow(test_outputs_ann_50)) .- 1)
tree_test_pred_str_50 = string.(tree_predictions_mode_50)
knn_test_pred_str_50 = string.(knn_predictions_mode_50)
all_predictions_50 = [ann_test_pred_str_50, tree_test_pred_str_50, knn_test_pred_str_50]

majority_predictions_50 = majorityVoting(all_predictions_50)
cm_results_majority_50 = confusionMatrix(majority_predictions_50, test_targets_str_50, classes_50; weighted=true)
println("✅ Majority Voting - F1: $(round(cm_results_majority_50.aggregated.f1*100, digits=2))%")

cv_scores_50 = [best_f1_ann_50, best_f1_tree_50, best_f1_knn_50]
weights_50 = cv_scores_50 ./ sum(cv_scores_50)
weighted_predictions_50 = weightedVoting(all_predictions_50, weights_50)
cm_results_weighted_50 = confusionMatrix(weighted_predictions_50, test_targets_str_50, classes_50; weighted=true)
println("✅ Weighted Voting - F1: $(round(cm_results_weighted_50.aggregated.f1*100, digits=2))%")

# Store Approach 2 results
approach2_results = Dict(
    "split" => "50/50",
    "ann" => (best_topology_ann_50, cm_results_ann_50.aggregated.f1, cm_results_ann_50.accuracy),
    "svm" => (best_desc_svm_50, cm_results_svm_50.aggregated.f1, cm_results_svm_50.accuracy),
    "tree" => (best_desc_tree_50, cm_results_tree_50.aggregated.f1, cm_results_tree_50.accuracy),
    "knn" => (best_k_knn_50, cm_results_knn_50.aggregated.f1, cm_results_knn_50.accuracy),
    "ensemble_majority" => (cm_results_majority_50.aggregated.f1, cm_results_majority_50.accuracy),
    "ensemble_weighted" => (cm_results_weighted_50.aggregated.f1, cm_results_weighted_50.accuracy)
)

# ############################################################################
#
#                         APPROACH 3: PCA + 80/20 SPLIT
#
# ############################################################################
#
# This approach applies Principal Component Analysis (PCA) for dimensionality
# reduction before training the models. We keep enough components to explain
# 95% of the variance.
#
# ############################################################################

println("\n")
println("="^80)
println("="^80)
println("                    APPROACH 3: PCA + 80/20 TRAIN/TEST SPLIT")
println("="^80)
println("="^80)

# Re-seed for reproducibility
Random.seed!(42)

println("\n" * "="^70)
println("✅ APPROACH 3: PCA DIMENSIONALITY REDUCTION")
println("="^70)

# Use the original 80/20 split data
# First normalize the training data
normParams_pca = calculateMinMaxNormalizationParameters(train_inputs)
train_inputs_norm_pca = normalizeMinMax(train_inputs, normParams_pca)
test_inputs_norm_pca = normalizeMinMax(test_inputs, normParams_pca)

# Apply PCA to training data
# Center the data
train_mean = mean(train_inputs_norm_pca, dims=1)
train_centered = train_inputs_norm_pca .- train_mean
test_centered = test_inputs_norm_pca .- train_mean

# Compute covariance matrix and eigendecomposition
cov_matrix = (train_centered' * train_centered) / (size(train_centered, 1) - 1)
eigenvalues, eigenvectors = eigen(cov_matrix)

# Sort by eigenvalue (descending)
sorted_indices = sortperm(eigenvalues, rev=true)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Determine number of components for 95% variance
total_variance = sum(eigenvalues)
explained_variance_ratio = eigenvalues ./ total_variance
cumulative_variance = cumsum(explained_variance_ratio)

n_components = findfirst(x -> x >= 0.95, cumulative_variance)
if n_components === nothing
    n_components = length(eigenvalues)
end

println("\n📊 PCA Analysis:")
println("  Original features:       $(size(train_inputs, 2))")
println("  Components for 95% var:  $n_components")
println("  Variance explained:      $(round(cumulative_variance[n_components]*100, digits=2))%")

# Project data onto principal components
projection_matrix = eigenvectors[:, 1:n_components]
train_inputs_pca = train_centered * projection_matrix
test_inputs_pca = test_centered * projection_matrix

println("  Reduced features:        $(size(train_inputs_pca, 2))")

# Create cross-validation indices for PCA approach
cv_indices_pca = crossvalidation(train_targets, k_folds)
println("\n✅ Cross-validation indices created ($k_folds folds, stratified)")

# ============================================================================
# APPROACH 3: ANN with PCA (8 topologies)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 3 - EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS (PCA)")
println("Testing 8 ANN Topologies")
println("="^70)

# Adjust topologies for reduced input dimension
topologies_pca = [
    [64],             # 1. 1 hidden layer
    [32],             # 2. 1 hidden layer
    [16],             # 3. 1 hidden layer
    [8],              # 4. 1 hidden layer
    [64, 32],         # 5. 2 hidden layers
    [32, 16],         # 6. 2 hidden layers
    [16, 8],          # 7. 2 hidden layers
    [32, 8]           # 8. 2 hidden layers
]

ann_results_pca = []
for (i, topology) in enumerate(topologies_pca)
    println("\n[$i/8] Testing topology: $topology")
    
    hyperparams = Dict(
        "topology" => topology,
        "learningRate" => 0.003,
        "validationRatio" => 0.1,
        "numExecutions" => 1,
        "maxEpochs" => 800,
        "maxEpochsVal" => 25
    )
    
    results = modelCrossValidation(:ANN, hyperparams, (train_inputs_pca, train_targets), cv_indices_pca)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(ann_results_pca, (topology, f1_stats[1], results))
end

sorted_ann_results_pca = sort(ann_results_pca, by=x->x[2], rev=true)
best_topology_ann_pca = sorted_ann_results_pca[1][1]
best_f1_ann_pca = sorted_ann_results_pca[1][2]
println("\n✨ Best ANN (PCA): $best_topology_ann_pca (CV F1: $(round(best_f1_ann_pca*100, digits=2))%)")

# Train final ANN (PCA)
train_targets_onehot_pca = oneHotEncoding(train_targets)
test_targets_onehot_pca = oneHotEncoding(test_targets)

N_train_pca = size(train_inputs_pca, 1)
(train_idx_pca, val_idx_pca) = holdOut(N_train_pca, 0.1)

final_ann_pca, _ = trainClassANN(best_topology_ann_pca,
    (train_inputs_pca[train_idx_pca, :], train_targets_onehot_pca[train_idx_pca, :]),
    validationDataset=(train_inputs_pca[val_idx_pca, :], train_targets_onehot_pca[val_idx_pca, :]),
    testDataset=(test_inputs_pca, test_targets_onehot_pca),
    maxEpochs=800, learningRate=0.003, maxEpochsVal=25)

test_outputs_ann_pca = final_ann_pca(test_inputs_pca')'
cm_results_ann_pca = confusionMatrix(test_outputs_ann_pca, test_targets_onehot_pca; weighted=true)
println("\n📊 ANN TEST SET RESULTS (PCA): F1=$(round(cm_results_ann_pca.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_ann_pca.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 3: SVM with PCA (8 configurations)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 3 - EXPERIMENT 2: SUPPORT VECTOR MACHINES (PCA)")
println("Testing 8 SVM Configurations")
println("="^70)

svm_results_pca = []
for (i, (kernel, C, gamma, degree, desc)) in enumerate(svm_configs)
    println("\n[$i/8] Testing: $desc")
    hyperparams = Dict("kernel" => kernel, "C" => C, "gamma" => gamma, "degree" => degree)
    results = modelCrossValidation(:SVC, hyperparams, (train_inputs_pca, train_targets), cv_indices_pca)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(svm_results_pca, (desc, f1_stats[1], kernel, C, gamma, degree, results))
end

sorted_svm_results_pca = sort(svm_results_pca, by=x->x[2], rev=true)
best_svm_pca = sorted_svm_results_pca[1]
best_desc_svm_pca, best_f1_svm_pca, best_kernel_svm_pca, best_C_svm_pca, best_gamma_svm_pca, best_degree_svm_pca = best_svm_pca[1:6]
println("\n✨ Best SVM (PCA): $best_desc_svm_pca (CV F1: $(round(best_f1_svm_pca*100, digits=2))%)")

# Train final SVM (PCA)
train_targets_str_pca = string.(train_targets)
test_targets_str_pca = string.(test_targets)
classes_pca = sort(unique(train_targets_str_pca))

if best_kernel_svm_pca == "linear"
    kernel_func_pca = LIBSVM.Kernel.Linear
elseif best_kernel_svm_pca == "poly"
    kernel_func_pca = LIBSVM.Kernel.Polynomial
else
    kernel_func_pca = LIBSVM.Kernel.RadialBasis
end

model_svm_pca = SVMClassifier(kernel=kernel_func_pca, cost=best_C_svm_pca, gamma=best_gamma_svm_pca, degree=Int32(best_degree_svm_pca))
mach_svm_pca = machine(model_svm_pca, MLJ.table(train_inputs_pca), categorical(train_targets_str_pca))
MLJ.fit!(mach_svm_pca, verbosity=0)
svm_predictions_pca = MLJ.predict(mach_svm_pca, MLJ.table(test_inputs_pca))
cm_results_svm_pca = confusionMatrix(svm_predictions_pca, test_targets_str_pca, classes_pca; weighted=true)
println("📊 SVM TEST SET RESULTS (PCA): F1=$(round(cm_results_svm_pca.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_svm_pca.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 3: Decision Trees with PCA (6 depths)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 3 - EXPERIMENT 3: DECISION TREES (PCA)")
println("Testing 6 Maximum Depths")
println("="^70)

tree_results_pca = []
for (i, max_depth) in enumerate(tree_depths)
    depth_str = max_depth == -1 ? "Unlimited" : string(max_depth)
    println("\n[$i/6] Testing: Depth=$depth_str")
    hyperparams = Dict("max_depth" => max_depth)
    results = modelCrossValidation(:DecisionTreeClassifier, hyperparams, (train_inputs_pca, train_targets), cv_indices_pca)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(tree_results_pca, (depth_str, max_depth, f1_stats[1], results))
end

sorted_tree_results_pca = sort(tree_results_pca, by=x->x[3], rev=true)
best_desc_tree_pca, best_max_depth_tree_pca, best_f1_tree_pca = sorted_tree_results_pca[1][1:3]
println("\n✨ Best Tree (PCA): Depth=$best_desc_tree_pca (CV F1: $(round(best_f1_tree_pca*100, digits=2))%)")

# Train final Decision Tree (PCA)
train_targets_str_tree_pca = string.(train_targets)
test_targets_str_tree_pca = string.(test_targets)

if best_max_depth_tree_pca == -1
    model_tree_pca = DTClassifier(rng=Random.MersenneTwister(42))
else
    model_tree_pca = DTClassifier(max_depth=best_max_depth_tree_pca, rng=Random.MersenneTwister(42))
end
mach_tree_pca = machine(model_tree_pca, MLJ.table(train_inputs_pca), categorical(train_targets_str_tree_pca))
MLJ.fit!(mach_tree_pca, verbosity=0)
tree_predictions_pca = MLJ.predict(mach_tree_pca, MLJ.table(test_inputs_pca))
tree_predictions_mode_pca = mode.(tree_predictions_pca)
cm_results_tree_pca = confusionMatrix(tree_predictions_mode_pca, test_targets_str_tree_pca, classes_pca; weighted=true)
println("📊 Decision Tree TEST SET RESULTS (PCA): F1=$(round(cm_results_tree_pca.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_tree_pca.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 3: kNN with PCA (6 k values)
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 3 - EXPERIMENT 4: k-NEAREST NEIGHBORS (PCA)")
println("Testing 6 k Values")
println("="^70)

knn_results_pca = []
for (i, k) in enumerate(k_values)
    println("\n[$i/6] Testing: k=$k")
    hyperparams = Dict("n_neighbors" => k)
    results = modelCrossValidation(:KNeighborsClassifier, hyperparams, (train_inputs_pca, train_targets), cv_indices_pca)
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    push!(knn_results_pca, (k, f1_stats[1], results))
end

sorted_knn_results_pca = sort(knn_results_pca, by=x->x[2], rev=true)
best_k_knn_pca, best_f1_knn_pca = sorted_knn_results_pca[1][1:2]
println("\n✨ Best kNN (PCA): k=$best_k_knn_pca (CV F1: $(round(best_f1_knn_pca*100, digits=2))%)")

# Train final kNN (PCA)
train_targets_str_knn_pca = string.(train_targets)
test_targets_str_knn_pca = string.(test_targets)

model_knn_pca = kNNClassifier(K=best_k_knn_pca)
mach_knn_pca = machine(model_knn_pca, MLJ.table(train_inputs_pca), categorical(train_targets_str_knn_pca))
MLJ.fit!(mach_knn_pca, verbosity=0)
knn_predictions_pca = MLJ.predict(mach_knn_pca, MLJ.table(test_inputs_pca))
knn_predictions_mode_pca = mode.(knn_predictions_pca)
cm_results_knn_pca = confusionMatrix(knn_predictions_mode_pca, test_targets_str_knn_pca, classes_pca; weighted=true)
println("📊 kNN TEST SET RESULTS (PCA): F1=$(round(cm_results_knn_pca.aggregated.f1*100, digits=2))%, Acc=$(round(cm_results_knn_pca.accuracy*100, digits=2))%")

# ============================================================================
# APPROACH 3: Ensemble Methods with PCA
# ============================================================================
println("\n" * "="^70)
println("🔬 APPROACH 3 - EXPERIMENT 5: ENSEMBLE METHODS (PCA)")
println("Combining ANN + Decision Tree + kNN")
println("="^70)

ann_test_pred_str_pca = string.(argmax.(eachrow(test_outputs_ann_pca)) .- 1)
tree_test_pred_str_pca = string.(tree_predictions_mode_pca)
knn_test_pred_str_pca = string.(knn_predictions_mode_pca)
all_predictions_pca = [ann_test_pred_str_pca, tree_test_pred_str_pca, knn_test_pred_str_pca]

majority_predictions_pca = majorityVoting(all_predictions_pca)
cm_results_majority_pca = confusionMatrix(majority_predictions_pca, test_targets_str_pca, classes_pca; weighted=true)
println("✅ Majority Voting - F1: $(round(cm_results_majority_pca.aggregated.f1*100, digits=2))%")

cv_scores_pca = [best_f1_ann_pca, best_f1_tree_pca, best_f1_knn_pca]
weights_pca = cv_scores_pca ./ sum(cv_scores_pca)
weighted_predictions_pca = weightedVoting(all_predictions_pca, weights_pca)
cm_results_weighted_pca = confusionMatrix(weighted_predictions_pca, test_targets_str_pca, classes_pca; weighted=true)
println("✅ Weighted Voting - F1: $(round(cm_results_weighted_pca.aggregated.f1*100, digits=2))%")

# Store Approach 3 results
approach3_results = Dict(
    "split" => "80/20 + PCA",
    "n_components" => n_components,
    "variance_explained" => cumulative_variance[n_components],
    "ann" => (best_topology_ann_pca, cm_results_ann_pca.aggregated.f1, cm_results_ann_pca.accuracy),
    "svm" => (best_desc_svm_pca, cm_results_svm_pca.aggregated.f1, cm_results_svm_pca.accuracy),
    "tree" => (best_desc_tree_pca, cm_results_tree_pca.aggregated.f1, cm_results_tree_pca.accuracy),
    "knn" => (best_k_knn_pca, cm_results_knn_pca.aggregated.f1, cm_results_knn_pca.accuracy),
    "ensemble_majority" => (cm_results_majority_pca.aggregated.f1, cm_results_majority_pca.accuracy),
    "ensemble_weighted" => (cm_results_weighted_pca.aggregated.f1, cm_results_weighted_pca.accuracy)
)

# ############################################################################
#
#                    FINAL SUMMARY - ALL APPROACHES
#
# ############################################################################

println("\n")
println("="^80)
println("="^80)
println("                         FINAL SUMMARY - ALL APPROACHES")
println("="^80)
println("="^80)

println("\n" * "="^70)
println("📊 COMPARISON OF ALL 3 APPROACHES")
println("="^70)

println("\n🏆 APPROACH 1: 80/20 SPLIT (Original)")
println("-"^70)
println("  ANN ($best_topology_ann):     F1=$(round(approach1_results["ann"][2]*100, digits=2))%, Acc=$(round(approach1_results["ann"][3]*100, digits=2))%")
println("  SVM ($best_desc_svm):         F1=$(round(approach1_results["svm"][2]*100, digits=2))%, Acc=$(round(approach1_results["svm"][3]*100, digits=2))%")
println("  Decision Tree (d=$best_desc_tree): F1=$(round(approach1_results["tree"][2]*100, digits=2))%, Acc=$(round(approach1_results["tree"][3]*100, digits=2))%")
println("  kNN (k=$best_k_knn):          F1=$(round(approach1_results["knn"][2]*100, digits=2))%, Acc=$(round(approach1_results["knn"][3]*100, digits=2))%")
println("  Ensemble Majority:            F1=$(round(approach1_results["ensemble_majority"][1]*100, digits=2))%, Acc=$(round(approach1_results["ensemble_majority"][2]*100, digits=2))%")
println("  Ensemble Weighted:            F1=$(round(approach1_results["ensemble_weighted"][1]*100, digits=2))%, Acc=$(round(approach1_results["ensemble_weighted"][2]*100, digits=2))%")

println("\n🏆 APPROACH 2: 50/50 SPLIT")
println("-"^70)
println("  ANN ($best_topology_ann_50):     F1=$(round(approach2_results["ann"][2]*100, digits=2))%, Acc=$(round(approach2_results["ann"][3]*100, digits=2))%")
println("  SVM ($best_desc_svm_50):         F1=$(round(approach2_results["svm"][2]*100, digits=2))%, Acc=$(round(approach2_results["svm"][3]*100, digits=2))%")
println("  Decision Tree (d=$best_desc_tree_50): F1=$(round(approach2_results["tree"][2]*100, digits=2))%, Acc=$(round(approach2_results["tree"][3]*100, digits=2))%")
println("  kNN (k=$best_k_knn_50):          F1=$(round(approach2_results["knn"][2]*100, digits=2))%, Acc=$(round(approach2_results["knn"][3]*100, digits=2))%")
println("  Ensemble Majority:               F1=$(round(approach2_results["ensemble_majority"][1]*100, digits=2))%, Acc=$(round(approach2_results["ensemble_majority"][2]*100, digits=2))%")
println("  Ensemble Weighted:               F1=$(round(approach2_results["ensemble_weighted"][1]*100, digits=2))%, Acc=$(round(approach2_results["ensemble_weighted"][2]*100, digits=2))%")

println("\n🏆 APPROACH 3: PCA + 80/20 SPLIT ($n_components components, $(round(cumulative_variance[n_components]*100, digits=1))% variance)")
println("-"^70)
println("  ANN ($best_topology_ann_pca):     F1=$(round(approach3_results["ann"][2]*100, digits=2))%, Acc=$(round(approach3_results["ann"][3]*100, digits=2))%")
println("  SVM ($best_desc_svm_pca):         F1=$(round(approach3_results["svm"][2]*100, digits=2))%, Acc=$(round(approach3_results["svm"][3]*100, digits=2))%")
println("  Decision Tree (d=$best_desc_tree_pca): F1=$(round(approach3_results["tree"][2]*100, digits=2))%, Acc=$(round(approach3_results["tree"][3]*100, digits=2))%")
println("  kNN (k=$best_k_knn_pca):          F1=$(round(approach3_results["knn"][2]*100, digits=2))%, Acc=$(round(approach3_results["knn"][3]*100, digits=2))%")
println("  Ensemble Majority:                 F1=$(round(approach3_results["ensemble_majority"][1]*100, digits=2))%, Acc=$(round(approach3_results["ensemble_majority"][2]*100, digits=2))%")
println("  Ensemble Weighted:                 F1=$(round(approach3_results["ensemble_weighted"][1]*100, digits=2))%, Acc=$(round(approach3_results["ensemble_weighted"][2]*100, digits=2))%")

# Determine best overall across all approaches
all_results = [
    ("Approach 1 - ANN", approach1_results["ann"][2]),
    ("Approach 1 - SVM", approach1_results["svm"][2]),
    ("Approach 1 - Tree", approach1_results["tree"][2]),
    ("Approach 1 - kNN", approach1_results["knn"][2]),
    ("Approach 1 - Ensemble Majority", approach1_results["ensemble_majority"][1]),
    ("Approach 1 - Ensemble Weighted", approach1_results["ensemble_weighted"][1]),
    ("Approach 2 - ANN", approach2_results["ann"][2]),
    ("Approach 2 - SVM", approach2_results["svm"][2]),
    ("Approach 2 - Tree", approach2_results["tree"][2]),
    ("Approach 2 - kNN", approach2_results["knn"][2]),
    ("Approach 2 - Ensemble Majority", approach2_results["ensemble_majority"][1]),
    ("Approach 2 - Ensemble Weighted", approach2_results["ensemble_weighted"][1]),
    ("Approach 3 - ANN (PCA)", approach3_results["ann"][2]),
    ("Approach 3 - SVM (PCA)", approach3_results["svm"][2]),
    ("Approach 3 - Tree (PCA)", approach3_results["tree"][2]),
    ("Approach 3 - kNN (PCA)", approach3_results["knn"][2]),
    ("Approach 3 - Ensemble Majority (PCA)", approach3_results["ensemble_majority"][1]),
    ("Approach 3 - Ensemble Weighted (PCA)", approach3_results["ensemble_weighted"][1])
]

best_overall_all = sort(all_results, by=x->x[2], rev=true)[1]

println("\n" * "="^70)
println("🎯 BEST OVERALL MODEL ACROSS ALL APPROACHES")
println("="^70)
println("  $(best_overall_all[1])")
println("  Test F1 Score: $(round(best_overall_all[2]*100, digits=2))%")

println("\n" * "="^70)
println("✅ ALL EXPERIMENTS COMPLETE!")
println("="^70)
println("\n📋 SUMMARY:")
println("  ✅ 3 Different Approaches tested (as required):")
println("     - Approach 1: Real dataset with 80/20 split")
println("     - Approach 2: Real dataset with 50/50 split")
println("     - Approach 3: PCA dimensionality reduction + 80/20 split")
println("  ✅ For each approach, tested 4 ML algorithms:")
println("     - ANN: 8 topologies (1-2 hidden layers)")
println("     - SVM: 8 configurations (different kernels and C values)")
println("     - Decision Tree: 6 maximum depths")
println("     - kNN: 6 different k values")
println("  ✅ Ensemble methods combining 3 models (ANN + Tree + kNN):")
println("     - Majority Voting")
println("     - Weighted Voting")
println("  ✅ Proper Train/Test split - no data leakage")
println("  ✅ 3-fold stratified cross-validation for model selection")
println("  ✅ Random seed (42) set for reproducibility")
println("  ✅ Confusion matrices for all evaluations")
println("  ✅ All code follows course methodology using modelCrossValidation")
println("="^70)