# ============================================================================
#                    kNN UTILITIES
# ============================================================================

module KNNUtils

export knnPredict, knnCrossValidation

using NearestNeighbors
using StatsBase
using Random

include("preprocessing.jl")
using .PreprocessingUtils

include("metrics.jl")
using .MetricsUtils

function knnPredict(trainInputs::AbstractArray{<:Real,2}, trainTargets::AbstractArray{Int,1},
                   testInputs::AbstractArray{<:Real,2}, k::Int)
    """
    kNN prediction using KDTree for efficiency
    """
    kdtree = KDTree(trainInputs')
    idxs, dists = knn(kdtree, testInputs', k, true)
    
    predictions = zeros(Int, size(testInputs, 1))
    
    for i in 1:size(testInputs, 1)
        neighbor_labels = trainTargets[idxs[i]]
        predictions[i] = mode(neighbor_labels)
    end
    
    return predictions
end

function knnCrossValidation(
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,1}},
        crossValidationIndices::Array{Int64,1};
        k::Int=5,
        verbose::Bool=true)
    """
    Cross-validation for kNN parameter selection
    """
    inputs, targets = dataset
    numFolds = maximum(crossValidationIndices)
    
    all_predictions = []
    all_targets = []
    
    if verbose
        println("  Testing kNN: k=$k")
    end
    
    for fold in 1:numFolds
        testMask = crossValidationIndices .== fold
        trainMask = .!testMask
        
        trainInputs = inputs[trainMask, :]
        trainTargets = targets[trainMask]
        testInputs = inputs[testMask, :]
        testTargets = targets[testMask]
        
        # Normalize (CRITICAL for kNN!)
        normParams = calculateMinMaxNormalizationParameters(trainInputs)
        trainInputsNorm = normalizeMinMax(trainInputs, normParams)
        testInputsNorm = normalizeMinMax(testInputs, normParams)
        
        # Predict with kNN
        predictions = knnPredict(trainInputsNorm, trainTargets, testInputsNorm, k)
        
        push!(all_predictions, predictions)
        push!(all_targets, testTargets)
    end
    
    all_preds_combined = vcat(all_predictions...)
    all_targets_combined = vcat(all_targets...)
    
    cm, acc, class_metrics, macro_f1, weighted_f1 = 
        confusionMatrixMulticlass(all_preds_combined, all_targets_combined, 3)
    
    if verbose
        println("    → Macro F1: $(round(macro_f1*100, digits=2))% | Weighted F1: $(round(weighted_f1*100, digits=2))%")
    end
    
    return macro_f1, weighted_f1, cm, class_metrics
end

end # module