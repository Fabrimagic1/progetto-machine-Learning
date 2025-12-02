# ============================================================================
#                    DECISION TREE UTILITIES
# ============================================================================

module TreeUtils

export decisionTreeCrossValidation

using DecisionTree
using Random

include("preprocessing.jl")
using .PreprocessingUtils

include("metrics.jl")
using .MetricsUtils

function decisionTreeCrossValidation(
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,1}},
        crossValidationIndices::Array{Int64,1};
        max_depth::Int=-1,
        min_samples_split::Int=2,
        verbose::Bool=true)
    """
    Cross-validation for Decision Tree parameter selection
    """
    inputs, targets = dataset
    numFolds = maximum(crossValidationIndices)
    
    all_predictions = []
    all_targets = []
    
    if verbose
        depth_str = max_depth == -1 ? "unlimited" : string(max_depth)
        println("  Testing Decision Tree: max_depth=$depth_str, min_samples_split=$min_samples_split")
    end
    
    for fold in 1:numFolds
        testMask = crossValidationIndices .== fold
        trainMask = .!testMask
        
        trainInputs = inputs[trainMask, :]
        trainTargets = targets[trainMask]
        testInputs = inputs[testMask, :]
        testTargets = targets[testMask]
        
        # Normalize (for consistency, though trees don't strictly need it)
        normParams = calculateMinMaxNormalizationParameters(trainInputs)
        trainInputsNorm = normalizeMinMax(trainInputs, normParams)
        testInputsNorm = normalizeMinMax(testInputs, normParams)
        
        # Train Decision Tree
        if max_depth == -1
            model = DecisionTreeClassifier(min_samples_split=min_samples_split)
        else
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        end
        
        DecisionTree.fit!(model, trainInputsNorm, trainTargets)
        
        # Predict
        predictions = DecisionTree.predict(model, testInputsNorm)
        
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