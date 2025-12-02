# ============================================================================
#                    SVM UTILITIES
# ============================================================================

module SVMUtils

export svmCrossValidation

using LIBSVM
using Random

include("preprocessing.jl")
using .PreprocessingUtils

include("metrics.jl")
using .MetricsUtils

function svmCrossValidation(
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,1}},
        crossValidationIndices::Array{Int64,1};
        kernel::Symbol=:linear,
        C::Float64=1.0,
        gamma::Union{Float64, Symbol}=:auto,
        degree::Int=3,
        verbose::Bool=true)
    """
    Cross-validation for SVM parameter selection
    """
    inputs, targets = dataset
    numFolds = maximum(crossValidationIndices)
    
    all_predictions = []
    all_targets = []
    
    if verbose
        println("  Testing SVM: kernel=$kernel, C=$C, gamma=$gamma, degree=$degree")
    end
    
    for fold in 1:numFolds
        testMask = crossValidationIndices .== fold
        trainMask = .!testMask
        
        trainInputs = inputs[trainMask, :]
        trainTargets = targets[trainMask]
        testInputs = inputs[testMask, :]
        testTargets = targets[testMask]
        
        # Normalize
        normParams = calculateMinMaxNormalizationParameters(trainInputs)
        trainInputsNorm = normalizeMinMax(trainInputs, normParams)
        testInputsNorm = normalizeMinMax(testInputs, normParams)
        
        # Convert targets to Float64 (required by LIBSVM)
        trainTargets_float = Float64.(trainTargets)
        
        # Set gamma for RBF kernel
        if kernel == :rbf
            if gamma == :auto
                gamma_val = 1.0 / size(trainInputsNorm, 2)
            else
                gamma_val = gamma
            end
        else
            gamma_val = 0.0
        end
        
        # Train SVM
        if kernel == :linear
            model = svmtrain(trainInputsNorm', trainTargets_float, 
                           kernel=LIBSVM.Kernel.Linear, cost=C, verbose=false)
        elseif kernel == :rbf
            model = svmtrain(trainInputsNorm', trainTargets_float,
                           kernel=LIBSVM.Kernel.RadialBasis, cost=C, gamma=gamma_val, verbose=false)
        elseif kernel == :polynomial
            model = svmtrain(trainInputsNorm', trainTargets_float,
                           kernel=LIBSVM.Kernel.Polynomial, cost=C, degree=degree, verbose=false)
        else
            error("Unknown kernel: $kernel")
        end
        
        # Predict
        predictions, _ = svmpredict(model, testInputsNorm')
        test_predictions = Int.(predictions)
        
        push!(all_predictions, test_predictions)
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