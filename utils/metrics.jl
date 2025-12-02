# ============================================================================
#                    METRICS UTILITIES
# ============================================================================

module MetricsUtils

export confusionMatrixMulticlass, displayConfusionMatrix, crossvalidation_multiclass

using Statistics
using LinearAlgebra
using Random

function confusionMatrixMulticlass(predictions::AbstractArray{Int,1}, targets::AbstractArray{Int,1}, numClasses::Int=3)
    """
    Compute confusion matrix and metrics for multiclass classification
    """
    cm = zeros(Int, numClasses, numClasses)
    
    for i in 1:length(predictions)
        cm[targets[i]+1, predictions[i]+1] += 1
    end
    
    accuracy = sum(diag(cm)) / sum(cm)
    
    class_metrics = []
    class_names = ["Legittimo", "Sospetto", "Fraudolento"]
    
    for class in 1:numClasses
        TP = cm[class, class]
        FP = sum(cm[:, class]) - TP
        FN = sum(cm[class, :]) - TP
        TN = sum(cm) - TP - FP - FN
        
        recall = TP / (TP + FN + 1e-10)
        precision = TP / (TP + FP + 1e-10)
        f1 = 2 * recall * precision / (recall + precision + 1e-10)
        
        push!(class_metrics, (class_names[class], recall, precision, f1))
    end
    
    macro_f1 = mean([m[4] for m in class_metrics])
    
    class_counts = [sum(targets .== (i-1)) for i in 1:numClasses]
    weights = class_counts ./ sum(class_counts)
    weighted_f1 = sum([class_metrics[i][4] * weights[i] for i in 1:numClasses])
    
    return cm, accuracy, class_metrics, macro_f1, weighted_f1
end

function displayConfusionMatrix(cm, class_names=["Legittimo", "Sospetto", "Fraudolento"])
    """
    Display confusion matrix in a readable format
    """
    println("\n📊 CONFUSION MATRIX (3x3):")
    println("="^70)
    println("                  Predicted")
    println("              Legit    Sospect   Fraud")
    println("-"^70)
    for (i, actual) in enumerate(class_names)
        println("$(rpad(actual, 12)) |  $(lpad(cm[i,1], 6))   $(lpad(cm[i,2], 6))   $(lpad(cm[i,3], 6))")
    end
    println("="^70)
end

function crossvalidation_multiclass(targets::AbstractArray{Int,1}, k::Int64)
    """
    Stratified cross-validation for multiclass targets
    """
    indices = zeros(Int, length(targets))
    
    for class_label in unique(targets)
        class_mask = targets .== class_label
        n_class = sum(class_mask)
        
        class_indices = repeat(1:k, Int(ceil(n_class/k)))[1:n_class]
        shuffle!(class_indices)
        
        indices[class_mask] .= class_indices
    end
    
    return indices
end

end # module