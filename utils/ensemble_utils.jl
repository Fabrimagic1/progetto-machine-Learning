# ============================================================================
#                    ENSEMBLE UTILITIES
# ============================================================================

module EnsembleUtils

export majorityVoting, weightedVoting

using StatsBase

function majorityVoting(predictions::Vector{Vector{Int}})
    """
    Majority voting: each model votes, winner takes all
    """
    n_samples = length(predictions[1])
    n_models = length(predictions)
    
    ensemble_predictions = zeros(Int, n_samples)
    
    for i in 1:n_samples
        votes = [pred[i] for pred in predictions]
        ensemble_predictions[i] = mode(votes)
    end
    
    return ensemble_predictions
end

function weightedVoting(predictions::Vector{Vector{Int}}, weights::Vector{Float64})
    """
    Weighted voting: models vote with different weights based on performance
    """
    n_samples = length(predictions[1])
    n_models = length(predictions)
    n_classes = 3
    
    ensemble_predictions = zeros(Int, n_samples)
    
    for i in 1:n_samples
        # Count weighted votes for each class
        class_scores = zeros(Float64, n_classes)
        
        for j in 1:n_models
            class_pred = predictions[j][i]
            class_scores[class_pred + 1] += weights[j]
        end
        
        # Select class with highest weighted vote
        ensemble_predictions[i] = argmax(class_scores) - 1
    end
    
    return ensemble_predictions
end

end # module