# ============================================================================
#                    PREPROCESSING UTILITIES
# ============================================================================

module PreprocessingUtils

export create_risk_classes, preprocess_multiclass

using Dates
using Statistics
using StatsBase
using DataFrames  

function create_risk_classes(dataframe, target_col::String="Is Fraudulent")
    """
    Create 3 risk classes from binary fraud label + features
    """
    data = copy(dataframe)
    
    println("\n📊 Calculating risk signals...")
    
    # Time risk
    if "Transaction Date" in names(data)
        try
            parsed_dates = DateTime.(data[!, "Transaction Date"], dateformat"y-m-d H:M:S")
            hours = hour.(parsed_dates)
            data.Hour_Risk = [h in [0,1,2,3,4,5,23] ? 1 : 0 for h in hours]
        catch
            data.Hour_Risk = zeros(Int, size(data, 1))
        end
    else
        data.Hour_Risk = zeros(Int, size(data, 1))
    end
    
    # Amount risk
    if "Transaction Amount" in names(data)
        p90 = quantile(data[!, "Transaction Amount"], 0.90)
        data.Amount_Risk = [amt > p90 ? 1 : 0 for amt in data[!, "Transaction Amount"]]
    else
        data.Amount_Risk = zeros(Int, size(data, 1))
    end
    
    # Account age risk
    if "Account Age Days" in names(data)
        data.Account_Risk = [age < 30 ? 1 : 0 for age in data[!, "Account Age Days"]]
    else
        data.Account_Risk = zeros(Int, size(data, 1))
    end
    
    data.Total_Risk = data.Hour_Risk .+ data.Amount_Risk .+ data.Account_Risk
    data.Risk_Class = zeros(Int, size(data, 1))
    
    is_fraud = data[!, target_col] .== 1
    
    data.Risk_Class[is_fraud .& (data.Total_Risk .>= 2)] .= 2
    data.Risk_Class[is_fraud .& (data.Total_Risk .< 2)] .= 1
    data.Risk_Class[(.!is_fraud) .& (data.Total_Risk .>= 2)] .= 1
    data.Risk_Class[(.!is_fraud) .& (data.Total_Risk .< 2)] .= 0
    
    println("\n✅ 3-Class distribution:")
    class_counts = countmap(data.Risk_Class)
    total = size(data, 1)
    
    for class_id in 0:2
        count = get(class_counts, class_id, 0)
        pct = round(count/total*100, digits=1)
        class_name = ["LEGITTIMO", "SOSPETTO", "FRAUDOLENTO"][class_id+1]
        println("  Class $class_id ($class_name): $count ($pct%)")
    end
    
    return data
end

function preprocess_multiclass(dataframe, target_col::String="Is Fraudulent")
    """
    Preprocess dataframe for multiclass fraud detection
    """
    data = copy(dataframe)
    
    # Time features
    if "Transaction Date" in names(data)
        try
            parsed_dates = DateTime.(data[!, "Transaction Date"], dateformat"y-m-d H:M:S")
            data.Transaction_Hour = hour.(parsed_dates)
            data.Is_Night = [h < 6 ? 1.0 : 0.0 for h in data.Transaction_Hour]
        catch e
            println("  ⚠ Error processing dates: $e")
        end
    end
    
    # Imputation
    for col in ["Transaction Amount", "Account Age Days"]
        if col in names(data) && any(ismissing, data[!, col])
            replace!(data[!, col], missing => median(skipmissing(data[!, col])))
        end
    end
    
    # Feature engineering
    if "Transaction Amount" in names(data) && "Account Age Days" in names(data)
        data.Amount_per_AccountAge = data[!, "Transaction Amount"] ./ (data[!, "Account Age Days"] .+ 1.0)
    end
    
    if "Transaction Amount" in names(data)
        p95 = quantile(data[!, "Transaction Amount"], 0.95)
        data.High_Value_Flag = [amt > p95 ? 1.0 : 0.0 for amt in data[!, "Transaction Amount"]]
    end
    
    if "Account Age Days" in names(data)
        data.New_Account_Flag = [age < 30 ? 1.0 : 0.0 for age in data[!, "Account Age Days"]]
    end
    
    # Drop unnecessary columns
    cols_to_drop = ["Transaction ID", "Customer ID", "Transaction Date", 
                    "IP Address", "Shipping Address", "Billing Address", "Customer Location",
                    "Quantity", "Customer Age", "Payment Method", "Product Category",
                    target_col, "Hour_Risk", "Amount_Risk", "Account_Risk", "Total_Risk",
                    "Device Used"]
    select!(data, Not(intersect(names(data), cols_to_drop)))
    
    return data
end

end # module