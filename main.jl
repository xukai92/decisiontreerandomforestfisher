# Load packages
using DataArrays, DataFrames
using Optim

include("helper.jl")
include("data.jl")
include("dt.jl")
include("rf.jl")

# Decision tree
tree = build_tree(training, 5, IG)

predictions = [traverse(validation[i, :], tree) for i = 1:size(validation, 1)]
targets = [Int64(validation[i, :][:Type]) for i = 1:size(validation, 1)]

acc = sum(predictions .== targets) / size(validation, 1)

# Random forest
forest = build_forest(training, 5, IG, 15, 30)

predictions = [walk(validation[i, :], forest) for i = 1:size(validation, 1)]
targets = [Int64(validation[i, :][:Type]) for i = 1:size(validation, 1)]

acc = sum(predictions .== targets) / size(validation, 1)
