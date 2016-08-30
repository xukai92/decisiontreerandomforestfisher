# Load packages
using DataArrays, DataFrames
using Optim

# Read dataset
df = readtable("Fisher.csv")

# Split dataset
training = df[1:100, :]
testing = df[101:150, :]

# Gini impurity
function IG(fs)
  return sum(map(x -> x * (1 - x), fs))
end

# Split by threshold
function split_by_th(df, by, threshold)
  return df[df[by] .> threshold, :], df[df[by] .<= threshold, :]
end

# To percent
function to_precent(ts)
  return counts(ts, 0:2) / length(ts)
end

# total IG after aplit
function totoal_IG_after_split(df, by, threshold)
  left, right = split_by_th(df, by, threshold)
  return IG(to_precent(left[:Type])) + IG(to_precent(right[:Type]))
end

# find optimum
function find_optimum_for(df, var)
  result = optimize(
    th -> totoal_IG_after_split(df, var, th),
    min(df[var]...),
    max(df[var]...))
  return Optim.minimizer(result)
end

find_optimum_for(training, :PW)
