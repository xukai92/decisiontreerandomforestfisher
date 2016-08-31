include("dt.jl")

# Random subset
Base.rand(df::DataFrame, n::Int64) = df[rand(1:size(df, 1), n), :]

# Build forest
function build_forest(df, early_stop_th, impurity_func, f_num, f_size)
  forest = Array{Any}(f_num)
  for i = 1:f_num
    forest[i] = build_tree(rand(df, f_size), early_stop_th, impurity_func)
  end
  return forest
end

# Walk
function walk(df, forest)
  return findmost([traverse(df, tree) for tree in forest])
end
