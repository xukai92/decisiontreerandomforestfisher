# Split by threshold
function split_by_th(df, by, threshold)
  return df[df[by] .> threshold, :], df[df[by] .<= threshold, :]
end

# To percent
function to_precent(ts)
  return counts(ts, 0:2) / length(ts)
end

# Total IG after aplit
function totoal_impurity_after_split(df, by, threshold, impurity_func)
  left, right = split_by_th(df, by, threshold)
  return impurity_func(to_precent(left[:Type])) + impurity_func(to_precent(right[:Type]))
end

# Find optimum for a variable
function find_optimum_for(df, var, impurity_func)
  x_lower, x_upper = min(df[var]...), max(df[var]...)
  if x_lower == x_upper
    return totoal_impurity_after_split(df, var, x_lower, impurity_func), x_lower
  else
    result = optimize(
      th -> totoal_impurity_after_split(df, var, th, impurity_func),
      x_lower,
      x_upper)
    return Optim.minimum(result), Optim.minimizer(result)
  end
end

# Build tree
function build_tree(df, early_stop_th, impurity_func)
  headers = keys(df)[2:5]
  if size(df, 1) < early_stop_th
    return findmost(df)
  else
    split_choice = [find_optimum_for(df, h, impurity_func) for h in headers]
    split_idx = findmax(map(x -> x[1], split_choice))[2]
    split_by = headers[split_idx], split_choice[split_idx][2]
    left, right = split_by_th(df, split_by...)
    if size(left, 1) == 0 || size(right, 1) == 0
      return findmost(df)
    else
      return ((split_by), build_tree(left, early_stop_th, impurity_func), build_tree(right, early_stop_th, impurity_func))
    end
  end
end

# Traverse
function traverse(df, tree)
  if length(tree) == 1
    return tree
  else
    by, threshold = tree[1]
    if Int64(df[by]) > threshold
      return traverse(df, tree[2])
    else
      return traverse(df, tree[3])
    end
  end
end

Base.convert(::Type{Int64}, x::DataArrays.DataArray{Int64,1}) = x[1]

# Training
function train(df, early_stop_th)
  # Build tree
  tree = build_tree(training, early_stop_th, IG)

  # Validation
  predictions = [traverse(validation[i, :], tree) for i = 1:size(validation, 1)]
  targets = [Int64(validation[i, :][:Type]) for i = 1:size(validation, 1)]

  acc = sum(predictions .== targets) / size(validation, 1)
  return tree, acc
end

# Cross validation
function cv()
  acc = 0
  early_stop_th = 0
  for th = 2:10
    new_tree, new_acc = train(training, th)
    if new_acc > acc
      tree, acc = new_tree, new_acc
      early_stop_th = th
    end
  end
  println("Accuracy\t:\t", acc, "\nEarly stop\t:\t", early_stop_th)

  # Testing
  predictions = [traverse(testing[i, :], tree) for i = 1:size(testing, 1)]
  targets = [Int64(testing[i, :][:Type]) for i = 1:size(testing, 1)]

  acc = sum(predictions .== targets) / size(testing, 1)
end
