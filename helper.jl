# Gini impurity
function IG(fs)
  return sum(map(x -> x * (1 - x), fs))
end

# Cross entropy
function CE(fs)
  return sum(map(x -> -x * log(x),fs))
end
