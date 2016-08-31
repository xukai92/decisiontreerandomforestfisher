# Gini impurity
function IG(fs)
  return sum(map(x -> x * (1 - x), fs))
end

# Cross entropy
function CE(fs)
  return sum(map(x -> -x * log(x),fs))
end

# Find most
function findmost(da::DataFrame)
  d = Dict({t => 0 for t in Set(da[:Type])})
  for t in da[:Type]
    d[t] += 1
  end
  return find_largest_value(d)
end

function findmost(a::Array)
  d = Dict({t => 0 for t in Set(a)})
  for t in a
    d[t] += 1
  end
  return find_largest_value(d)
end

function find_largest_value(a::Dict)
  most_type = 0
  most = 0
  for tp in keys(a)
    if a[tp] > most
      most_type = tp
      most = a[tp]
    end
  end
  return most_type
end
