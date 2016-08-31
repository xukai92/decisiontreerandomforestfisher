# Read dataset
df = readtable("Fisher.csv")

# Split dataset
training = df[1:100, :]
validation = df[101:130, :]
testing = df[131:150, :]
