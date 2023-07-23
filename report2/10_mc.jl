using Plots
using DelimitedFiles
using Statistics
using LaTeXStrings
Plots.pythonplot()
default(fontfamily="Monaco")
x = readdlm("vecr.txt")

y_values = []

for i in 1:10
    filename = string(i) * "/gofr.txt"
    y = readdlm(filename)
    push!(y_values, y)
end

std_devs = map(std, zip(y_values...))

scatter(x, std_devs, xlabel=L"r", ylabel="Standard deviation of \$g(r)\$", label="12800 Nsample")
savefig("10_samples.png")
