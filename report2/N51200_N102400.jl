using Plots
using DelimitedFiles
using Statistics
using LaTeXStrings
Plots.pythonplot()
default(fontfamily="Monaco")
x = readdlm("vecr.txt")

y_values = []

for i in 1:10
    filename = string(i) * "/gofr_51200_Nsample.txt"
    y = readdlm(filename)
    push!(y_values, y)
end

std_devs = (1 / sqrt(2)) * map(std, zip(y_values...))

p = scatter(x, std_devs, xlabel=L"r", ylabel="Standard deviation of \$g(r)\$", color=:blue, label="Scaled 51200 Nsample")

y_values = []

for i in 1:10
    filename = string(i) * "/gofr_102400_Nsample.txt"
    y = readdlm(filename)
    push!(y_values, y)
end

std_devs = map(std, zip(y_values...))
scatter!(p, x, std_devs, color=:red, label="102400 Nsample")
savefig(p, "Nsamples51200_102400.png")
