using DelimitedFiles
using LsqFit
using Plots
using LaTeXStrings
using Printf
Plots.pythonplot()
default(fontfamily="Monaco")
e1 = readdlm("e1.txt", Float64)
e2 = readdlm("e2.txt", Float64)
data = e2 - e1


x = collect(4:2:16) 
y = vec(data)
model(x, p) = p[1] .+ p[2] ./ x .+ p[3] ./ (x .^ 2) 


p0 = [rand(), rand(), rand(), rand(), rand(), rand(), rand()]

fit = curve_fit(model, x, y, p0)

title_str = @sprintf("\$c + \\frac{b}{L} + \\frac{a}{L^2}\$, c = %.2f, b = %.2f, a = %.2f", fit.param[1], fit.param[2], fit.param[3])
title_str = LaTeXString(title_str)

scatter(x, y, label="gap", xticks=4:2:16, title=title_str)


p = plot!(collect(4:2:32), model(collect(4:2:32), fit.param), label=L"c+\frac{b}{L}+\frac{a}{L^2}")


savefig(p, "a2.png")
