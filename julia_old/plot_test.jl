### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 9d4fe173-5516-42e4-83eb-365cdb6f5214
begin
    import Pkg
    Pkg.activate(".")
	
    import DarkMode
    DarkMode.CSSDarkMode("monokai", darkenPluto=true)
end

# ╔═╡ c298125e-e227-41fd-a3da-5883db022c08
begin
	import JSServe
	using WGLMakie
	JSServe.Page()
end

# ╔═╡ d9f9d52c-100e-11ec-0054-1bedb63af96b
scatter(rand(20), rand(20))

# ╔═╡ 3c7007dd-3755-4855-be2f-ffe91102356c
begin
	N = 60
	function xy_data(x, y)
		r = sqrt(x^2 + y^2)
		r == 0.0 ? 1f0 : (sin(r)/r)
	end
	l = range(-10, stop = 10, length = N)
	z = Float32[xy_data(x, y) for x in l, y in l]
	surface(
		-1..1, -1..1, z,
		colormap = :Spectral
	)
end

# ╔═╡ Cell order:
# ╟─9d4fe173-5516-42e4-83eb-365cdb6f5214
# ╠═c298125e-e227-41fd-a3da-5883db022c08
# ╠═d9f9d52c-100e-11ec-0054-1bedb63af96b
# ╠═3c7007dd-3755-4855-be2f-ffe91102356c
