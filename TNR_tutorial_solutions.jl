# TNR tutorial — plain-script version of the solutions notebook
# MMBW02, Isaac Newton Institute, 30 October 2026

# # Tensor Renormalization — hands-on tutorial
# 
# **European Tensor Networks School (MMBW02) · Isaac Newton Institute · Friday 30 October 2026, 14:00–16:00**
# 
# Atsushi Ueda — Ghent University · built on [`TNRKit.jl`](https://github.com/QuantumKitHub/TNRKit.jl)
# 
# ---
# 
# ### What you will have done by 16:00
# 
# | | | time |
# |---|---|---|
# | **0** | Warm-up: build the Ising tensor by hand, and from the library | 15 min |
# | **1** | TRG free energy vs Onsager — and the `scalefactor` trap | 20 min |
# | **2** | TRG vs BTRG vs HOTRG vs ATRG: accuracy at equal cost | 20 min |
# | **3** | Watch TRG break: the CFT spectrum drifts | 20 min |
# | **4** | LoopTNR: a spectrum that stays put. Extract $c$ and $\Delta_i$ | 25 min |
# | **5** | Play with the geometry: shapes, wide tubes, conformal spins | 15 min |
# | **6** | Choose your own model | leftover |
# 
# ### How to use this notebook
# 
# Cells marked **▸ YOUR TURN** have something missing. Everything else runs as-is.
# If you get stuck for more than ~3 minutes, open `TNR_tutorial_solutions.ipynb` — you will learn more
# from reading a working answer and modifying it than from fighting a type error.
# 
# > **Before you start:** `using TNRKit` takes 5–10 minutes the first time. Run the next cell
# > *now*, then read the warm-up while it compiles.

using TNRKit, TensorKit, KrylovKit
using Plots, Printf, LinearAlgebra

@show ising_βc                    # = 0.5*log(1+sqrt(2)) = 0.4406867935
println("\nTNRKit loaded. Exported names, for reference when a signature surprises you:")
println(sort(string.(names(TNRKit))))

# Exact Onsager free energy density, in case TNRKit doesn't export one.
# f = -(1/beta) * [ log(2)/2 + (1/2pi) * int_0^pi dtheta log( cosh(2b)^2
#                   + (1/k) sqrt(1 + k^2 - 2k cos(2 theta)) ) ],   k = 1/sinh(2b)^2
using QuadGK
function f_exact(beta)
    k = 1 / sinh(2beta)^2
    integrand(th) = log(cosh(2beta)^2 + (1/k)*sqrt(1 + k^2 - 2k*cos(2th)))
    val, _ = quadgk(integrand, 0, pi)
    return -(log(2)/2 + val/(2pi)) / beta
end

@printf("beta_c       = %.10f\n", ising_βc)
@printf("f_exact(b_c) = %.10f\n", f_exact(ising_βc))   # expect -2.1096511446

# ---
# ## 0 · Warm-up: where does the tensor come from?
# 
# In the lecture we derived, from the high-temperature (character) expansion,
# 
# $$T_{ijkl} = 2\,(\tanh\beta)^{(i+j+k+l)/2}\;\delta_{i+j+k+l \equiv 0 \ (\mathrm{mod}\ 2)}$$
# 
# with $(i,j,k,l)$ the link variables on the **left, bottom, top, right** bonds of a site, and
# 
# $$Z = \cosh(\beta)^{2V}\;\mathrm{tTr}\Big[\bigotimes_x T^{(x)}\Big].$$
# 
# > The factor $2$ *inside* $T$ is already the on-site spin sum
# > $\sum_{\sigma=\pm1}\sigma^{N_x} = 2\,\delta_{N_x \bmod 2,\,0}$, so there is **no** extra
# > $2^V$ in front. (If you carry both you overcount $Z$ by $2^V$, which shifts $f$ by
# > $-\ln 2/\beta \approx -1.57$ at $\beta_c$ — a very visible bug.)
# That leg ordering is TNRKit's convention for every 2D partition-function tensor —
# **write it on a sticky note.** In TensorKit terms a scheme wants a map
# 
# $$T : V_1 \otimes V_2 \longleftarrow V_3 \otimes V_4 .$$
# 
# ### ▸ YOUR TURN
# Build that tensor by hand as a plain (`Trivial`, i.e. symmetry-free) `TensorMap` and check it
# against the library's `classical_ising`.

function my_ising_tensor(β)
    V = ComplexSpace(2)
    T = zeros(Float64, V ⊗ V ← V ⊗ V)
    t = tanh(β)
    for i in 0:1, j in 0:1, k in 0:1, l in 0:1
        if (i + j + k + l) % 2 == 0
            T[i+1, j+1, k+1, l+1] = 2 * t^((i + j + k + l) / 2)
        end
    end
    return T
end

T_mine = my_ising_tensor(ising_βc)
T_lib  = classical_ising(Trivial, ising_βc)
@show norm(T_mine - T_lib) / norm(T_lib)
# They may differ by an overall normalisation convention — if so, compare T/norm(T).
@show norm(T_mine/norm(T_mine) - T_lib/norm(T_lib))

# **Discussion (2 min, out loud with your neighbour).** Only 8 of the 16 entries are non-zero.
# Which symmetry is that, and *where did it come from* in the derivation?
# 
# Now the same tensor, but with the $\mathbb{Z}_2$ symmetry made explicit to TensorKit:

T_triv = classical_ising(Trivial, ising_βc)
T_z2   = classical_ising(Z2Irrep, ising_βc)

println("Trivial : ", space(T_triv))
println("Z2Irrep : ", space(T_z2))
println()
println("blocks of the Z2-symmetric tensor:")
for (c, b) in blocks(T_z2)
    println("  sector ", c, "  →  size ", size(b))
end

# The $\mathbb{Z}_2$ version stores the *same* information as two small blocks instead of one
# $4\times4$ matrix. Every SVD inside the algorithm then splits into two smaller SVDs.
# At $\chi=2$ this is pointless; at $\chi=32$ it is roughly a factor 3–4 in wall time.

# ---
# ## 1 · TRG and the free energy
# 
# The physics is **not** in the final tensor. It is in the normalisation factors
# $g^{(j)}$ thrown away at each step:
# 
# $$\frac{\ln Z(b^n,b^n)}{b^{2n}} = \sum_{j=1}^{n} \frac{\ln g^{(j)}}{b^{2j}} .$$
# 
# `run!` returns exactly those factors; `free_energy` does the sum.

scheme = TRG(classical_ising(Z2Irrep, ising_βc))
data   = run!(scheme, truncrank(16), maxiter(25))

f_trg = free_energy(data, ising_βc)          # TRG: b = √2, so scalefactor = 2 (the default)
@printf("TRG   f = %.12f\n", f_trg)
@printf("exact f = %.12f\n", f_exact(ising_βc))
@printf("rel err = %.3e\n", abs(f_trg - f_exact(ising_βc)) / abs(f_exact(ising_βc)))

# ### ▸ YOUR TURN — the `scalefactor` trap
# 
# `free_energy(data, β; scalefactor, initial_size)` needs $b^2$, the factor by which the
# **area** per tensor grows each step.
# 
# - TRG / BTRG / ATRG: $b=\sqrt2$, so `scalefactor = 2.0`
# - HOTRG: a full step rescales **both** directions by 2, so $b=2$ and `scalefactor = 4.0`
# 
# Deliberately pass the wrong one and see what it costs you.

scheme_h = HOTRG(classical_ising(Z2Irrep, ising_βc))
data_h   = run!(scheme_h, truncrank(16), maxiter(20))

f_right = free_energy(data_h, ising_βc; scalefactor = 4.0)
f_wrong = free_energy(data_h, ising_βc; scalefactor = 2.0)

@printf("HOTRG, scalefactor=4.0 : %.12f   (rel err %.3e)\n",
        f_right, abs(f_right - f_exact(ising_βc))/abs(f_exact(ising_βc)))
@printf("HOTRG, scalefactor=2.0 : %.12f   <-- wrong\n", f_wrong)

# ### ▸ YOUR TURN — convergence in $\chi$
# 
# Plot the relative error of the TRG free energy at $\beta_c$ against $\chi \in \{4,8,12,16,24,32\}$,
# on a log scale. Time each run too.
# 
# **Predict before you plot:** does the error fall like a power of $\chi$, or exponentially?
# And does $\chi=32$ buy you what the extra wall time cost?

chis = [4, 8, 12, 16, 24, 32]
errs  = Float64[]
times = Float64[]
fex   = f_exact(ising_βc)

for χ in chis
    s = TRG(classical_ising(Z2Irrep, ising_βc))
    t = @elapsed d = run!(s, truncrank(χ), maxiter(25); verbosity=0)
    f = free_energy(d, ising_βc)
    push!(errs, abs(f - fex)/abs(fex))
    push!(times, t)
    @printf("χ = %2d :  err = %.3e   (%.2f s)\n", χ, errs[end], t)
end

plot(chis, errs; yscale=:log10, marker=:circle, xlabel="χ", ylabel="relative error",
     label="TRG", title="2D Ising free energy at β_c", lw=2)

# ---
# ## 2 · The TRG family: accuracy at equal cost
# 
# | scheme | idea | cost | $b$ |
# |---|---|---|---|
# | `TRG` | SVD-split $1\to2$, recombine $4\to1$ | $\mathcal{O}(\chi^6)$ | $\sqrt2$ |
# | `BTRG` | same, but keep the bond weights | $\mathcal{O}(\chi^6)$ | $\sqrt2$ |
# | `HOTRG` | merge $2\to1$, squeeze with an isometry | $\mathcal{O}(\chi^7)$ | $2$ |
# | `ATRG` | anisotropic; for large $d$ — **not** a 2D workhorse | $\mathcal{O}(\chi^{2d+1})$ | see note |
# 
# > **Careful with ATRG.** It is *anisotropic*: like HOTRG it renormalises $x$ and $y$
# > alternately, so whether its `scalefactor` is 2 or 4 depends on what counts as one "step" in
# > the version you installed. **Check it** — compare against the exact $f$ at a non-critical
# > $\beta$, where convergence is fast and unambiguous. ATRG is also known to be rather
# > inaccurate in 2D; its reason to exist is $d\ge3$.
# 
# ### ▸ YOUR TURN
# Compare all four at $\chi=16$, at $\beta_c$. Report relative error **and** wall time.
# Then sweep $\beta$ from $0.95\beta_c$ to $1.05\beta_c$ and plot the error curves.

schemes = [
    ("TRG",   T -> TRG(T),   2.0),
    ("BTRG",  T -> BTRG(T),  2.0),
    ("HOTRG", T -> HOTRG(T), 4.0),
    ("ATRG",  T -> ATRG(T),  2.0),
]

fex = f_exact(ising_βc)
println("χ = 16, β = β_c\n")
for (name, ctor, sf) in schemes
    s = ctor(classical_ising(Z2Irrep, ising_βc))
    t = @elapsed d = run!(s, truncrank(16), maxiter(20); verbosity=0)
    f = free_energy(d, ising_βc; scalefactor = sf)
    @printf("%-6s  f = %.10f   rel err = %.3e   (%.2f s)\n",
            name, f, abs(f-fex)/abs(fex), t)
end

βs = range(0.95*ising_βc, 1.05*ising_βc; length = 11)
p = plot(; yscale=:log10, xlabel="β", ylabel="relative error",
         title="Free energy accuracy, χ = 16", legend=:topright, lw=2)

for (name, ctor, sf) in schemes
    e = Float64[]
    for β in βs
        s = ctor(classical_ising(Z2Irrep, β))
        d = run!(s, truncrank(16), maxiter(20); verbosity=0)
        f = free_energy(d, β; scalefactor = sf)
        push!(e, abs(f - f_exact(β))/abs(f_exact(β)))
    end
    plot!(p, βs, e; marker=:circle, label=name)
end
vline!(p, [ising_βc]; ls=:dash, c=:gray, label="β_c")
p

# **What you should see.** The error peaks *at* $\beta_c$ — that is the one place where
# $\xi=\infty$, so a finite $\chi$ hurts most. And **BTRG beats TRG by one to two orders of
# magnitude at identical cost.** If you only ever need a free energy, that is the whole
# answer: use BTRG.

# ---
# ## 3 · Watching TRG break
# 
# The free energy is forgiving: it is a single number dominated by the leading eigenvalue.
# The **spectrum** is not forgiving. This is where corner double lines show up.
# 
# Recall from the lecture: glue the fixed-point tensor into a tube and its transfer-matrix
# eigenvalues encode the CFT,
# 
# $$\lambda_i \approx \mathrm{e}^{f}\,\mathrm{e}^{-2\pi\left(\Delta_i - c/12\right)}
# \qquad\Longrightarrow\qquad
# \Delta_i - \Delta_{\min} \approx -\frac{1}{2\pi}\log\frac{\lambda_i}{\lambda_{\max}} .$$
# 
# `CFTData` does this for you. First, find out what it actually gives you:

s = TRG(classical_ising(Z2Irrep, ising_βc))
run!(s, truncrank(16), maxiter(20); verbosity=0)

cft = CFTData(s; shape = [√2, 2√2, 0])

@show typeof(cft)
@show fieldnames(typeof(cft))
@show cft.central_charge
cft.scaling_dimensions

# > If a field name differs from what you expect, `?CFTData` and `fieldnames` are your friends.
# > The package moves; the physics does not.
# 
# ### ▸ YOUR TURN — the drift
# 
# Collect the CFT data **at every RG step** and plot the lowest ~8 scaling dimensions against
# the iteration number. Use a `Finalizer`, exactly as in the review:
# 
# ```julia
# function my_finalize_cftdata!(scheme::TNRScheme)
#     n = finalize!(scheme)
#     return CFTData(scheme; shape = [√2, 2√2, 0])
# end
# my_CFT_Finalizer = Finalizer(my_finalize_cftdata!, CFTData)
# ```
# 
# **Exact answers for the Ising CFT:** $c = \tfrac12$ and, in increasing order,
# $\Delta = 0,\ \tfrac18,\ 1,\ \tfrac98,\ 2,\ \tfrac{17}{8},\ \dots$
# 
# > Note $\tfrac{17}{8} = 2.125$ is **above** 2, not below. And $\Delta = 2$ is **four**-fold:
# > $T,\bar T$ at spin $s=\pm2$ *plus* $\partial\varepsilon,\bar\partial\varepsilon$ at $s=\pm1$.

function my_finalize_cftdata!(scheme::TNRScheme)
    n = finalize!(scheme)
    return CFTData(scheme; shape = [√2, 2√2, 0])
end
my_CFT_Finalizer = Finalizer(my_finalize_cftdata!, CFTData)

scheme = TRG(classical_ising(Z2Irrep, ising_βc))
data   = run!(scheme, truncrank(16), maxiter(40), my_CFT_Finalizer; verbosity=0)

# `data` is now a vector of CFTData, one per RG step.
nlev = 8
Δs = [collect(Iterators.take(sort(vcat(collect(values(d.scaling_dimensions))...)), nlev))
      for d in data]

p = plot(; xlabel="RG step", ylabel="Δ", ylim=(-0.1, 2.6),
         title="TRG, χ = 16: the spectrum drifts", legend=false)
for lev in 1:nlev
    plot!(p, [d[lev] for d in Δs if length(d) >= lev]; marker=:circle, ms=2, c=:steelblue)
end
for exact in (0, 1/8, 1, 9/8, 2, 17/8)
    hline!(p, [exact]; ls=:dash, c=:gray, lw=0.8)
end
p

# **What you should see.** The first ~10 steps look fine. Then the dimensions start to wander
# away from the dashed exact values and never come back.
# 
# That is the corner-double-line problem, with numbers on it: TRG's local SVD cannot tell
# short-range junk from long-distance physics, so the junk eats bond dimension step after
# step, and after enough steps there is nothing left for the actual CFT.
# 
# Repeat with `BTRG` instead of `TRG`. It stays put noticeably longer at the same cost —
# but it still eventually drifts, because BTRG is still a *local* truncation.

scheme = BTRG(classical_ising(Z2Irrep, ising_βc))
data_b = run!(scheme, truncrank(16), maxiter(40), my_CFT_Finalizer; verbosity=0)

Δb = [collect(Iterators.take(sort(vcat(collect(values(d.scaling_dimensions))...)), 8))
      for d in data_b]
p = plot(; xlabel="RG step", ylabel="Δ", ylim=(-0.1,2.6),
         title="BTRG, χ = 16", legend=false)
for lev in 1:8
    plot!(p, [d[lev] for d in Δb if length(d) >= lev]; marker=:circle, ms=2, c=:seagreen)
end
for exact in (0, 1/8, 1, 9/8, 2, 17/8); hline!(p,[exact];ls=:dash,c=:gray,lw=0.8); end
p

# ---
# ## 4 · LoopTNR: a spectrum that stays put
# 
# LoopTNR replaces the local SVD with a cost function over a whole loop,
# 
# $$\mathcal{L}(\{S_i\}) = \big\|\, \underbrace{T_AT_B/T_BT_A}_{2\times2\ \text{patch}} - \underbrace{S_1\cdots S_8}_{\text{octagon}} \,\big\|_F^2 ,$$
# 
# minimised by sweeping the linear solves $N_jS_j = W_j$, with an entanglement-filtering step
# in front to squeeze out the CDL structure.
# 
# **This is slower.** Budget a few minutes per run, and start with `maxiter(20)` before you
# go to 40.
# 
# ### ▸ YOUR TURN
# Produce the same plot for `LoopTNR`, and read off $c$.

scheme = LoopTNR(classical_ising(Z2Irrep, ising_βc))
data_l = run!(scheme, truncrank(16), maxiter(20), LoopParameters(), my_CFT_Finalizer;
              verbosity=0)

Δl = [collect(Iterators.take(sort(vcat(collect(values(d.scaling_dimensions))...)), 10))
      for d in data_l]
p = plot(; xlabel="RG step", ylabel="Δ", ylim=(-0.1,2.6),
         title="LoopTNR, χ = 16: flat", legend=false)
for lev in 1:10
    plot!(p, [d[lev] for d in Δl if length(d) >= lev]; marker=:circle, ms=2, c=:firebrick)
end
for exact in (0, 1/8, 1, 9/8, 2, 17/8); hline!(p,[exact];ls=:dash,c=:gray,lw=0.8); end
p

cs = [d.central_charge for d in data_l]
plot(cs; marker=:circle, xlabel="RG step", ylabel="c", label="LoopTNR, χ=16", lw=2)
hline!([0.5]; ls=:dash, c=:gray, label="exact c = 1/2")

@printf("c at the last step: %.6f   (exact 0.5)\n", cs[end])

# **Discussion.** Two things to notice, and they are the point of the whole lecture:
# 
# 1. LoopTNR's spectrum is *flat* across 20+ steps, where TRG's wandered off after 10. The
#    entanglement filtering is what buys you that: it lets the flow **sit on an unstable fixed
#    point** instead of sliding off it.
# 2. You are reading the central charge of a CFT off a $16\times16\times16\times16$ array of
#    numbers. No Monte Carlo, no finite-size scaling, no fitting.
# 
# **Sanity check worth doing:** run LoopTNR slightly off criticality, say $\beta = 1.02\beta_c$.
# What happens to $c$ as you coarse-grain, and why? *(It should flow to 0 — the ordered phase
# has zero correlation length, hence no CFT. How many steps does it take? That number is
# $\sim\log_b \xi$.)*

for β in (ising_βc, 1.02*ising_βc)
    s = LoopTNR(classical_ising(Z2Irrep, β))
    d = run!(s, truncrank(12), maxiter(16), LoopParameters(), my_CFT_Finalizer; verbosity=0)
    @printf("β/β_c = %.3f :  c(final) = %+.4f\n", β/ising_βc, d[end].central_charge)
end

# ---
# ## 5 · Playing with the geometry
# 
# The fixed-point tensor is a **square patch of spacetime**. How you glue it determines which
# surface you compute on, and therefore which data you can read off. The shape is a triple
# $[\,h, L, x\,]$: height, circumference, horizontal shift.
# 
# ```julia
# CFTData(scheme; shape = [√2, 2√2, 0])   # the workhorse
# CFTData(scheme; shape = [1, 4, 1])      # wider, and x = 1 → conformal spins mod 4
# CFTData(scheme; shape = [1, 8, 1], trunc1, trunc2)
# ```
# 
# Two rules from the lecture:
# 
# - **Wide and short, never tall.** Larger $L$ enlarges the effective Hilbert space
#   ($[1,1,0]\to[2,2,0]$ takes $\chi \to \chi^2$) so descendants are resolved.
#   Larger $h$ adds *no* states and pushes eigenvalues towards $10^{-11}$, i.e. into the noise.
# - $x\neq0$ gives the eigenvalues a phase $\mathrm{e}^{2\pi i x s_i}$: **modulus $\to \Delta$,
#   phase $\to s$.** With $x = p/q$, spins come out mod $q$.
# 
# ### ▸ YOUR TURN
# Take one converged LoopTNR fixed-point tensor and extract CFT data with three different
# shapes. Which shape resolves the most levels? How far up can you go before it stops
# looking like $0, \tfrac18, 1, \tfrac98, \tfrac{17}{8}, 2, \dots$?

s = LoopTNR(classical_ising(Z2Irrep, ising_βc))
run!(s, truncrank(16), maxiter(20), LoopParameters(); verbosity=0)

for shp in ([1, 1, 0], [√2, 2√2, 0], [1, 4, 1])
    cft = CFTData(s; shape = shp)
    Δ = sort(vcat(collect(values(cft.scaling_dimensions))...))
    println("shape = ", shp)
    println("   c  = ", round(cft.central_charge; digits=5))
    println("   Δ  = ", round.(Δ[1:min(10,end)]; digits=4))
    println()
end

# **Exact Ising spectrum (diagonal modular invariant), in increasing order:**
# 
# | $\Delta$ | 0 | 1/8 | 1 | 9/8 | 2 | 17/8 |
# |---|---|---|---|---|---|---|
# | spin $s$ | 0 | 0 | 0 | $\pm1$ | $\pm2,\ \pm1$ | $+2,\,0,\,-2$ |
# | | $\mathbb{1}$ | $\sigma$ | $\varepsilon$ | $\partial\sigma,\bar\partial\sigma$ | $T,\bar T$; $\partial\varepsilon,\bar\partial\varepsilon$ | |
# 
# Two things to notice, because they trip people up:
# 
# - $\tfrac{17}{8} = 2.125$ sits **above** $\Delta=2$.
# - $\Delta=2$ is **four**-fold degenerate, not two-fold: the stress tensor $T,\bar T$ at
#   $s=\pm2$, *and* $\partial\varepsilon,\bar\partial\varepsilon$ at $s=\pm1$. In your numerics
#   that line will have four points on it.
# - The gap between $\tfrac98$ and $2$ is because $L_{-1}|0\rangle = 0$ — the identity module has
#   no level-1 state.
# 
# $T$ and $\bar T$ are a good target: with $x \neq 0$ in the shape you should be able to *see*
# the conformal spin split them.

cft = CFTData(s; shape = [1, 4, 1])
println(fieldnames(typeof(cft)))

# If a spin field exists, a (s, Δ) scatter is the clearest possible picture of a CFT:
#   scatter(spins, Δs; xlabel="conformal spin s", ylabel="Δ",
#           title="Ising CFT from a 16×16×16×16 tensor")
# Overlay the exact primaries + descendants and you should see them line up on
# integer spins, which is the robustness argument from the lecture.

# ---
# ## 6 · Choose your own model
# 
# Pick **one**. Twenty minutes each, and each one is a real physics question.
# 
# ### (a) The $q$-state clock model — a BKT transition
# `classical_clock(ZNIrrep{q}, q, β)`. For $q \ge 5$ there are **two** transitions with a
# critical phase between them. Scan $\beta$ and watch $c$: it should sit near 1 in the
# intermediate phase and drop to 0 outside. Can you locate both transitions?
# 
# ### (b) The six-vertex model — a line of fixed points
# `sixvertex(U1Irrep; a, b, c)`. The critical phase $-1 \le \Delta \le 1$ is a $c=1$ CFT, but
# a *one-parameter family* labelled by the Luttinger parameter $K$. Fix $a=b=1$ so
# $\Delta = \tfrac{c^2}{2}-1$, then extract $K = 1/(4\Delta_1)$ from the lowest scaling
# dimension and compare to Bethe ansatz. The BKT point $\Delta=1$ should give $K=\tfrac12$.
# 
# > **Mind the convention.** In the XXZ convention used here,
# > $\Delta = \dfrac{c^2-a^2-b^2}{2ab}$, which at $a=b=1$ gives $\Delta = \tfrac{c^2}{2}-1$ and
# > puts the BKT point at $\Delta=+1$. Baxter's six-vertex anisotropy is the opposite sign,
# > $\Delta_{6v} = \dfrac{a^2+b^2-c^2}{2ab} = -\Delta_{\rm XXZ}$, which puts the BKT point at
# > $-1$. Both appear in the literature; say out loud which one you are using. *(The formula
# > printed in §4.3 of the review has a stray factor 2 on $c^2$ — it disagrees with that
# > paper's own Fig. 16 caption. Use the one above.)*
# 
# ### (c) $\phi^4$ in two dimensions — a continuous field
# `phi4_real(Z2Irrep, K, μ0, λ, h)`. The bond dimension now comes from a *truncated Taylor
# expansion*, not from a spin. Check convergence in that truncation as well as in $\chi$.
# The transition is Ising ($c=\tfrac12$) — verify it.
# 
# ### (d) Gross–Neveu — fermions at finite density
# `gross_neveu_start(FermionParity, μ, m, g)`. Compute
# $\langle n\rangle = \frac{1}{V}\frac{\ln Z(\mu+\Delta\mu) - \ln Z(\mu-\Delta\mu)}{2\Delta\mu}$
# and plot it against $\mu$. **There is no sign problem here** — that is the headline.
# 
# ### (e) 3D Ising — the open frontier
# `classical_ising_3D` with `HOTRG_3D` or `ATRG_3D`. Get the free energy. Then try to get the
# spectrum, and see for yourself why nobody has solved this yet.

# Your model here.

# ---
# ## Where to go next
# 
# - `TNRKit.jl` docs: <https://quantumkithub.github.io/TNRKit.jl/stable/>
# - The review this tutorial follows: **arXiv:2604.06922**
# - Original papers: Levin & Nave PRL **99**, 120601 (2007) · Xie *et al.* PRB **86**, 045139 (2012) ·
#   Gu & Wen PRB **80**, 155131 (2009) · Evenbly & Vidal PRL **115**, 180405 (2015) ·
#   Yang, Gu & Wen PRL **118**, 110504 (2017) · Adachi *et al.* PRB **105**, L060402 (2022) ·
#   Homma, Okubo & Kawashima PRR **6**, 043102 (2024)
# 
# **If you want a project:** 3D entanglement filtering. Corner triple lines and edge double
# lines grow *linearly* with system size rather than saturating, so no 3D coarse-graining
# scheme currently converges to the right fixed point. Solve that and you open up lattice
# gauge theory in 3+1 dimensions.

