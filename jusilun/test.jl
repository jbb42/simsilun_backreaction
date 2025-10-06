using NPZ
using PyPlot
const plt = PyPlot
pygui(true)

data = npzread("data/ics/delta.npy")
println(minimum(data))

plt.imshow(data[:,:,32]', origin="lower", cmap="viridis", aspect="equal")
plt.colorbar()
plt.show()