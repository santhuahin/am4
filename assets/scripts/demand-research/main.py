# %%
import sys
from pathlib import Path

sys.path.insert(0, "..")  # for plots

import frustrum
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plots import MPL

base_path = Path("../../../../src/am4/utils/data/")
path_assets = Path(__file__).parent.parent.parent
path_img = path_assets / "img" / "demand-research"
path_video = path_assets / "video"
path_img.mkdir(parents=True, exist_ok=True)
path_video.mkdir(parents=True, exist_ok=True)

airports = pl.read_parquet(base_path / "airports.parquet")
routes = pl.read_parquet(base_path / "routes.parquet")

demands_y = np.array(routes["yd"], dtype=np.uint16)
demands_j = np.array(routes["jd"], dtype=np.uint16)
demands_f = np.array(routes["fd"], dtype=np.uint16)
hub_costs = np.array(airports["hub_cost"])
latitudes = np.array(airports["lat"])
longitudes = np.array(airports["lng"])
runways = np.array(airports["rwy"])
hub_costs = np.array(airports["hub_cost"])
iata = np.array(airports["iata"])
icao = np.array(airports["icao"])


def iter_indices(n: int = 3907):
    x, y = 0, 0
    while True:
        y += 1
        if y == n:
            x += 1
            y = x + 1
        yield x, y


# %%
matrix_y = np.empty((3907, 3907), dtype=np.float32)
matrix_y.fill(np.nan)
matrix_j = np.empty((3907, 3907), dtype=np.float32)
matrix_j.fill(np.nan)
matrix_f = np.empty((3907, 3907), dtype=np.float32)
matrix_f.fill(np.nan)

for (x, y), dy, dj, df in zip(iter_indices(), demands_y, demands_j, demands_f):
    matrix_y[x, y] = dy
    matrix_y[y, x] = dy
    matrix_j[x, y] = dj
    matrix_j[y, x] = dj
    matrix_f[x, y] = df
    matrix_f[y, x] = df

# %%
MPL.init()
# %% full D[i, j] matrix plots

for matrix, name in zip([matrix_y, matrix_j, matrix_f], ["yds", "jds", "fds"]):
    if name != "jds":
        continue  # temp
    fig = plt.figure(figsize=(10, 8.684))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 10])
    gs.update(hspace=0.0, wspace=0.0)

    ax1 = plt.subplot(gs[0])
    ax1.set_aspect("auto")
    ax1.tick_params(labelbottom=False, labelleft=False)
    cax1 = ax1.imshow(hub_costs.reshape(1, -1), interpolation="none", cmap="viridis", aspect="auto")

    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_aspect(1)
    cax2 = ax2.imshow(matrix, interpolation="none", cmap="viridis", aspect="auto")

    fig.colorbar(cax1, ax=ax1, orientation="vertical", label="Hub Cost", shrink=0.8)
    fig.colorbar(cax2, ax=ax2, orientation="vertical", label=name, shrink=0.8)

    plt.tight_layout()
    plt.savefig(path_img / f"matrix_{name}.webp", dpi=300)
    plt.close()

# %% zoom into the matrix

x0, x1 = 3400, 3500
y0, y1 = 400, 500

fig, axs = plt.subplots(1, 3, figsize=(12, 5), layout="tight", sharey=True)

for ax, matrix, name in zip(axs, [matrix_y, matrix_j, matrix_f], ["yds", "jds", "fds"]):
    im = ax.imshow(matrix[x0:x1, y0:y1], interpolation="none", cmap="viridis", extent=[y0, y1, x1, x0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im, cax=cax, orientation="horizontal")
    ax.set_title(name)

plt.tight_layout()
plt.savefig(path_img / "matrix_zoom.webp", dpi=300)
plt.close()
# %%
rows = 3907 * 3906 // 2
data = np.empty((rows, 3), dtype=np.uint32)

for i, ((x, y), dy, dj, df) in enumerate(zip(iter_indices(), demands_y, demands_j, demands_f)):
    hclow, hchigh = hub_costs[x], hub_costs[y]
    if hclow > hchigh:
        hclow, hchigh = hchigh, hclow
    data[i] = hclow, hchigh, dy

# %%
df = pl.DataFrame(data, schema=["hc0", "hc1", "yd"])
agg = df.group_by(["hc0", "hc1"]).agg(pl.col("yd").mean(), pl.col("yd").count().alias("count"))

fig, ax = plt.subplots(figsize=(10, 8), layout="tight")
sc = ax.scatter(agg["hc0"], agg["hc1"], c=agg["yd"], s=np.log1p(agg["count"]) * 5, cmap="viridis")
ax.set_xlabel("Origin Hub Cost")
ax.set_ylabel("Destination Hub Cost")
cb = plt.colorbar(sc, ax=ax, label="Mean Y Demand", orientation="horizontal")
cb.solids.set(alpha=1)
plt.savefig(path_img / "scatter_hubcost_demand.webp", dpi=300)
plt.close()
# %%

samples_y, samples_j, samples_f = [], [], []
for (x, y), dy, dj, df in zip(iter_indices(), demands_y, demands_j, demands_f):
    if hub_costs[x] == 48000 and hub_costs[y] == 48000:
        samples_y.append(dy)
        samples_j.append(dj)
        samples_f.append(df)
len(samples_y), len(samples_j), len(samples_f)
# %%

fig, ax = plt.subplots(figsize=(12, 5))
for matrix, name, color in zip(
    [samples_y, samples_j, samples_f], ["yds", "jds", "fds"], ["#007cad", "#008760", "#ac525d"]
):
    ax.hist(matrix, bins=np.arange(0, 600), alpha=0.5, label=name, color=color)

ax.set_title(f"hc0 == hc1 == 48000, N={len(samples_y)}")
ax.set_xlabel("Demand")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.savefig(path_img / "hist_demand_hc48000.svg")

# %% prepare dataset

X = []
Y = []

for (x, y), dy, dj, df in zip(iter_indices(), demands_y, demands_j, demands_f):
    if hub_costs[x] == 48000 and hub_costs[y] == 48000:
        X.append((x, y, latitudes[x], longitudes[x], runways[x], latitudes[y], longitudes[y], runways[y]))
        Y.append((dy, dj, df, dy + 2 * dj + 3 * df))

xs, ys = zip(*sorted(zip(X, Y), key=lambda x: x[1][0], reverse=True))
# %%

pminj = [[432, 45, 26], [204, 45, 102]]
pmaxj = [[204, 180, 12], [96, 180, 48]]

plt.close()
fig = plt.figure(layout="tight")
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    [y[0] for y in ys],
    [y[1] for y in ys],
    [y[2] for y in ys],
    c=[y[3] for y in ys],
    cmap="turbo_r",
    alpha=0.6,
    s=0.2,
    edgecolor="none",
)
ax.set_xlabel("y")
ax.set_ylabel("j")
ax.set_zlabel("f")
ax.view_init(elev=30, azim=30)
cb = plt.colorbar(sc)
cb.set_label("y+2j+3f")
cb.solids.set(alpha=1)
k = 1.4
for y, j, f in pminj:
    ax.plot([0, y * k], [0, j * k], [0, f * k], lw=0.5)
for y, j, f in pmaxj:
    ax.plot([0, y * k], [0, j * k], [0, f * k], lw=0.5)
plt.savefig(path_img / "3d_demand.webp", dpi=300)
# %%


def init():
    return (sc,)


def update(num):
    print(num)
    ax.view_init(azim=num)
    return (sc,)


ani = animation.FuncAnimation(fig, update, frames=range(30, 390, 5), interval=1000 / 60)
ani.save(path_video / "3d_demand_anim.webm", writer="ffmpeg", fps=30, dpi=200)
# %%
mat = np.empty((3907, 3907), dtype=np.float32)
mat.fill(np.nan)
y2j3f = [y[3] for y in ys]
for x, y, z in zip([x[0] for x in xs], [x[1] for x in xs], y2j3f):
    mat[x][y] = z
    mat[y][x] = z
# %%
points = {}
for (x, y), dy, dj, df in zip(iter_indices(), demands_y, demands_j, demands_f):
    ya = dy + 2 * dj + 3 * df
    if hub_costs[x] == 48000 and hub_costs[y] == 48000:
        if ya not in points:
            points[ya] = []
        points[ya].append((dy, dj, df))
# %% transform space

# corners of the far plane
# u=0, v=0 -> bottom-left (low j, low y)
p001 = np.array([204.0, 45.0, 102.0])
# u=1, v=0 -> bottom-right (low j, high y)
p101 = np.array([432.0, 45.0, 26.0])
# u=0, v=1 -> top-left (high j, low y)
p011 = np.array([96.0, 180.0, 48.0])
# u=1, v=1 -> top-right (high j, high y)
p111 = np.array([204.0, 180.0, 12.0])
k = 0.5  # dummy, doesn't affect uv

t_data = frustrum.create_transform_data(p001, p101, p011, k, p111)

u0s, u1s, u2s = [], [], []
for dy, dj, df, val in ys:
    xyz = np.array([dy, dj, df], dtype=np.float64)
    u, v, w = frustrum.inverse_map(xyz, t_data)
    u0s.append(val)
    u1s.append(v)
    u2s.append(1.0 - u)

fig = plt.figure(figsize=(10, 8), layout="tight")
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(u1s, u2s, u0s, c=u0s, cmap="turbo_r", s=0.3)
ax.set_xlabel("u1")
ax.set_ylabel("u2")
ax.set_zlabel("u0 (y+2j+3f)")
ax.set_title("Transformed Demand Space")
ax.set_proj_type("ortho")
plt.savefig(path_img / "transformed_space.webp", dpi=200)


# %%
def update(num):
    print(num)
    ax.view_init(azim=num)
    return (sc,)


ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 5), interval=1000 / 60)
ani.save(path_video / "transformed_space_anim.webm", writer="ffmpeg", fps=30, dpi=200)
# %%

ybds = (100, 700)
jbds = (25, 275)
fbds = (0, 150)

fig = plt.figure(figsize=(12, 6), layout="tight")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
ax = fig.add_subplot(gs[0, 0], projection="3d")
ax.view_init(elev=45, azim=0)

y2j3f_start = sorted(points.keys())[99]
p_start = np.array(points[y2j3f_start])

sc = ax.scatter(p_start[:, 0], p_start[:, 1], p_start[:, 2], s=2)
ax.set_xlabel("y")
ax.set_ylabel("j")
ax.set_zlabel("f")
ax.set_xlim(*ybds)
ax.set_ylim(*jbds)
ax.set_zlim(*fbds)
ax.set_title(f"y+2j+3f = {y2j3f_start}")

ax2 = fig.add_subplot(gs[0, 1])

u1s, u2s = [], []
for row in p_start:
    xyz = row.astype(np.float64)
    u, v, w = frustrum.inverse_map(xyz, t_data)
    u1s.append(v)
    u2s.append(1.0 - u)

sc2 = ax2.scatter(u1s, u2s, s=2)
ax2.set_xlabel("u1")
ax2.set_ylabel("u2")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.set_title("Transformed Slice (u1, u2)")
plt.savefig(path_img / "transformed_slices.webp", dpi=300)
# %%


def update(frame):
    print(f"update {frame}")
    p = np.array(points[frame])

    sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
    ax.set_title(f"y+2j+3f = {frame}")

    u1s_frame, u2s_frame = [], []
    for row in p:
        xyz = row.astype(np.float64)
        u, v, w = frustrum.inverse_map(xyz, t_data)
        u1s_frame.append(v)
        u2s_frame.append(1.0 - u)

    sc2.set_offsets(np.c_[u1s_frame, u2s_frame])
    return sc, sc2


ani = animation.FuncAnimation(fig, update, frames=sorted(points.keys()), interval=1000 / 60, blit=True)
ani.save(path_video / "transformed_slices_anim.mp4", writer="ffmpeg", fps=30, dpi=300)

# %%
