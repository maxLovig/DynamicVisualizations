import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary
import time
from torch.utils.data import Dataset
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap
import matplotlib.patches as mpatches


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index   # <-- keep original ID
    def __len__(self):
        return len(self.dataset)

def moving_average(arr, window=5):
    """
    arr: shape (T, N, p)
    window: int, smoothing window size
    returns smoothed array of shape (T - window + 1, N, p)
    """
    kernel = np.ones(window) / window
    # convolve along axis=0 (time), separately for each (N, p)
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='valid'),
        axis=0,
        arr=arr
    )
    return smoothed

def class_average(X, labels):
    """
    X: array of shape (T, N, p)
    labels: array of shape (N,), integers or categorical
    
    returns: dict mapping label -> averaged trajectory (T, p)
    """
    results = {}
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]    # sequences with this label
        results[lbl] = X[:, idx, :].mean(axis=1)  # average over N dimension
    return results



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 16

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
trainset = IndexedDataset(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
0 plane 
1 car
2 bird 
3 cat 
4 deer
5 dog
6 frog
7 horse
8 ship 
9 truck
'''

class Net(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, 10)

        # store activation function
        activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "swish": lambda x: x * torch.sigmoid(x),
            "gelu": F.gelu,
            "softplus": F.softplus
        }
        self.act = activations[activation]

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


net = Net(activation="relu")

params = list(net.parameters())
epochs = 10
lr = .01
training_log = []
trained_images = 0   # acts as timestamp

for epoch in range(epochs):
    for batch_idx, (inputs, labels, ids) in enumerate(trainloader):
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)

        net.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in params:
                p -= lr * p.grad

        # --- record info ---
        batch_size = inputs.size(0)
        preds = outputs.detach().cpu()

        for idx, label, out in zip(ids, labels, preds):
            training_log.append({
                "id": int(idx.item()),                # true dataset ID
                "timestamp": trained_images,          # number of samples seen so far
                "label": int(label.item()),
                "output": out.tolist()
            })

        trained_images += batch_size

        # optional print
        if (batch_idx + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}, Batch {batch_idx+1}] loss: {loss.item():.3f}")


groups = [ [] for _ in range(10)]
for e in training_log:
    groups[e["label"]].append(e)


window = 2000  # adjust as you like
smoothed = {}

for lbl, entries in enumerate(groups):
    # sort by timestamp (just in case)
    entries = sorted(entries, key=lambda x: x["timestamp"])

    timestamps = np.array([e["timestamp"] for e in entries])
    outputs = np.array([e["output"] for e in entries])  # shape: (N, num_classes)

    # --- moving average over time ---
    if len(outputs) >= window:
        kernel = np.ones(window) / window
        smooth_outputs = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='valid'),
            axis=0, arr=outputs
        )
    else:
        smooth_outputs = outputs  # too short, skip smoothing

    smoothed[lbl] = {
        "timestamps": timestamps[: len(smooth_outputs)],
        "avg_outputs": smooth_outputs
    }


# label names
label_names = {
    0: "plane",
    1: "car",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

cat1 = 1
cat2 = 5

plt.figure(figsize=(7,7))
step = 1000  # mark every 100 points

for lbl, data in smoothed.items():
    avg_out = torch.tensor(data["avg_outputs"])
    avg_out = torch.softmax(avg_out, dim=1)
    ts = np.arange(len(avg_out))  # or use data["timestamps"]

    # plot the main line and get its color
    line, = plt.plot(avg_out[:, cat1], avg_out[:, cat2],
                     label=label_names[lbl], alpha=0.8)
    color = line.get_color()

    # start and end markers
    plt.scatter(avg_out[0, cat1], avg_out[0, cat2], color="black", s=25, zorder=3)
    plt.scatter(avg_out[-1, cat1], avg_out[-1, cat2], color="red", s=25, zorder=3)

    # tick marks (larger × in same color as the line)
    idxs = np.arange(0, len(avg_out), step)
    plt.scatter(avg_out[idxs, cat1], avg_out[idxs, cat2],
                marker='x', color=color, s=60, linewidths=2, zorder=3)


plt.xlabel("Smoothed probability (Category 1)")
plt.ylabel("Smoothed probability (Category 2)")
plt.title("Trajectory of smoothed outputs")
plt.legend()
plt.grid(True)
plt.show()


# --- collect all outputs across labels to fit PCA ---
all_outputs = np.concatenate(
    [data["avg_outputs"] for data in smoothed.values()],
    axis=0
)

# fit PCA on full output space
pca = PCA(n_components=2)
pca.fit(all_outputs)

plt.figure(figsize=(7,7))
step = int(len(all_outputs) / 200)  # adjustable tick density

for lbl, data in smoothed.items():
    avg_out = torch.tensor(data["avg_outputs"])
    ts = data["timestamps"]

    # project onto PCA space
    proj = pca.transform(avg_out)

    # plot main trajectory
    line, = plt.plot(proj[:, 0], proj[:, 1],
                     label=label_names[lbl], alpha=0.8)
    color = line.get_color()

    # mark start and end points
    plt.scatter(proj[0, 0], proj[0, 1], color="black", s=25, zorder=3)
    plt.scatter(proj[-1, 0], proj[-1, 1], color="red", s=25, zorder=3)

    # tick marks along trajectory
    idxs = np.arange(0, len(proj), step)
    plt.scatter(proj[idxs, 0], proj[idxs, 1],
                marker='x', color=color, s=50, linewidths=2, zorder=3)

# label axes by PCA components
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("Trajectory of smoothed outputs (PCA projection)")
plt.legend()
plt.grid(True)
plt.show()

# Evolving Ridges Plot
# --- choose classes ---
target_label = 3   # e.g. "cat"
compare_label = 5  # e.g. "dog"
target_name = "cat"
compare_name = "dog"

# --- extract probabilities ---
entries = sorted(groups[target_label], key=lambda x: x["timestamp"])
timestamps = np.array([e["timestamp"] for e in entries])
outputs = torch.tensor([e["output"] for e in entries])
probs = torch.softmax(outputs, dim=1).numpy()
target_probs = probs[:, target_label]
compare_probs = probs[:, compare_label]

# --- bin setup ---
n_bins = 5
bins = np.linspace(timestamps.min(), timestamps.max(), n_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
xs = np.linspace(0, 1, 200)

# --- KDE per bin ---
target_curves, compare_curves = [], []
for i in range(n_bins):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i + 1])
    if mask.sum() < 10:
        target_curves.append(np.zeros_like(xs))
        compare_curves.append(np.zeros_like(xs))
        continue
    kde_t = gaussian_kde(target_probs[mask], bw_method=0.1)
    kde_c = gaussian_kde(compare_probs[mask], bw_method=0.1)
    target_curves.append(kde_t(xs))
    compare_curves.append(kde_c(xs))

# --- Ridge plot ---
plt.figure(figsize=(8, 7))
target_palette = sns.color_palette("Blues", n_bins)
compare_palette = sns.color_palette("Reds", n_bins)

for i, (center, density_t, density_c) in enumerate(zip(bin_centers, target_curves, compare_curves)):
    y_offset = i * 0.12
    # Target class
    plt.fill_between(xs, y_offset, density_t + y_offset,
                     color=target_palette[i], alpha=0.8)
    plt.plot(xs, density_t + y_offset, color='k', linewidth=0.6)
    # Dog class overlay
    plt.fill_between(xs, y_offset, density_c + y_offset,
                     color=compare_palette[i], alpha=0.4)
    plt.plot(xs, density_c + y_offset, color='k', linewidth=0.4)

plt.title(f"Ridge plot of P({target_name}) (blue) vs P({compare_name}) (red)")
plt.xlabel("Predicted probability")
plt.ylabel("Training time (binned)")
plt.yticks([])
plt.tight_layout()
plt.show()



# ------------------ color setup ------------------
def lighten_color(color, amount=0.7):
    """Return a lighter version of the given color (mix with white)."""
    r, g, b = to_rgb(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return to_hex((r, g, b))

target_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

cmaps = []
for c in target_colors:
    light = lighten_color(c, amount=0.5)
    cmap = LinearSegmentedColormap.from_list(f"{c}_fade", [light, c], N=256)
    cmaps.append(cmap)

# ------------------ ellipse helper ------------------
def plot_cov_ellipse(mean, cov, ax, color, n_std=1, face_alpha=0.01, edge_alpha=1.0):
    """Draw an ellipse with faint fill but solid outline."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    
    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor="black",
        lw=1.5,
        alpha=face_alpha
    )
    r, g, b, _ = ell.get_edgecolor()
    ell.set_edgecolor((r, g, b, edge_alpha))
    ax.add_patch(ell)

# ------------------ parameters ------------------
sets = [0,1,2,3,4,5,6,7,8,9]
k = 50
label_names = {
    0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

# ------------------ PCA projection ------------------
# collect all outputs across all categories
all_outputs = []
for lbl in range(10):
    entries = sorted(groups[lbl], key=lambda e: e["timestamp"])
    if entries:
        all_outputs.append(np.array([e["output"] for e in entries]))
if not all_outputs:
    raise ValueError("No outputs found in groups.")
all_outputs = np.vstack(all_outputs)

# fit PCA on the full output space
pca = PCA(n_components=2)
pca.fit(all_outputs)

# ------------------ plotting ------------------

def plot_category(target_label, cmap):
    entries = sorted(groups[target_label], key=lambda e: e["timestamp"])
    if not entries:
        return
    outputs = np.array([e["output"] for e in entries])
    # project onto first 2 principal components
    points = pca.transform(outputs)

    n = len(points)
    if n < k:
        return
    block_size = n // k
    blocks = [points[i*block_size:(i+1)*block_size] for i in range(k)]
    if n % k != 0:
        blocks[-1] = np.vstack((blocks[-1], points[k*block_size:]))

    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, k))
    for i, seg in enumerate(blocks):
        if len(seg) < 3:
            continue
        mean = seg.mean(axis=0)
        cov = np.cov(seg, rowvar=False)
        plot_cov_ellipse(mean, cov, ax, color=colors[i], face_alpha=0.08, edge_alpha=1.0)


fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect("equal", "box")

# plot selected categories
for lbl, cmap in zip(sets, [cmaps[i] for i in sets]):
    plot_category(lbl, cmap)

# build legend with solid-color patches
handles = [
    mpatches.Patch(color=cmaps[i](1.0), label=label_names[i])
    for i in sets
]

ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2, title="Categories")
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.set_xlim(-7.5, 10)
ax.set_ylim(-5, 5)
ax.set_aspect("equal", "box")
ax.set_title("Ellipsoid evolution (first two PCA components)")
plt.show()
