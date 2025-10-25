import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import torch

# ======== CHANNEL TRANSFORMS ========

class GreenOnly(object):
    def __call__(self, x):
        g = x[1:2, :, :]
        return torch.cat([
            torch.zeros_like(g),
            g,
            torch.zeros_like(g)
        ], dim=0)

class GreenRed(object):
    def __call__(self, x):
        r, g = x[0:1, :, :], x[1:2, :, :]
        return torch.cat([
            r,
            g,
            torch.zeros_like(g)
        ], dim=0)


# ======== DATASET WRAPPER ========

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index
    def __len__(self):
        return len(self.dataset)


# ======== MODEL ========

class Net(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, 10)

        activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "swish": lambda x: x * torch.sigmoid(x),
            "gelu": F.gelu
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


# ======== SETUP ========

batch_size = 16
lr = 0.01
epochs = 30  # total; split into 3 phases

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
to_tensor = transforms.ToTensor()

transform_green     = transforms.Compose([to_tensor, GreenOnly(), normalize])
transform_greenred  = transforms.Compose([to_tensor, GreenRed(), normalize])
transform_full      = transforms.Compose([to_tensor, normalize])

phases = [
    ("Green-only", transform_green),
    ("Green+Red", transform_greenred),
    ("Full RGB", transform_full)
]

# create model
net = Net(activation="relu")
params = list(net.parameters())

# tracking variables
training_log = []
loss_history = []
steps = []
trained_images = 0
step = 0
swap_steps = []

# ======== TRAINING LOOP ========

phase_epochs = epochs // len(phases)

for phase_idx, (phase_name, transform) in enumerate(phases):
    print(f"\n=== Starting phase {phase_idx+1}: {phase_name} ===")

    for epoch in range(phase_epochs):
        trainset = IndexedDataset(torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        for batch_idx, (inputs, labels, ids) in enumerate(trainloader):
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)

            net.zero_grad()
            loss.backward()
            with torch.no_grad():
                for p in params:
                    p -= lr * p.grad

            # track loss
            loss_history.append(loss.item())
            steps.append(step)
            step += 1

            # track outputs for later analysis
            preds = outputs.detach().cpu()
            for idx, label, out in zip(ids, labels, preds):
                training_log.append({
                    "id": int(idx.item()),
                    "timestamp": trained_images,
                    "label": int(label.item()),
                    "output": out.tolist()
                })

            trained_images += inputs.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(f"[{phase_name}] Epoch {epoch+1}, Batch {batch_idx+1}] loss: {loss.item():.3f}")

    swap_steps.append(trained_images)  # mark end of each phase

print("Training complete.")


# ======== LOSS CURVE ========

window = 100
if len(loss_history) >= window:
    kernel = np.ones(window) / window
    smooth_loss = np.convolve(loss_history, kernel, mode="valid")
    smooth_steps = steps[:len(smooth_loss)]
else:
    smooth_loss = loss_history
    smooth_steps = steps

plt.figure(figsize=(9,5))
plt.plot(steps, loss_history, color='lightgray', linewidth=0.7, label='Raw loss')
plt.plot(smooth_steps, smooth_loss, color='steelblue', linewidth=2, label='Smoothed loss')

for s in swap_steps[:-1]:
    plt.axvline(x=s/16, color='black', linestyle='--', alpha=0.8)

plt.title("Training Loss with Channel Additions")
plt.xlabel("Batch steps")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# ======== SMOOTH TRAJECTORIES ========

groups = [[] for _ in range(10)]
for e in training_log:
    groups[e["label"]].append(e)

window = 2000
smoothed = {}
for lbl, entries in enumerate(groups):
    entries = sorted(entries, key=lambda x: x["timestamp"])
    timestamps = np.array([e["timestamp"] for e in entries])
    outputs = np.array([e["output"] for e in entries])
    if len(outputs) >= window:
        kernel = np.ones(window) / window
        smooth_outputs = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='valid'), axis=0, arr=outputs
        )
    else:
        smooth_outputs = outputs
    smoothed[lbl] = {"timestamps": timestamps[:len(smooth_outputs)],
                     "avg_outputs": smooth_outputs}


# ======== PCA PROJECTION ========

label_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

all_outputs = np.concatenate([data["avg_outputs"] for data in smoothed.values()], axis=0)
pca = PCA(n_components=2)
pca.fit(all_outputs)

plt.figure(figsize=(7,7))
step = int(len(all_outputs) / 200)

for lbl, data in smoothed.items():
    avg_out = torch.tensor(data["avg_outputs"])
    ts = data["timestamps"]
    proj = pca.transform(avg_out)

    line, = plt.plot(proj[:, 0], proj[:, 1], label=label_names[lbl], alpha=0.8)
    color = line.get_color()

    plt.scatter(proj[0, 0], proj[0, 1], color="black", s=25, zorder=3)
    plt.scatter(proj[-1, 0], proj[-1, 1], color="red", s=25, zorder=3)

    idxs = np.arange(0, len(proj), step)
    plt.scatter(proj[idxs, 0], proj[idxs, 1],
                marker='x', color=color, s=25, linewidths=2, zorder=3)

    for j, s in enumerate(swap_steps[:-1]):
        swap_idx = np.searchsorted(ts, s)
        if swap_idx < len(proj):
            if j == 0:
                # first transition: black X
                plt.scatter(proj[swap_idx, 0], proj[swap_idx, 1],
                            marker='X', color='black', s=100, zorder=4)
            elif j == 1:
                # second transition: white X with black edge
                plt.scatter(proj[swap_idx, 0], proj[swap_idx, 1],
                            marker='X', facecolors='white', edgecolors='black',
                            linewidths=1.5, s=100, zorder=4)

plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("Trajectory of Smoothed Outputs (PCA projection)")
plt.legend()
plt.grid(True)
plt.show()

#Evolving Ridges

# --- choose class ---
target_label = 9
target_name = label_names[target_label]

# --- extract probabilities ---
entries = sorted(groups[target_label], key=lambda x: x["timestamp"])
timestamps = np.array([e["timestamp"] for e in entries])
outputs = torch.tensor([e["output"] for e in entries])
probs = torch.softmax(outputs, dim=1).numpy()[:, target_label]

# --- bin setup ---
n_bins = 12
bins = np.linspace(timestamps.min(), timestamps.max(), n_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# --- KDE per bin ---
xs = np.linspace(0, 1, 200)
curves = []
for i in range(n_bins):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i + 1])
    if mask.sum() < 10:
        curves.append(np.zeros_like(xs))
        continue
    kde = gaussian_kde(probs[mask], bw_method=0.1)
    curves.append(kde(xs))

# --- divide bins into 3 phases ---
phase_bins = np.array_split(range(n_bins), 3)
phase_titles = ["Phase 1: Green-only", "Phase 2: Green+Red", "Phase 3: Full RGB"]
phase_colors = ["#2ca02c", "#d62728", "#1f77b4"]

# --- create 3 stacked plots ---
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

for ax, idxs, color, title in zip(axes, phase_bins, phase_colors, phase_titles):
    for j, i in enumerate(idxs):
        y_offset = j * 0.12
        ax.fill_between(xs, y_offset, curves[i] + y_offset,
                        color=color, alpha=0.35)
        ax.plot(xs, curves[i] + y_offset, color='k', linewidth=0.6)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=90, labelpad=45, fontsize=10, va='center')
    ax.grid(False)
    ax.set_xlim(0, 1)

axes[-1].set_xlabel("Predicted probability")
fig.suptitle(f"Evolution of P({target_name}) across training phases", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
