import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt

# grad_data = np.load("./c4_llama_grad_dict.npz")
grad_data = np.load("./grad_dict1.npz")
grad_dict = {key: grad_data[key] for key in grad_data}
# llama_loss = np.load("c4_llama_loss.npy")[:1000]

num_spikes = []  # Used to hold the number of elements greater than 50 in each layer
for key in grad_data.keys():
    print(f"layers at {key}")
    # Get the gradient data for each layer
    data = grad_data[key]
    # Calculate the absolute value and mean
    grad_abs = np.abs(data)
    grad_mean = np.mean(grad_abs, axis=0)
    # Calculate the times value
    times = grad_abs / grad_mean
    mask = times > 50  # Set the threshold
    # Get the number of masks that are True
    count_above_50 = np.sum(mask)
    print(f"Number of elements > 50 in layer {key}: {count_above_50}")
    # Save the number of elements > 50 in each layer
    num_spikes.append(count_above_50)

# Print the number of elements > 50 in each layer print("Number of elements > 50 for each layer:")
print("Number of elements > 50 for each layer:")
for i, count in enumerate(num_spikes):
    print(f"Layer {i}: {count}")
grad_detach = grad_dict['20']
grad_abs_detach = np.abs(grad_detach)
grad_mean_detach = np.mean(grad_abs_detach, axis=0)
times_detach = grad_abs_detach / grad_mean_detach
mask_detach = times_detach > 50
print(mask_detach.sum())
true_indices_detach = np.where(mask_detach == True)
a_detach = true_indices_detach[1][0]
b_detach = true_indices_detach[2][0]

# fig = figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
# vgg_all = fig.add_subplot(1, 1, 1)
# vgg_all.plot(grad_detach[:, true_indices_detach[1][:], true_indices_detach[2][:]])

fig = plt.figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
plt.plot(grad_detach[:, true_indices_detach[1][:], true_indices_detach[2][:]])
plt.show()

for i in range(0, true_indices_detach[1].shape[0], 20):
    fig = plt.figure(num=None, figsize=(22, 5), dpi=120, facecolor='w', edgecolor='k')
    plt.plot(grad_detach[:, true_indices_detach[1][i], true_indices_detach[2][i]])
    plt.show()

print('Done')
