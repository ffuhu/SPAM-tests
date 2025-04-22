import json
import os.path

import numpy as np
import math
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# real example of training for 100 update steps in asus saving all layers
exp_date = "2025-04-10_16-35-53"
grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_grads.h5"
loss_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy"
model_path = f"logs/c4_llama_60m_none/{exp_date}/model_layers.json"
thresh = 40
x_limit = 700
max_traj = 100
range_xs_spike_detection = 10
show_figs = False

# exp_date = "2025-04-16_14-54-04"
# exp_date = "2025-04-16_20-57-47"
# exp_date = "2025-04-16_21-42-53"
# grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_AdamWGradientSaving_grads.h5"
# grad_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_MuonGradientSaving_grads.h5"
# loss_path = f"logs/c4_llama_60m_none/{exp_date}/c4_llama_60m_none_loss.npy"
# model_path = f"logs/c4_llama_60m_none/{exp_date}/model_layers.json"
# thresh = 40
# x_limit = 700
# max_traj = 100
# range_xs_spike_detection = 10
# show_figs = False

# function used to summarize trajectories
def summarize_numbers(numbers, as_int=False):
    """Summarize consecutive numbers into ranges (e.g., [1,2,3,4] -> '1-4')."""
    if not len(numbers):
        return ""

    # Sort and remove duplicates
    sorted_numbers = sorted(set(numbers))

    ranges = []
    start = sorted_numbers[0]
    prev = start

    for num in sorted_numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = prev = num
    ranges.append((start, prev))  # Add the last range

    if as_int:
        return ranges

    # Format ranges
    result = []
    for start, end in ranges:
        if start == end:
            result.append(str(start))
        else:
            result.append(f"{start}-{end}")

    return ", ".join(result)

num_spikes = []  # Used to hold the number of elements greater than thresh in each layer
llama_loss = np.load(loss_path)[:x_limit]

# load model layers
model_layers = json.loads(open(model_path).read())
column_names = ["layer #", "layer_name", "layer_shape", "# params", "threshold", "max spike", "# spikes", "steps"]
spikes_info = pd.DataFrame(columns=column_names)
spikes_rows = []

with h5py.File(grad_path, "r") as grad_data_file:
    for layer_info in model_layers.items(): # grad_data_file.keys():
        layer_idx = int(layer_info[0])
        layer_name = layer_info[1]["name"]  # + "_before"
        layer_shape = layer_info[1]["shape"]
        layer_nparams = math.prod(layer_shape)

        # layer_name = 'model.layers.2.self_attn.k_proj.weight'
        if 'embed' in layer_name or 'head' in layer_name:
            continue

        # Get the gradient data for each layer
        try:
            grad_data = grad_data_file[layer_name][:x_limit]
        except KeyError as e:
            print(f"{layer_name} not found in {grad_path}!")
            continue

        # Print layer info
        print(f"Layer grad: {layer_idx}\t{layer_name}", end='\t')
        print(f"Shape: {layer_shape}", end='\t')
        print(f"# params: {layer_nparams}", end='\t')

        # Calculate the absolute value and mean
        grad_abs = np.abs(grad_data)
        grad_mean = np.mean(grad_abs, axis=0)
        # Calculate the times value
        times = grad_abs / (grad_mean + 1e-6)
        max_spike = np.nanmax(times)
        print(f"max(times): {max_spike}", end='\t')
        # Keep only above the threshold
        mask = times > thresh
        # Get the number of masks that are True
        count_above_thresh = np.sum(mask)
        print(f"# of elements > {thresh}: {count_above_thresh}", end='\t')
        # Save the number of elements > thresh in each layer
        num_spikes.append(count_above_thresh)

        # study spikes
        true_indices_detach = np.where(mask == True)
        steps_with_spike = np.unique(true_indices_detach[0])

        # Save information in a csv
        column_values = [layer_idx, layer_name, layer_shape, layer_nparams, thresh, max_spike, count_above_thresh,
                         steps_with_spike]
        spikes_row = dict(zip(column_names, column_values))
        spikes_rows.append(spikes_row)

        if len(steps_with_spike) == 0 and f"{layer_name}_before" not in grad_data_file.keys():
            # print(f"{layer_name} has no spikes (thresh={thresh}).")
            print()
            continue
        print(f"steps: {summarize_numbers(steps_with_spike)}")

        for step in steps_with_spike:
            steps_idxs = true_indices_detach[0] == step
            values_step_i = true_indices_detach[1][steps_idxs]
            values_step_j = true_indices_detach[2][steps_idxs]
            assert len(values_step_i) == len(values_step_j), "Error: values_i don't match values_j"
            # print(f"i values for step: {step}: {len(values_step_i)}")
            # print(f"j values for step: {step}: {len(values_step_j)}")

        # if gradients_before are saved, plot them as well
        if f"{layer_name}_before" in grad_data_file.keys():
            try:
                grad_data_before = grad_data_file[layer_name + '_before'][:x_limit]
            except KeyError as e:
                print(f"{layer_name}_before not found in {grad_path}!")
                continue

            # if gradients don't have spikes, compute gradients_before
            if len(steps_with_spike) == 0:
                # Calculate the absolute value and mean
                grad_abs = np.abs(grad_data_before)
                grad_mean = np.mean(grad_abs, axis=0)
                # Calculate the times value
                times = grad_abs / (grad_mean + 1e-6)
                max_spike = np.nanmax(times)
                print(f"[gradients_before] max(times): {max_spike}", end='\t')
                # Keep only above the threshold
                mask = times > thresh
                # Get the number of masks that are True
                count_above_thresh = np.sum(mask)
                print(f"# of elements > {thresh}: {count_above_thresh}", end='\t')

                # study spikes for _before
                true_indices_detach = np.where(mask == True)
                steps_with_spike = np.unique(true_indices_detach[0])

                if len(steps_with_spike) == 0:
                    # print(f"{layer_name} has no spikes (thresh={thresh}).")
                    print()
                    continue
                print(f"steps: {summarize_numbers(steps_with_spike)}")

                for step in steps_with_spike:
                    steps_idxs = true_indices_detach[0] == step
                    values_step_i = true_indices_detach[1][steps_idxs]
                    values_step_j = true_indices_detach[2][steps_idxs]
                    assert len(values_step_i) == len(values_step_j), "Error: values_i don't match values_j"
                    # print(f"i values for step: {step}: {len(values_step_i)}")
                    # print(f"j values for step: {step}: {len(values_step_j)}")

            fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            fig.suptitle(exp_date.replace("_", " "))
            subfig = fig.add_subplot(3, 1, 1)
            subfig.set_title(f"{layer_idx}: {layer_name}")
            subfig.plot(llama_loss)
            subfig.grid(True)
            subfig.set_xlim(0, x_limit)  # Establecer límites en el eje x
            subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            num_traj = true_indices_detach[1].shape[0]
            subfig = fig.add_subplot(3, 1, 2)
            subfig.set_title(f"# of trajectories with spike in gradients after NS = 0")
            subfig.plot(grad_data[:, true_indices_detach[1][:max_traj], true_indices_detach[2][:max_traj]])
            subfig.grid(True)
            subfig.set_xlim(0, x_limit)  # Establecer límites en el eje x
            # Increase grid size by setting custom ticks
            subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            subfig = fig.add_subplot(3, 1, 3)
            subfig.set_title(f"# of traj. with spike (before NS): {true_indices_detach[1].shape[0]} (showing {max_traj})"
                             f" in {len(steps_with_spike)} steps")
            subfig.plot(grad_data_before[:, true_indices_detach[1][:max_traj], true_indices_detach[2][:max_traj]])
            subfig.grid(True)
            subfig.set_xlim(0, x_limit)  # Establecer límites en el eje x
            # Increase grid size by setting custom ticks
            subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
                    transform=subfig.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=10)

            fig_path = os.path.join(os.path.dirname(grad_path),
                                    f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads_before.png")
            plt.savefig(fig_path)
            # plt.show()
            plt.close()
        else:

            fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
            fig.suptitle(exp_date.replace("_", " "))
            subfig = fig.add_subplot(2, 1, 1)
            subfig.set_title(f"{layer_idx}: {layer_name}")
            subfig.plot(llama_loss)
            subfig.grid(True)
            subfig.set_xlim(0, x_limit)  # Establecer límites en el eje x
            subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            # Add dashed vertical lines for detected spikes
            spikes_xs = [r[0] for r in summarize_numbers(steps_with_spike, as_int=True)]
            ymin, ymax = plt.ylim()  # Get current y-limits
            subfig.vlines(x=spikes_xs,  ymin=ymin, ymax=ymax,
                          colors='black', linestyles='dashed', linewidths=1)


            subfig = fig.add_subplot(2, 1, 2)
            subfig.set_title(f"# of trajectories with spike: {true_indices_detach[1].shape[0]}"
                             f" in {len(steps_with_spike)} steps")
            grad_data_plot = grad_data[:, true_indices_detach[1][:], true_indices_detach[2][:]]
            subfig.plot(grad_data_plot)
            subfig.grid(True)
            subfig.set_xlim(0, x_limit)  # Establecer límites en el eje x
            # Increase grid size by setting custom ticks
            subfig.set_xticks(range(0, x_limit + 1, 100))  # X-axis grid every 100 units

            subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
                        transform=subfig.transAxes,
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=10)

            # Add dashed vertical lines for detected spikes
            spikes_xs = [r[0] for r in summarize_numbers(steps_with_spike, as_int=True)]
            ymin, ymax = plt.ylim()  # Get current y-limits
            subfig.vlines(x=spikes_xs, ymin=ymin, ymax=ymax,
                          colors='black', linestyles='dashed', linewidths=1)

            fig_path = os.path.join(os.path.dirname(grad_path), f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads.png")
            plt.savefig(fig_path)
            if show_figs:
                plt.show()
            plt.close()

            # inspection of gradients close to the spikes detected
            for spike_x in spikes_xs:
                for range_x in range(-range_xs_spike_detection, range_xs_spike_detection):
                    pos = spike_x + range_x
                    print(f"[{pos}]\tloss={llama_loss[pos]:.4f}\tgrads_abs_max={np.abs(grad_data_plot[pos]).max():.4f}")

                fig = plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
                fig.suptitle(exp_date.replace("_", " ") + f" gradient spike {spike_x}")
                subfig = fig.add_subplot(2, 1, 1)
                subfig.set_title(f"{layer_idx}: {layer_name}")
                subfig.plot(llama_loss[spike_x - range_xs_spike_detection: spike_x + range_xs_spike_detection])
                subfig.grid(True)
                subfig.set_xlim(0, 2 * range_xs_spike_detection - 1)  # Establecer límites en el eje x
                subfig.set_xticks(range(0, 2 * range_xs_spike_detection, 1))  # X-axis grid every 100 units
                subfig.set_xticklabels(range(spike_x - range_xs_spike_detection, spike_x + range_xs_spike_detection))

                # Add dashed vertical lines for detected spikes
                ymin, ymax = plt.ylim()  # Get current y-limits
                subfig.vlines(x=range_xs, ymin=ymin, ymax=ymax,
                              colors='black', linestyles='dashed', linewidths=1)

                subfig = fig.add_subplot(2, 1, 2)
                subfig.set_title(f"# of trajectories with spike: {true_indices_detach[1].shape[0]}"
                                 f" in {len(steps_with_spike)} steps")
                grad_data_plot = grad_data[:, true_indices_detach[1][:], true_indices_detach[2][:]]
                subfig.plot(grad_data_plot[spike_x - range_xs_spike_detection: spike_x + range_xs_spike_detection])
                subfig.grid(True)
                subfig.set_xlim(0, 2 * range_xs_spike_detection - 1)  # Establecer límites en el eje x
                subfig.set_xticks(range(0, 2 * range_xs_spike_detection, 1))  # X-axis grid every 100 units
                subfig.set_xticklabels(range(spike_x - range_xs_spike_detection, spike_x + range_xs_spike_detection))

                subfig.text(0.95, 0.95, f"steps: {summarize_numbers(steps_with_spike)}",
                            transform=subfig.transAxes,
                            horizontalalignment='right',
                            verticalalignment='top',
                            fontsize=10)

                # Add dashed vertical lines for detected spikes
                ymin, ymax = plt.ylim()  # Get current y-limits
                subfig.vlines(x=range_xs_spike_detection, ymin=ymin, ymax=ymax,
                              colors='black', linestyles='dashed', linewidths=1)

                fig_path = os.path.join(os.path.dirname(grad_path),
                                        f"fig_{layer_idx}_{layer_name.replace('.', '-')}_grads_inspection_{spike_x}.png")
                plt.savefig(fig_path)
                if show_figs:
                    plt.show()
                plt.close()

    # Save the info of spikes to a csv
    spikes_info = pd.concat([spikes_info, pd.DataFrame(spikes_rows)], ignore_index=True)
    spikes_info.to_csv(os.path.join(os.path.dirname(grad_path), 'grad_spikes_report.csv'), index=False)