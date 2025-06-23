"""
    This script is used to analyze the variance of the actions generated from pi0 model. 
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    actions_dir = pathlib.Path("data/aloha_sim/actions")
    actions_files = list(actions_dir.glob("actions_[0-9]*.npy"))
    # print if the files are found
    if len(actions_files) == 0:
        raise FileNotFoundError("No actions files found")
    else:
        print(f"Found {len(actions_files)} actions files")

    # load the actions from the files
    actions = []
    for file in actions_files:
        # print the file name
        print(f"Loading actions from {file}")
        actions.append(np.load(file))

    # concatenate the actions
    actions = np.concatenate(actions, axis=0)
    print(f"Actions shape: {actions.shape}")

    # visualize the sequence of actions of each action dimension
    plt.figure(figsize=(15, 10))
    # num_dims = actions.shape[1]
    num_dims = 3
    for i in range(num_dims):
        plt.subplot(num_dims, 1, i+1)
        plt.plot(actions[:, i])
        plt.ylabel(f'Dim {i}')
        if i == num_dims - 1:
            plt.xlabel('Time Step')
    plt.tight_layout()
    plot_dir = pathlib.Path("data/aloha_sim/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / 'action_sequences.png')
    plt.close()

if __name__ == "__main__":
    main()