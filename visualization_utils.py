
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def BarhPlot(latent_vector, full_savepath):
    """Plot horizontal bars for absolute values of each latent vector components"""
    plt.figure(figsize=(10,20))
    plt.barh(range(latent_vector.shape[0]), latent_vector)
    plt.xlabel("Value", fontsize=16)
    plt.ylabel("Component number", fontsize=16)
    # plt.title("Latent vector components values")
    plt.ylim(0, latent_vector.shape[0])
    plt.xticks(fontsize=16)
    plt.savefig(full_savepath, bbox_inches='tight')
    plt.close()

def CountPlot(latent_vector, savepath, eps=1e-2):
    """Plot histogram of zero/non-zero elements for latent vector"""
    binary_vector = np.array(["Non-zero" if abs(value) > eps else "Close to zero" for value in latent_vector])
    binary_df = pd.DataFrame(columns=["zero_prox"], data=binary_vector)
    ax = sns.countplot(x="zero_prox", data=binary_df)
    ax.set(xlabel="latent vector components values")
    ax.figure.savefig(savepath)
    plt.close(ax.figure)