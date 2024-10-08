import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def probability_histplot(
    label, probabilities, ax=None, fname=None, binwidth=0.01, thresholds=None
):
    label = ["benign" if l == 0 else "malignant" for l in label]
    data_plot = pd.DataFrame(
        {"Label": label, "Model output (probability)": probabilities}
    )

    plt.figure()
    sns.histplot(
        data_plot,
        x="Model output (probability)",
        hue="Label",
        kde=True,
        binwidth=binwidth,
        binrange=[0.0, 1.0],
        ax=ax,
        palette=[(5 / 255, 126 / 255, 255 / 255), (251 / 255, 4 / 255, 79 / 255)],
        alpha=0.7,
    )
    plt.ylabel("Number of patients")

    if fname is not None:
        plt.gcf().savefig(fname, bbox_inches="tight", dpi=1200)
    plt.close()


def probability_scatterplot(
    label,
    probabilities,
    probabilities2,
    ax=None,
    fname=None,
    binwidth=0.01,
    x="Probability",
    y="Standard deviation",
    hue="Label",
    xlim=[0.0, 1.0],
    ylim=None,
):
    label = ["benign" if l == 0 else "malignant" for l in label]
    data_plot = pd.DataFrame({hue: label, x: probabilities, y: probabilities2})
    plt.figure()
    g = sns.jointplot(
        data=data_plot,
        x=x,
        y=y,
        hue=hue,
        s=10,
        xlim=xlim,
        ylim=ylim,
        palette=[(5 / 255, 126 / 255, 255 / 255), (251 / 255, 4 / 255, 79 / 255)],
        alpha=0.8,
    )

    if fname is not None:
        plt.gcf().savefig(fname, bbox_inches="tight", dpi=600)
    plt.close()
