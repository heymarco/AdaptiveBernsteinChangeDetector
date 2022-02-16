import matplotlib.pyplot as plt


def move_legend_below_graph(axes, ncol: int, title: str):
    handles, labels = axes.flatten()[-1].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    plt.figlegend(handles, labels, loc='lower center', frameon=False, ncol=ncol, title=title)
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
