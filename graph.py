
import pickle 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.ticker import MaxNLocator
from data import * 

def plot_stored_data(path=f"uq/metrics/IIIC/cov_cls_plot_nbs{normalize_by_sample()}.pkl"):
    with open(path, 'rb') as file:
        df = pickle.load(file)
    
    # None

    # df = {"alpha": np.array(alphas), 
    #                         "acc_softmax": acc_softmax, 
    #                         "acc_cov_softmax": acc_cov_softmax,
    #                           "avg_len_softmax": avg_len_softmax, 
    #                           "avg_len_cov_softmax": avg_len_cov_softmax, 
    #                           "cc_base_accs": cc_base_accs, 
    #                           "cc_cov_accs": cc_cov_accs,
    #                             "cc_spcfc_accs": cc_spcfc_accs, 
    #                             "cc_base_lens": cc_base_lens,
    #                               "cc_cov_lens": cc_cov_lens,
    #                               "cc_spcfc_lens": cc_spcfc_lens, 
    #                               "cc_base_mean_lens": cc_base_mean_lens, 
    #                               "cc_cov_mean_lens": cc_cov_mean_lens,
    #                                 "cc_spcfc_mean_lens": cc_spcfc_mean_lens,
    #                                 "pset_len_softmax": pset_len_softmax,
    #                                 "pset_len_cov_softmax": pset_len_cov_softmax,
    #                                 "class_coverage_softmax": class_coverage_softmax,
    #                                 "class_coverage_cov_softmax": class_coverage_cov_softmax,
    #                                 "cc_base_cls_coverages": cc_base_cls_coverages,
    #                                 "cc_cov_cls_coverages": cc_cov_cls_coverages,
    #                                 "cc_spcfc_cls_coverages": cc_spcfc_cls_coverages}
    alphas = df["alpha"]
    acc_softmax = df["acc_softmax"]
    acc_cov_softmax = df["acc_cov_softmax"]
    cc_base_accs = df["cc_base_accs"]
    cc_cov_accs = df["cc_cov_accs"]
    cc_spcfc_accs = df["cc_spcfc_accs"]

    cc_base_lens = df["cc_base_lens"]
    cc_cov_lens = df["cc_cov_lens"]
    cc_spcfc_lens = df["cc_spcfc_lens"]
    pset_len_softmax = df["pset_len_softmax"]
    pset_len_cov_softmax = df["pset_len_cov_softmax"]
    
    class_coverage_softmax = df["class_coverage_softmax"]
    class_coverage_cov_softmax = df["class_coverage_cov_softmax"]
    cc_base_cls_coverages = df["cc_base_cls_coverages"]
    cc_cov_cls_coverages = df["cc_cov_cls_coverages"]
    cc_spcfc_cls_coverages = df["cc_spcfc_cls_coverages"]
    
    hist_titles = ["Softmax", 
                   "Softmax with Covariate Shift",
                     "Class Conditional Softmax",
                       "Class Conditional Softmax with Covariate Shift", 
                       "Class Conditional Softmax with Class Relevance Covariate Shift"]
    hist_plots = [pset_len_softmax, pset_len_cov_softmax, cc_base_lens, cc_cov_lens, cc_spcfc_lens]
    coverage_plots = [acc_softmax, acc_cov_softmax, cc_base_accs, cc_cov_accs, cc_spcfc_accs]
    class_coverage_plots = [class_coverage_softmax, class_coverage_cov_softmax, cc_base_cls_coverages, cc_cov_cls_coverages, cc_spcfc_cls_coverages]
    # Plot many histograms
    fig, axes = plt.subplots(nrows=5, ncols=len(alphas), figsize=(30,30))
    warm_color_palette = ["#FF5733", "#FFAA33", "#FFC733", "#FFEE33", "#FFD433"]
    cool_colors =  ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
    lw = 3
    transparency_factor = 0.9
    for i in range(len(hist_plots)):
        for j in range(len(alphas)):
            if j == 0:
                # only need labels for just 1
                axes[i,j].hist(hist_plots[i][j], bins=6, color=cool_colors[i], label=hist_titles[i])
            else: 
                axes[i,j].hist(hist_plots[i][j], bins=6, color=cool_colors[i])

            # plot alphas at the bottom of all the subplots
            # axes[i,j].set_title(hist_titles[i] + " Alpha: " + str(alphas[j]))
            axes[i,j].tick_params(axis='x', labelsize=24)
            axes[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[i,j].set_ylabel("# of Samples")
            # axes[i,j].set_xlabel("Prediction Set Lengths")
            if i== len(hist_plots) - 1:
                axes[i,j].set_xlabel("Size of Prediction Set \n Alpha: " + str(alphas[j]), fontsize=28)
    
    # legend_ax = fig.add_axes([0, 0, 1, 0.1]) 
    fig.legend(fontsize=32, loc = "lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    # fig.tight_layout()
    plt.savefig("fig/cp/IIIC/hist_cls_nbs" + str(normalize_by_sample()) + ".png")
    
    # compute adjusted means
    means_hist_alpha = np.zeros((len(hist_plots), len(alphas)))
    for i in range(len(hist_plots)):
        for j in range(len(alphas)):
            mean = 0
            for sample in range(len(hist_plots[i][j])):
                if hist_plots[i][j][sample] == 0:
                    mean += 6 # max length
                else: 
                    mean += hist_plots[i][j][sample]

            means_hist_alpha[i,j] = mean / len(hist_plots[i][j])
    
    # now let's plot them across alphas
    plt.figure(figsize=(8,8)) # big plot
    for i in range(len(hist_plots)):
        plt.plot(alphas, means_hist_alpha[i,:], label=hist_titles[i], color=cool_colors[i], lw=lw, alpha=transparency_factor)

    plt.title("Expected Prediction Set Lengths")
    plt.ylabel("Adjusted Mean Prediction Set Length")
    plt.xlabel("Alpha")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig("fig/cp/IIIC/adjusted_mean_len_cls_nbs" + str(normalize_by_sample()) + ".png")


    # plot the accuracy 
    plt.figure(figsize=(8,8)) # big plot
    for i in range(len(hist_plots)):
        plt.plot(alphas, coverage_plots[i], label=hist_titles[i], color=cool_colors[i], lw=lw, alpha=transparency_factor)
    
    plt.plot(alphas, 1- np.array(alphas), label="Ideal", color="black", lw=lw, alpha=transparency_factor)
    plt.title("Coverage")
    plt.ylabel("Coverage")
    plt.xlabel("Alpha")
    plt.tight_layout()
    plt.legend()
    
    plt.savefig("fig/cp/IIIC/acc_cls_nbs" + str(normalize_by_sample()) + ".png")



    # plot the ratio between accuracy and adjusted mean_lens
    plt.figure(figsize=(8,8)) # big plot
    for i in range(len(hist_plots)):
        if i  != 1:
            plt.plot(alphas, coverage_plots[i] / means_hist_alpha[i,:], label=hist_titles[i], color=cool_colors[i], lw=lw, alpha=transparency_factor)
        elif i == 1:
            plt.plot(alphas, coverage_plots[i] / means_hist_alpha[i,:], label=hist_titles[i], color=cool_colors[i], lw=lw *4, alpha=0.5)
    plt.title("Ratio of Coverage and Expected Prediction Set Lengths")
    plt.ylabel("Coverage / Adjusted Mean Prediction Set Length")
    plt.xlabel("Alpha")
    plt.tight_layout()
    plt.legend()
   
    plt.savefig("fig/cp/IIIC/ratio_acc_adj_mean_len_cls_nbs" + str(normalize_by_sample()) + ".png")

    # class mappings 
    class_labels = {
        0 : 'Seizure', 1 : 'LPD', 2 : 'GPD', 3 : 'LRDA', 4: 'GRDA', 5 :'Other'
    }
    class_labels = [class_labels[i] for i in range(6)]

    fig, axes = plt.subplots(nrows=len(alphas), figsize=(30,30))
    # plot the class coverage
    w = 0.3 # width
    x = np.arange(len(class_labels))
    for j in range(len(alphas)):
        data = {}
        for i in range(len(hist_titles)):
            data[hist_titles[i]] = class_coverage_plots[i][j]
        bar_plot(axes[j], data, colors=cool_colors, x_labels=class_labels)
        #     if j == 0:
        #         axes[j].bar(x + i*w, class_coverage_plots[i][j], color=cool_colors[i], label=hist_titles[i], width=w)
        #     else:
        #         axes[j].bar(x + i*w, class_coverage_plots[i][j], color=cool_colors[i], width=w)
        # # plot alphas at the bottom of all the subplots
        # # axes[i,j].set_title(hist_titles[i] + " Alpha: " + str(alphas[j]))
        # axes[j].tick_params(axis='x', labelsize=24)
        # # axes[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))
        # axes[j].set_ylabel("Coverage")
        # # axes[i,j].set_xlabel("Prediction Set Lengths")
        # axes[j].set_xticks(range(len(class_labels)), class_labels)
        # axes[j].set_xlabel("Class Type \n Alpha: " + str(alphas[j]), fontsize=28)
    
    # legend_ax = fig.add_axes([0, 0, 1, 0.1]) 
    # fig.legend(fontsize=32, loc = "lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    # fig.tight_layout()
    plt.savefig("fig/cp/IIIC/bar_cls_nbs" + str(normalize_by_sample()) + ".png")

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, x_labels=[]):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        bar = [None]
        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)], align='center')

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    ax.set_xticks(np.array(range(len(x_labels))), x_labels)
    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())