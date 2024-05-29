from collections import defaultdict
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics


# Useful to uniquely identify metrics across functions
class Metric(Enum):
    ACC = 'Accuracy'
    SEN = 'Sensitivity'
    SPE = 'Specificity'
    PPV = 'PPV/Precision'  # precision / positive predictive value
    NPV = 'NPV'  # negative predictive value
    PEOPLE = 'People included'
    BRIER = 'Brier Score'
    ECE = 'ECE'


def plot_mean_std(joined_df: pd.DataFrame, diagnosis_arr: list, save_fig: Union[None, str] = None) -> None:
    plt.subplots(figsize=(10, 5))

    for diagnosis in diagnosis_arr:
        tmp_df = joined_df[joined_df.diagnosis == diagnosis]

        plt.scatter(tmp_df['std'], tmp_df['mean'], label=diagnosis, s=6)

    plt.legend()
    plt.xlabel('Standard Deviation', fontsize=15)
    plt.ylabel('Mean', fontsize=15)
    
    if save_fig is not None:
        plt.tight_layout()
        plt.savefig(save_fig)
    plt.show()
    plt.close()


def plot_all_roc_curves(mcdrop_df: pd.DataFrame, singlpass_df: pd.DataFrame) -> None:
    def plot_roc_stuff(tmp_df, label, color):
        preds = tmp_df['mean']
        fpr, tpr, thresholds = metrics.roc_curve(tmp_df['diagnosis'], preds)
        roc_auc = metrics.auc(fpr, tpr)

        # Youden’s J statistic
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        print(f'Best treshold for {label}: {best_thresh}')

        plt.plot(fpr, tpr, 'b', label=f'AUC {label} = %0.2f' % roc_auc, color=color)
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best {label}')

    plt.subplots(figsize=(20, 10))
    plt.title('Receiver Operating Characteristic')
    plot_roc_stuff(mcdrop_df, 'MC-Drop', 'orange')
    plot_roc_stuff(singlpass_df, 'Single Pass', 'green')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.close()


def plot_all_pr_curves(mcdrop_df: pd.DataFrame, singlpass_df: pd.DataFrame) -> None:
    def plot_pr_stuff(tmp_df, label, color):
        preds = tmp_df['mean']
        precision, recall, thresholds = metrics.precision_recall_curve(tmp_df['diagnosis'], preds)

        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        print(f'Best Threshold for {label}=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label=f'Best {label}')
        plt.plot(recall, precision, marker='.', label=f'{label}', color=color)

    plt.subplots(figsize=(20, 10))
    plt.title('PR-Curve')
    plot_pr_stuff(mcdrop_df, 'MC-Drop', 'orange')
    plot_pr_stuff(singlpass_df, 'Single Pass', 'green')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    plt.close()

# From: https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing
def calc_bins(preds, labels):
    # Assign each prediction to a bin
    num_bins = 15
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

# From: https://colab.research.google.com/drive/1H_XlTbNvjxlAXMW5NuBDWhxF3F2Osg1F?usp=sharing
def get_metrics(preds, labels):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def populate_arrs_for_df(df: pd.DataFrame, metrics_dict: defaultdict, threshold: float = 0.5, with_ece=False):
    probs = df['mean'].copy().values
    probs[probs < threshold] = 0
    probs[probs >= threshold] = 1

    tn, fp, fn, tp = metrics.confusion_matrix(df['diagnosis'].values, probs).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    metrics_dict[Metric.ACC.name].append(metrics.accuracy_score(df['diagnosis'].values, probs))
    metrics_dict[Metric.SEN.name].append(sensitivity)
    metrics_dict[Metric.SPE.name].append(specificity)
    metrics_dict[Metric.PEOPLE.name].append(df.shape[0])
    metrics_dict[Metric.PPV.name].append(ppv)
    metrics_dict[Metric.NPV.name].append(npv)
    
    if with_ece:
        metrics_dict[Metric.ECE.name].append(get_metrics(probs, df['diagnosis'].values)[0])


def round_and_str(val: float) -> str:
    return str(round(val, 2))


def print_latex_performance(df: pd.DataFrame) -> None:
    df_metrics = defaultdict(list)
    populate_arrs_for_df(df, df_metrics, threshold=0.5, with_ece=True)

    print(round_and_str(df_metrics[Metric.ACC.name][0]), end=' & ')
    print(round_and_str(metrics.roc_auc_score(df['diagnosis'].values, df['mean'].values)), end=' & ')

    for met in [Metric.SEN, Metric.SPE, Metric.PPV]:
        print(round_and_str(df_metrics[met.name][0]), end=' & ')
    print(round_and_str(df_metrics[Metric.NPV.name][0]), end=' \\\\ \n')
    
    print(round_and_str(metrics.brier_score_loss(df['diagnosis'].values, df['mean'].values)))
    print(round_and_str(df_metrics[Metric.ECE.name][0]))


def plot_across_metrics(x_vals_mc, x_vals_1, mcdrop_metrics, single_metrics, x_label, save_fig=None):
    fig, axs = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for i, val in enumerate(Metric):
        if val.name in [Metric.PEOPLE.name, Metric.BRIER.name, Metric.ECE.name]:
            continue
        axs[i].plot(x_vals_mc, mcdrop_metrics[val.name], label=f'{val.value} - MC Drop')
        axs[i].plot(x_vals_1, single_metrics[val.name], label=f'{val.value} - Single')

    for ax in axs:
        # ax.set_xlabel(x_label)
        # ax.set_ylabel('Performance achieved')
        ax.legend()

    # Fake plot to encapsulate all others and plot common x-axis label
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(x_label, fontsize=15)

    plt.tight_layout()
    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches='tight', pad_inches=0)
    plt.show()


def print_title(title_str: str) -> None:
    print('#####################################################################################')
    print(f'################# {title_str}')
    print('#####################################################################################')


def plot_all_comparisons(joined_df: pd.DataFrame, single_pass: pd.DataFrame,
                         threshold: float, starting_num_people: int, save_plot: Union[None, str] = None) -> None:
    print_title('MC-Drop with uncertainty thresholding')

    people_std_mcdrop_metrics = defaultdict(list)

    joined_df = joined_df.sort_values(by=['std'])

    for i in np.arange(starting_num_people, len(joined_df) + 1, 1):
        tmp_df = joined_df.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_std_mcdrop_metrics, threshold=threshold)

    # Plotting it
    fig, ax = plt.subplots(figsize=(15, 5))
    for val in Metric:
        if val.name in [Metric.PEOPLE.name, Metric.BRIER.name, Metric.ECE.name]:
            continue
        ax.plot(people_std_mcdrop_metrics[Metric.PEOPLE.name], people_std_mcdrop_metrics[val.name], label=val.value)
    ax.set_xlabel(Metric.PEOPLE.value)
    ax.set_ylabel('Performance achieved - MC Drop')
    ax.legend()

    plt.show()

    ###############################
    print_title('MC-Drop vs Single-pass')

    joined_df = joined_df.sort_values(by='extremes')
    single_pass = single_pass.sort_values(by='extremes')

    people_delta_mcdrop_metrics, people_delta_1_metrics = defaultdict(list), defaultdict(list)

    for i in np.arange(starting_num_people, len(joined_df) + 1, 1):
        tmp_df = joined_df.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_delta_mcdrop_metrics, threshold=threshold)

        tmp_df = single_pass.iloc[:i, :]
        populate_arrs_for_df(tmp_df, people_delta_1_metrics, threshold=threshold)

    plot_across_metrics(people_delta_mcdrop_metrics[Metric.PEOPLE.name],
                        people_delta_1_metrics[Metric.PEOPLE.name],
                        people_delta_mcdrop_metrics,
                        people_delta_1_metrics,
                        'People included')

    ###############################
    delta_delta_mcdrop_metrics, delta_delta_1_metrics = defaultdict(list), defaultdict(list)

    # Finding a minimal delta to start the for loop
    ini_val = 0
    for delta_val in np.arange(0.01, 0.51, 0.001):
        t1_df = joined_df.loc[(joined_df['mean'] < delta_val) | (joined_df['mean'] > 1 - delta_val), :]
        t2_df = single_pass.loc[(single_pass['mean'] < delta_val) | (single_pass['mean'] > 1 - delta_val), :]
        if t1_df.shape[0] >= 4 and t2_df.shape[0] >= 4:
            ini_val = round(delta_val, 3)
            break

    # Now getting for different values of delta
    for delta_val in np.arange(ini_val, 0.51, 0.001):
        tmp_df = joined_df.loc[(joined_df['mean'] < delta_val) | (joined_df['mean'] > 1 - delta_val), :]
        populate_arrs_for_df(tmp_df, delta_delta_mcdrop_metrics, threshold=threshold)

        tmp_df = single_pass.loc[(single_pass['mean'] < delta_val) | (single_pass['mean'] > 1 - delta_val), :]
        populate_arrs_for_df(tmp_df, delta_delta_1_metrics, threshold=threshold)

    plot_across_metrics(np.arange(ini_val, 0.51, 0.001),
                        np.arange(ini_val, 0.51, 0.001),
                        delta_delta_mcdrop_metrics,
                        delta_delta_1_metrics,
                        'Delta',
                        save_fig=save_plot)

    ###############################
    print_title('All 3 approaches together')

    for val in Metric:
        if val.name in [Metric.PEOPLE.name, Metric.BRIER.name, Metric.ECE.name]:
            continue
        plt.subplots(figsize=(20, 5))
        plt.plot(people_delta_mcdrop_metrics[Metric.PEOPLE.name], people_delta_mcdrop_metrics[val.name],  # 'o-',
                 label='Considering delta - MC Drop')
        plt.plot(people_delta_1_metrics[Metric.PEOPLE.name], people_delta_1_metrics[val.name],  # '|-',
                 label='Considering delta - Single')
        plt.plot(people_std_mcdrop_metrics[Metric.PEOPLE.name], people_std_mcdrop_metrics[val.name],  # 'x-',
                 label='Considering uncertainty')
        plt.xlabel('People included')
        plt.ylabel(val.value)
        plt.legend()
        plt.show()
