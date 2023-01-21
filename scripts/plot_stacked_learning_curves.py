# %reset -f
import argparse
import os
from enum import Enum
from glob import glob
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from implicit_kernel_meta_learning.parameters import FIGURES_DIR, GUILD_RUNS_DIR

sns.set_theme()

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 16

GUILD_RUNS_DIR = Path("/Users/daiki/.guild/runs")
def load_ids_from_batch(gid):
    """Load all results from a batch run"""
    # complete_dir = glob(str(GUILD_RUNS_DIR / gid) + "*")[0]
    complete_dirs = glob(str(GUILD_RUNS_DIR / gid) + "*")
    # all_dirs = glob(os.path.join(complete_dir, "*"))
    # all_dirs = [Path(p).name for p in all_dirs]
    # return all_dirs
    return complete_dirs


def load_result(gid):
    complete_dir = glob(str(GUILD_RUNS_DIR / gid) + "*")[0]
    result_path = os.path.join(complete_dir, "result.pkl")
    result = pd.read_pickle(result_path)
    return result


run_cols = ["meta_train_error", "meta_valid_error"]
num_cols = ["holdout_meta_valid_error", "holdout_meta_test_error"]


def npfy_results(result):
    run_cols = ["meta_train_error", "meta_valid_error"]
    for run_col in run_cols:
        result[run_col] = np.array(result[run_col])
    return result


def make_ts(results, col):
    new_list = []
    for result in results:
        new_list.append(result[col])
    return pd.DataFrame(data=np.stack(new_list))


# Get learning curves
palette = sns.color_palette()
figsize = (8, 6)


class ExperimentColormap(Enum):
    GAUSS = palette[2]
    BOCHNER = palette[3]
    MKL_GOOD = palette[4]
    LSQ_BIAS = palette[5]
    MAML = palette[7]
    R2D2 = palette[1]
    BOCHNER_SVM = palette[8]
    GAUSS_ORACLE = "black"


def plot_ci(timeseries, t, fig, ax, alpha, alpha_fill, **kwargs):
    # RMSE and not MSE
    timeseries = np.sqrt(timeseries)
    mean = timeseries.mean()
    std = timeseries.std()
    ax.plot(t, mean, alpha=alpha, **kwargs)
    ax.fill_between(
        t, (mean - std), (mean + std), alpha=alpha_fill, color=kwargs["color"]
    )
    return fig, ax


def prepare_oracle(results):
    errors = []
    for result in results:
        errors.append(result["holdout_meta_test_error"])
    errors = np.sqrt(np.array(errors)).squeeze()
    return errors


def plot_oracle(errors, fig, ax, alpha, alpha_fill, color, label):
    mean = errors.mean()
    std = errors.std()
    ax.axhline(mean, alpha=alpha, linestyle="--", color=color, label=label)
    ax.axhspan(mean - std, mean + std, alpha=alpha_fill, color=color)
    return fig, ax


lw = 2.0
alpha = 1.0
alpha_fill = 0.2


def main(
    mkl_id,
    lsq_bias_id,
    maml_id,
    r2d2_id,
    gauss_id,
    gauss_oracle_id,
    bochner_id,
    bochner_svm_id,
    bochner_gauss_id,
    y_upper_lim,
    y_lower_lim,
    output_dir,
):
    # Create directory
    output_dir = Path(output_dir)
    output_dir = FIGURES_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(nrows=2, figsize=(10,12), dpi=160, sharex='col',
                       gridspec_kw={'height_ratios': (1,2)})
    fig.subplots_adjust(hspace=0.0) 

    # if mkl_id:
    #     mkl_good_ids = load_ids_from_batch(mkl_id)
    #     mkl_good_results = []
    #     for gid in mkl_good_ids:
    #         mkl_good_results.append(load_result(gid))

    #     meta_val = make_ts(mkl_good_results, "meta_valid_error")
    #     meta_val_every = mkl_good_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     print("meta_val is {}, meta_val_every is {}, t is {}, gid is {}".format(meta_val, meta_val_every, t, mkl_id) )
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.MKL_GOOD.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="Gaussian MKL meta-KRR",
    #     )

    # if lsq_bias_id:
    #     lsq_bias_ids = load_ids_from_batch(lsq_bias_id)
    #     lsq_bias_results = []
    #     for gid in lsq_bias_ids:
    #         lsq_bias_results.append(load_result(gid))

    #     meta_val = make_ts(lsq_bias_results, "meta_valid_error")
    #     meta_val_every = lsq_bias_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.LSQ_BIAS.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="LS Biased Regularization",
    #     )

    # if maml_id:
    #     maml_ids = load_ids_from_batch(maml_id)
    #     maml_results = []
    #     for gid in maml_ids:
    #         maml_results.append(load_result(gid))

    #     meta_val = make_ts(maml_results, "meta_valid_error")
    #     meta_val_every = maml_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.MAML.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="MAML",
    #     )

    # if r2d2_id:
    #     r2d2_ids = load_ids_from_batch(r2d2_id)
    #     r2d2_results = []
    #     for gid in r2d2_ids:
    #         r2d2_results.append(load_result(gid))

    #     meta_val = make_ts(r2d2_results, "meta_valid_error")
    #     meta_val_every = r2d2_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.R2D2.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="R2D2",
    #     )

    # if gauss_id:
    #     gauss_ids = load_ids_from_batch(gauss_id)
    #     gauss_results = []
    #     for gid in gauss_ids:
    #         gauss_results.append(load_result(gid))

    #     meta_val = make_ts(gauss_results, "meta_valid_error")
    #     meta_val_every = gauss_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.GAUSS.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="Gaussian meta-KRR",
    #     )

    if bochner_id:
        bochner_ids = load_ids_from_batch(bochner_id)
        bochner_results = []
        for gid in bochner_ids:
            bochner_results.append(load_result(gid))

        meta_val = make_ts(bochner_results, "meta_valid_error")
        meta_val_every = bochner_results[0]["meta_val_every"]
        t = meta_val_every * np.arange(len(meta_val.T))
        timeseries = np.sqrt(meta_val)
        mean = timeseries.mean()
        std = timeseries.std()
        ax[0].plot(t, mean, alpha=alpha, label="KRR", color=ExperimentColormap.BOCHNER.value)
        ax[0].fill_between(
            t, (mean - std), (mean + std), alpha=alpha_fill, color=ExperimentColormap.BOCHNER.value
        )
        print("time series and its size are {}, {}, {}".format(timeseries.to_numpy(), timeseries.to_numpy().size, type(timeseries)))
        print("max and min are {}, {}".format(np.min(timeseries.to_numpy()), np.max(timeseries.to_numpy())))
        ax[0].set_ylim(np.min(timeseries.to_numpy()), np.nanmax(timeseries.to_numpy()))
        ax[0].spines['bottom'].set_visible(False)
        ax[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax[0].legend()
        ax[0].set_title('KRR, KSVM : beijing air quality\nkernel : CO detection')
        # fig, ax = plot_ci(
        #     meta_val,
        #     t,
        #     fig,
        #     ax,
        #     color=ExperimentColormap.BOCHNER.value,
        #     alpha=alpha,
        #     alpha_fill=alpha_fill,
        #     lw=lw,
        #     label="KRR",
        # )

    # if gauss_oracle_id:
    #     gauss_oracle_ids = load_ids_from_batch(gauss_oracle_id)
    #     gauss_oracle_results = []
    #     for gid in gauss_oracle_ids:
    #         gauss_oracle_results.append(load_result(gid))

    #     errors = prepare_oracle(gauss_oracle_results)
    #     fig, ax = plot_oracle(
    #         errors,
    #         fig,
    #         ax,
    #         alpha,
    #         alpha_fill,
    #         color=ExperimentColormap.GAUSS_ORACLE.value,
    #         label="Gaussian Oracle",
    #     )

    if bochner_svm_id:
        bochner_svm_ids = load_ids_from_batch(bochner_svm_id)
        bochner_svm_results = []
        for gid in bochner_svm_ids:
            bochner_svm_results.append(load_result(gid))

        meta_val = make_ts(bochner_svm_results, "meta_valid_error")
        meta_val_every = bochner_svm_results[0]["meta_val_every"]
        t = meta_val_every * np.arange(len(meta_val.T))
        timeseries = np.sqrt(meta_val)
        mean = timeseries.mean()
        std = timeseries.std()
        ax[1].plot(t, mean, alpha=alpha, label="KSVM", color=ExperimentColormap.BOCHNER_SVM.value)
        ax[1].fill_between(
            t, (mean - std), (mean + std), alpha=alpha_fill, color=ExperimentColormap.BOCHNER_SVM.value
        )
        ax[1].set_ylim(0, np.nanmax(timeseries.to_numpy()))
        ax[1].spines['top'].set_visible(False)
        ax[1].legend()
        
        # fig, ax = plot_ci(
        #     meta_val,
        #     t,
        #     fig,
        #     ax,
        #     color=ExperimentColormap.BOCHNER_SVM.value,
        #     alpha=alpha,
        #     alpha_fill=alpha_fill,
        #     lw=lw,
        #     label="SVM",
        # )
    
    # if bochner_gauss_id:
    #     bochner_gauss_ids = load_ids_from_batch(bochner_gauss_id)
    #     bochner_gauss_results = []
    #     for gid in bochner_gauss_ids:
    #         bochner_gauss_results.append(load_result(gid))

    #     meta_val = make_ts(bochner_gauss_results, "meta_valid_error")
    #     meta_val_every = bochner_gauss_results[0]["meta_val_every"]
    #     t = meta_val_every * np.arange(len(meta_val.T))
    #     fig, ax = plot_ci(
    #         meta_val,
    #         t,
    #         fig,
    #         ax,
    #         color=ExperimentColormap.BOCHNER_SVM.value,
    #         alpha=alpha,
    #         alpha_fill=alpha_fill,
    #         lw=lw,
    #         label="IKML_gauss",
    #     )

    # leg = ax.legend()
    # for line in leg.get_lines():
    #     line.set_linewidth(4.0)
    # ax.set_ylim((y_lower_lim, y_upper_lim))
    from matplotlib.path import Path as Path_
    d1 = 0.02 # X軸のはみだし量
    d2 = 0.03 # ニョロ波の高さ
    wn = 21   # ニョロ波の数（奇数値を指定）

    pp = (0,d2,0,-d2)
    px = np.linspace(-d1,1+d1,wn)
    py = np.array([1+pp[i%4] for i in range(0,wn)])
    p = Path_(list(zip(px,py)), [Path_.MOVETO]+[Path_.CURVE3]*(wn-1))

    line1 = mpatches.PathPatch(p, lw=4, edgecolor='black',
                            facecolor='None', clip_on=False,
                            transform=ax[1].transAxes, zorder=10)

    line2 = mpatches.PathPatch(p,lw=3, edgecolor='white',
                            facecolor='None', clip_on=False,
                            transform=ax[1].transAxes, zorder=10,
                            capstyle='round')

    a = ax[1].add_patch(line1)
    a = ax[1].add_patch(line2)
    fig.suptitle('generalization error', fontsize=30)
    # fig.suptitle('\n\nKRR : beijing air quality\nkernel : gas sensor')
    ax[1].set_ylabel("RMSE(generalization error)", loc='top', labelpad=20)
    ax[1].set_xlabel("Iteration")
    # ax.legend()
    fig.savefig(output_dir / "learning_curves_valid.pdf", format="pdf")
    fig.savefig(output_dir / "learning_curves_valid.png", format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mkl_id", type=str, default="")
    parser.add_argument("--lsq_bias_id", type=str, default="")
    parser.add_argument("--maml_id", type=str, default="")
    parser.add_argument("--r2d2_id", type=str, default="")
    parser.add_argument("--gauss_id", type=str, default="")
    parser.add_argument("--gauss_oracle_id", type=str, default="")
    parser.add_argument("--bochner_id", type=str, default="")
    parser.add_argument("--bochner_svm_id", type=str, default="")
    parser.add_argument("--bochner_gauss_id", type=str, default="")
    parser.add_argument("--y_upper_lim", type=float, default=60.0)
    parser.add_argument("--y_lower_lim", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    main(**vars(args))
