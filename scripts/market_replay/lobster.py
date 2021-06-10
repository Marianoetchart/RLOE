import argparse
import datetime as dt
import json
import logging as log
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import psutil
from joblib import Parallel, delayed
from pathlib import Path
p = str(Path(__file__).resolve().parents[2])
sys.path.append(p)

from gymlob.simulator.wrapper.wrapper import abides
from gymlob.simulator.wrapper.configs.mr import get_lobster_marketreplay_config
from gymlob.simulator.util.liquidity_telemetry import create_orderbooks, bin_and_sum, np_bar_plot_hist_input


def make_plots(plot_inputs, title=None, out_file="plot.png"):
    fig, axes = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [3, 3, 3, 3]})
    fig.set_size_inches(h=9, w=15)

    date = plot_inputs['mid_price'].index[0].date()
    midnight = pd.Timestamp(date)
    xmin = midnight + pd.to_timedelta("10:00:00")
    xmax = midnight + pd.to_timedelta("16:00:00")

    #  top plot -- mid price
    plot_inputs['mid_price'][xmin:xmax].plot(ax=axes[0], color='black', label="Mid price")
    axes[0].xaxis.set_visible(False)
    axes[0].legend(fontsize='large')
    axes[0].set_ylabel("Mid-price ($)", fontsize='large')
    axes[0].set_xlim(xmin, xmax)

    # spread
    plot_inputs['spread'][xmin:xmax].plot(ax=axes[1], color='black', label="Spread")
    axes[1].xaxis.set_visible(False)
    axes[0].legend(fontsize='large')
    axes[1].set_ylabel("Spread ($)", fontsize='large')
    axes[1].set_xlim(xmin, xmax)

    # order volume imbalance
    plot_inputs['order_volume_imbalance'][xmin:xmax].plot(ax=axes[2], color='black', label="Order volume imbalance")
    axes[2].xaxis.set_visible(False)
    axes[0].legend(fontsize='large')
    axes[2].set_ylabel("Volume Imbalance")
    axes[2].set_xlim(xmin, xmax)

    axes[3].bar(plot_inputs['transacted_volume']['center'], plot_inputs['transacted_volume']['counts'], align='center',
                width=plot_inputs['transacted_volume']['width'], fill=False)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes[3].tick_params(axis='both', which='major', labelsize=10)
    axes[3].set_ylabel("Transacted volume", fontsize='large')
    axes[3].set_xlim(xmin, xmax)

    if title:
        plt.suptitle(title, fontsize=12, y=0.925)

    plt.subplots_adjust(hspace=0.02)
    plt.show()
    fig.savefig(out_file, format='png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.03)


def run_marketreplay(security, date):

    log.info('Security: {}, Date: {}'.format(security, date))

    agents, agent_count = get_lobster_marketreplay_config(security=security,
                                                          date=date,
                                                          book_freq=0,
                                                          log_orders=True)

    simulation_name = '{}_{}'.format(security, date)
    log_folder = f'{LOG_FOLDER}/{simulation_name}/'

    abides(name=simulation_name,
           agents=agents,
           date=date,
           stop_time=datetime.strptime('{} 16:30:00'.format(date), '%Y-%m-%d %H:%M:%S'),
           log_folder=LOG_FOLDER,
           skip_log=False)

    log.info("{} {}: processing the orderbook data...".format(security, date))

    baseline_exchange_path = f'{log_folder}/EXCHANGE_AGENT.bz2'
    baseline_ob_path = f'{log_folder}/ORDERBOOK_{security}_FULL.bz2'

    _, transacted_orders, cleaned_orderbook = create_orderbooks(baseline_exchange_path, baseline_ob_path)

    counts, center, width = np_bar_plot_hist_input(bin_and_sum(transacted_orders["SIZE"], 60))

    plot_inputs = {
        "mid_price": cleaned_orderbook["MID_PRICE"],
        "spread": cleaned_orderbook["SPREAD"],
        "order_volume_imbalance": cleaned_orderbook["ORDER_VOLUME_IMBALANCE"],
        "transacted_volume": {
            'center': center,
            'width': width,
            'counts': counts
        }
    }

    log.info("{} {}: visualising the orderbook data...".format(security, date))

    make_plots(plot_inputs, title=f'{security}_{date}', out_file=f'{log_folder}/{security}_{date}.png')
    cleaned_orderbook.to_pickle(f'{log_folder}/{security}_{date}_orderbook.bz2')

    log.info("{} {}: processing completed.".format(security, date))


if __name__ == "__main__":
    """
    market replay
    """

    script_start_time = dt.datetime.now()
    log.basicConfig(level=log.DEBUG)

    LOG_FOLDER = '/efs/mm/gymlob/log/'
    SEED = 5422608
    np.random.seed(SEED)

    SECURITIES = ['TSLA']
    DATES = ['2015-01-02', '2015-01-05', '2015-01-06']

    num_jobs = len(SECURITIES)*len(DATES) if len(SECURITIES)*len(DATES) < psutil.cpu_count() else psutil.cpu_count()
    log.info('Total Number of Market Replay Runs: {}'.format(num_jobs))

    Parallel(n_jobs=num_jobs, backend='multiprocessing')(delayed(run_marketreplay)(security, date)
                                                         for security in SECURITIES
                                                         for date in DATES)
    script_end_time = dt.datetime.now()
    log.info('Total time taken for the experiments: {}'.format(script_end_time - script_start_time))