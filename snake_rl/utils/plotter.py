"""
Plot the training progress data collected by the Monitor.

"""

import os
import csv
import sys
import logging
import matplotlib.pyplot as plt

from snake_rl.algorithms import a2c_vae
from snake_rl.algorithms.baselines import a2c
from snake_rl.utils import init_logger, Monitor
from snake_rl.utils.misc import get_experiment_name

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue', 'yellow']

logger = logging.getLogger(os.path.basename(__file__))


def main():
    init_logger()

    experiments = [
        get_experiment_name('Snake-Simple-v0', 'a2c_v8'),
        get_experiment_name(a2c.CURRENT_ENV, a2c.CURRENT_EXPERIMENT),

        get_experiment_name(a2c_vae.CURRENT_ENV, a2c_vae.CURRENT_EXPERIMENT),
    ]

    for i in range(len(experiments)):
        experiment = experiments[i]
        x, y = [], []
        stats_filename = Monitor.stats_filename(experiment)

        if not os.path.exists(stats_filename):
            logger.info('Skipping %s, no stats file found...', stats_filename)
            continue

        with open(stats_filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x.append(int(row[0]))
                y.append(float(row[1]))

        skip_coeff = max(1, len(x) // 200)

        x_filtered, y_filtered = [], []
        for j in range(len(x)):
            if j % skip_coeff == 0:
                x_filtered.append(x[j])
                y_filtered.append(y[j])

        x = x_filtered
        y = y_filtered

        logger.info('Plotting %s...', experiment)
        plt.plot(x, y, color=COLORS[i], label=experiment)

    plt.title('Reward over time')
    plt.xlabel('Training step (batch #)')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
