# Author(s): Silvio Gregorini (silviogregorini@openforce.it)
# Copyright 2019 Silvio Gregorini (github.com/SilvioGregorini)
# License LGPL-3.0 or later (https://www.gnu.org/licenses/lgpl).

from collections import namedtuple

from numpy import quantile, trapz
from matplotlib import pyplot as plt

Row = namedtuple('Row', ['seq', 'freq', 'freqp', 'freqc', 'freqcp'])


class Graph:

    def __init__(self, data, quants):
        self.data = {
            # Normalize data by ascending values
            k: v for k, v in sorted(data.items(), key=lambda item: item[1])
        }
        self.quants = quants

    def get_gini_coeff(self):
        return 1 - (2 * trapz(self.y_normalized, dx=1 / self.x_tot))

    def get_x_quants_data(self):
        x_quants = [float(q) for q in quantile(
            self.x, [.0] + [q / self.quants for q in range(1, self.quants + 1)]
        )]
        x_quants_data = {}
        for seq in range(1, self.quants + 1):
            x_qfreq = sum([
                self.data[x] for x in self.x
                if x_quants[seq - 1] <= x < x_quants[seq]
            ])
            if seq == self.quants:
                # Add last value, otherwise left out
                x_qfreq += self.data[x_quants[seq]]
            x_qfreqp = x_qfreq / self.y_tot
            x_qfreqc = x_qfreq
            if x_quants_data.get(seq - 1):
                x_qfreqc += x_quants_data.get(seq - 1).freqc
            x_qfreqpc = x_qfreqc / self.y_tot
            x_quants_data[seq] = Row(
                seq=seq,
                freq=x_qfreq,
                freqp=x_qfreqp,
                freqc=x_qfreqc,
                freqcp=x_qfreqpc,
            )
        return x_quants_data

    def get_y_quants_data(self):
        y_quants = list(quantile(
            self.y, [.0] + [q / self.quants for q in range(1, self.quants + 1)]
        ))
        y_quants_data = {}
        for seq in range(1, self.quants + 1):
            y_qfreq = len([
                x for x in self.x
                if y_quants[seq - 1] <= self.data[x] <= y_quants[seq]
            ])
            y_qfreqp = y_qfreq / self.x_tot
            y_qfreqc = y_qfreq
            if y_quants_data.get(seq - 1):
                y_qfreqc += y_quants_data.get(seq - 1).freqc
            y_qfreqpc = y_qfreqc / self.x_tot
            y_quants_data[seq] = Row(
                seq=seq,
                freq=y_qfreq,
                freqp=y_qfreqp,
                freqc=y_qfreqc,
                freqcp=y_qfreqpc,
            )
        return y_quants_data

    def show(self):
        plt.scatter(self.x_normalized, self.y_normalized)
        plt.show()

    gini_coeff = property(
        fget=get_gini_coeff
    )

    x = property(
        fget=lambda self: sorted(list(self.data.keys()))
    )

    x_normalized = property(
        fget=lambda self: [x / self.x_tot for x in self.x]
    )

    x_quants_data = property(
        fget=get_x_quants_data
    )

    x_tot = property(
        fget=lambda self: len(self.data.keys())
    )

    y = property(
        fget=lambda self: sorted(list(self.data.values()))
    )

    y_normalized = property(
        fget=lambda self: [sum(self.y[:n + 1]) / self.y_tot for n in range(self.x_tot)]
    )

    y_quants_data = property(
        fget=get_y_quants_data
    )

    y_tot = property(
        fget=lambda self: sum(self.data.values())
    )
