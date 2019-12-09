# Author(s): Silvio Gregorini (silviogregorini@openforce.it)
# Copyright 2019 Silvio Gregorini (github.com/SilvioGregorini)
# License LGPL-3.0 or later (https://www.gnu.org/licenses/lgpl).

import time

from wealth_distribution_test import Population


if __name__ == '__main__':
    population = Population(
        allow_duplicates=True,
        days=365,
        loss=100,
        max_actors=2,
        max_winners=1,
        meritocracy=True,
        num=1000,
        precision=2,
        random_loss=True,
        random_win=True,
        show_graph_after_days=30,
        starting_max_wealth=100,
        starting_min_wealth=100,
        win=100,
    )
    population.set_population_wealth()
    population.make_transactions()
    population.set_results()
    population.print_results()
    time.sleep(1)
