# Author(s): Silvio Gregorini (silviogregorini@openforce.it)
# Copyright 2019 Silvio Gregorini (github.com/SilvioGregorini)
# License LGPL-3.0 or later (https://www.gnu.org/licenses/lgpl).

import time

from population_wealth import Population, get_population_vals


if __name__ == '__main__':
    population = Population(**get_population_vals())
    population.set_population_wealth()
    population.make_transactions()
    population.set_results()
    population.print_results()
    time.sleep(1)
