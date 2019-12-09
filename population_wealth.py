# Author(s): Silvio Gregorini (silviogregorini@openforce.it)
# Copyright 2019 Silvio Gregorini (github.com/SilvioGregorini)
# License LGPL-3.0 or later (https://www.gnu.org/licenses/lgpl).

import csv
import sys

from collections import Counter
from math import floor
from random import randint, uniform

from frozen_dict import FrozenDict
from graph import Graph


POPULATION_ATTRS = {
    'allow_duplicates': (bool,),
    'days': (int,),
    'equal_start': (bool,),
    'max_actors': (int,),
    'max_loss': (float, int),
    'max_win': (float, int),
    'max_winners': (int,),
    'meritocracy': (bool,),
    'num': (int,),
    'precision': (int,),
    'random_loss': (bool,),
    'random_win': (bool,),
    'show_graph_after_days': (int,),
    'starting_max_wealth': (float, int),
    'starting_min_wealth': (float, int),
}


def get_population_vals():
    """
    Finds vals for Population constructor when launching script from terminal
    """
    vals = {}

    # If there's a config file, read it and create `vals` from it
    if any(a.startswith('--csv_config=') for a in sys.argv[1:]):
        # If multiple paths are defined, get only the first one
        file_path = [a.replace('--csv_config=', '') for a in sys.argv[1:]][0]
        with open(file_path, 'rt') as csvf:
            for attr, val in csv.reader(csvf, delimiter=';', quotechar='"'):
                val = eval(val)
                if attr in POPULATION_ATTRS:
                    if isinstance(val, POPULATION_ATTRS[attr]):
                        vals[attr] = val
                    else:
                        raise ValueError(
                            "'{}' parameter should be {}, not {}"
                            .format(attr, POPULATION_ATTRS[attr], type(val))
                        )
                else:
                    print(">>> Unknown parameter in config file: " + attr)

    # Then, update `vals` from command line parameters, eventually overriding
    # the config file
    for arg in sys.argv[1:]:
        for attr_name, attr_type in POPULATION_ATTRS.items():
            arg_name = '--{}='.format(attr_name)
            if arg.startswith(arg_name):
                try:
                    attr = eval(arg.replace(arg_name, ''))
                except:
                    attr = None
                if not isinstance(attr, attr_type):
                    raise ValueError(
                        "--{} parameter should be {}, not {}"
                        .format(attr_name, attr_type, type(attr))
                    )
                vals[attr_name] = attr
    return vals


class Population:
    """
    This class simulates a population and the people's wealth.

    Parameters that can be passed to the constructor:
    > 'allow_duplicates': bool, defines if a single person can enter more
        than 1 transaction per day
    > 'days': int, defines for how many days transactions take place
    > 'equal_start': bool, defines whether everyone start with same wealth
    > 'max_actors': int, max number of people taking part into a transaction
    > 'max_loss': float or int, defines loss percentage for losing transactions
    > 'max_win': float or int, defines win percentage for losing transactions
    > 'max_winners': int, max number of winners in a transaction
    > 'meritocracy': bool, gives wealthier persons more chance to win
        transactions
    > 'num': int, population number
    > 'precision': int, number of allowed decimal digits
    > 'random_loss': bool, use random loss percentage (lower than fixed loss)
    > 'random_win': bool, use random win percentage (lower than fixed win)
    > 'starting_max_wealth': float or int, max wealth allowed at start
    > 'starting_min_wealth': float or int, min wealth allowed at start
    """

    def __init__(self, **kwargs):
        self.allow_duplicates = kwargs.get('allow_duplicates', True)
        self.days = kwargs.get('days') or 365
        self.max_actors = kwargs.get('max_actors') or 2
        self.max_loss = kwargs.get('max_loss') or 100
        self.max_win = kwargs.get('max_win') or 100
        self.max_winners = kwargs.get('max_winners') or 1
        self.meritocracy = kwargs.get('meritocracy', True)
        self.num = kwargs.get('num') or 1000
        self.precision = kwargs.get('precision') or 2
        self.random_loss = kwargs.get('random_loss', True)
        self.random_win = kwargs.get('random_win', True)
        self.show_graph_after_days = kwargs.get('show_graph_after_days') or 7
        self.starting_max_wealth = kwargs.get('starting_max_wealth') or 100
        self.starting_min_wealth = kwargs.get('starting_min_wealth') or 100

        # Define attributes that can be computed now
        self.equal_start = kwargs.get('equal_start', False) or self.is_zero(
            self.starting_max_wealth - self.starting_min_wealth
        )
        self.name = "Population data:\n    {}".format(
            "\n    ".join([
                "{}: {}".format(attr_name, attr)
                for attr_name, attr in self.read().items()
            ])
        )

        # Defines attributes that should be computed later
        self.graph = None
        self.people_wealth = {}
        self.people_wealth_start = {}
        self.results = {}

        # Check attributes
        if self.days < 1:
            raise ValueError("Days can't be lower than 1.")
        if not (0 <= self.max_loss <= 100) \
                or self.max_win < 0 or self.max_win < self.max_loss:
            raise ValueError(
                "Loss percentage can't be lower than 0 or higher than 100."
                " Win percentage can't be lower than 0 and must be higher "
                " than loss percentage."
            )
        if self.max_actors < 2 or self.max_actors > self.num:
            raise ValueError(
                "Max actors number can't be lower than 2 nor higher than"
                " the population itself."
            )
        if not 0 < self.max_winners < self.max_actors:
            raise ValueError(
                "Max winners number must be higher than 0 (at least 1 winner)"
                " and lower than max actors number (at least 1 loser)."
            )
        if self.num < 2:
            raise ValueError("Population can't be lower than 2.")
        if self.precision < 0:
            raise ValueError(
                "Precision can't be lower than 0. Set precision to 0 if you"
                " want only integers, else set the number of decimal places"
                " you want to use."
            )
        if not 0 <= self.starting_min_wealth <= self.starting_max_wealth:
            raise ValueError(
                "Starting max and min wealth must both be >= 0, and max wealth"
                " can't be lower than min wealth."
            )

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return self.name

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return repr(self)

    def compare_floats(self, value1, value2):
        """
        Compares value1 and value2.
        Returns 0 if value1 and value2 are the same value up to ``precision``
        digits, else returns 1 if value1 > value2 or -1 if value1 < value2
        """
        if self.is_zero(value1 - value2):
            return 0
        elif value1 > value2:
            return 1
        else:
            return -1

    @staticmethod
    def get_population_attrs():
        return POPULATION_ATTRS

    @staticmethod
    def get_random_perc(perc):
        """
        Returns a percentage `p` such that 0 < p <= perc
        Instead of computing `p` directly, a random integer is chosen in the
        interval [1, perc * 10 ** (perc digits)].
        Then the value is normalized to its default magnitude.
        I.e.:
            perc = 1.258 => return randint(1, 1258) / 1000
        """
        magnitude = 0
        normalized_perc = perc * 10 ** magnitude
        while int(normalized_perc) != normalized_perc:
            magnitude += 1
            normalized_perc = perc * 10 ** magnitude
        return randint(1, normalized_perc) / 10 ** magnitude

    def get_transaction_people(self, person1, done):
        """ Gets people to be used in transactions """
        if self.skip_person(person1, done=done):
            return

        def get_new_person():
            person2 = randint(1, self.num)
            while self.skip_person(person2, others=people, done=done):
                person2 = randint(1, self.num)
            return person2

        people = (person1,)
        # NB: `-1` because we already have `person1` in `people`, so there's
        # already an actor
        actors_num = min(randint(2, self.max_actors), self.num - len(done)) - 1
        while actors_num:
            people += (get_new_person(),)
            actors_num -= 1

        return people

    def get_transaction_vals(self, people):
        """
        Computes transactions between people and
        :param people: tuple of integers (each one representing one person)
        :return result: dict {person: person's wealth after transactions}
        """
        result = {p: self.people_wealth[p] for p in people}

        # NB: not calling variables `winners` and `losers` because not every
        # loser gets to pay its quota (a loser can't pay if has no wealth)
        gainers = self.get_winners(people)
        payers = tuple([
            p for p in people
            if p not in gainers and not self.is_zero(result[p])
        ])
        if not payers:
            return result

        win, loss = self.get_win_loss_perc()

        gainers_wealth = sum(result[p] for p in gainers)
        payers_wealth = sum(result[p] for p in payers)
        comparison = self.compare_floats(gainers_wealth, payers_wealth)
        if comparison == 1:
            perc = loss
        elif comparison == -1:
            perc = win
        else:
            if len(gainers) >= len(payers):
                perc = loss
            else:
                perc = win

        # `val` is the transaction total amount
        val = payers_wealth * perc / 100

        # Give every gainer the same amount
        gain = val / len(gainers)
        for p in gainers:
            result[p] += gain

        to_pay = val / len(payers)
        has_paid = ()

        # Define those who can't pay the whole amount; make them pay what they
        # can, set them into `has_paid` while lowering `val` each time someone
        # pays. Then, recompute the single person's payment and go through this
        # again until payment is low enough to be affordable to all those who
        # still have to pay their share.
        cant_pay = tuple([
            p for p in payers if result[p] and result[p] <= to_pay
        ])
        while cant_pay:
            for p in cant_pay:
                has_paid += (p,)
                val -= result[p]
                result[p] = 0
            to_pay = val / len(payers)
            cant_pay = tuple([
                p for p in payers if result[p] and result[p] <= to_pay
            ])
        for p in tuple(set(payers) - set(has_paid)):
            result[p] -= to_pay

        return result

    def get_win_loss_perc(self):
        win = self.max_win
        if self.random_win:
            win = Population.get_random_perc(win)
        loss = self.max_loss
        if self.random_loss:
            loss = Population.get_random_perc(loss)
        return win, loss

    def get_winners(self, people):
        winners = ()

        # Winners num is upper-bounded
        winners_num = randint(1, self.max_winners)
        p_wealth = self.round(sum(self.people_wealth[p] for p in people))

        if not self.meritocracy or self.is_zero(p_wealth):
            # Choose people in a completely random way
            while winners_num:
                winners += (people[randint(0, len(people) - 1)], )
                winners_num -= 1
            # Avoid repetitions in `winners`
            return tuple(set(winners))

        # Probability is defined as (inf, sup, person) for every person, where
        # 0 <= inf < sup <= 1.
        # This way, we'll extract a random float between 0 and 1, which defines
        # who wins.
        # I.e.:
        #     probability = [
        #           inf     sup  person
        #         (0.00,   0.20,      1),
        #         (0.20,   0.75,      2),
        #         (0.75,   1.00,      3)
        #     ]
        #     res = 0.8518519551928782
        #     => person 3 wins
        #
        # NB: if `res` is an overlapping value for some inf-sup consecutive
        # couples (in the above example, 0.20 or 0.75), the person with `res`
        # as `inf` wins.
        # I.e.:
        #     probability = [
        #           inf     sup  person
        #         (0.00,   0.20,      1),
        #         (0.20,   0.75,      2),
        #         (0.75,   1.00,      3)
        #     ]
        #     res = 0.20
        #     => person 2 wins
        probability = []
        for num, person in enumerate(people):
            # People with 0 wealth have no chance to win a transaction.
            # That would be non-meritocratic, of course.
            if not self.people_wealth[person]:
                continue

            if not probability:
                inf = 0
            else:
                inf = max(s for (i, s, p) in probability)
            sup = inf + (self.people_wealth[person] / p_wealth)
            probability.append((inf, sup, person))

        # Get every winner (each one may appear multiple times)
        while winners_num:
            res = uniform(0, 1)
            for min_prob, max_prob, person in probability:
                # Checking equality only on lower value because of:
                #   1. duplicates, since a person's lower value can be
                #      another person's higher value;
                #   2. function `uniform`'s known issue of not always
                #      including second parameter in possible outputs
                if min_prob <= res < max_prob:
                    winners += (person, )
                    winners_num -= 1

        # Remove repetitions in `winners`
        return tuple(set(winners))

    def is_zero(self, value):
        return abs(self.round(value)) < 10 ** (- self.precision)

    def make_single_transaction(self, people):
        self.people_wealth.update(self.get_transaction_vals(people))

    def make_transactions(self):
        for day in range(1, self.days + 1):
            done = ()
            for person in self.people_wealth.keys():
                people = self.get_transaction_people(person, done)

                # Make transactions only if there are 2 or more people to
                # actually interact with each other
                if people and len(people) > 1:
                    self.make_single_transaction(people)

                    # No need to update `done` if duplicates are allowed
                    if not self.allow_duplicates:
                        done += people

            print("Day {} completed".format(day))
            self.update_graph()
            if day == self.days:
                # Show graph after last day
                self.graph.show()
            elif self.show_graph_after_days > 0 \
                    and not day % self.show_graph_after_days:
                self.graph.show()

    def print_results(self):
        prec = self.precision
        fmt = '.{}f'.format(prec)
        res = sorted([self.round(x) for x in self.people_wealth.values()])

        print("\n* RESULTS *\n")
        print(
            ">>> Ordered distribution:\n    " + '; '.join(
                "{}: {}".format(format(k, fmt), v)
                for k, v in dict(Counter(res)).items()
            )
        )
        print(">>> " + str(self))
        print(">>> Total wealth: {}".format(format(
            self.round(sum(res)), fmt
        )))
        print(">>> Average wealth: " + format(
            self.results['avg_wealth'], fmt
        ))
        print(">>> Median wealth: " + format(
            self.results['median_wealth'], fmt
        ))
        print(
            ">>> Number of people who are poorer than the beginning: {} ({}%)"
            .format(str(self.results['poorer_than_begin']),
                    format(self.results['poorer_than_begin_perc'], fmt))
        )
        print(
            ">>> Number of people who are poorer than the average: {} ({}%)"
            .format(str(self.results['poorer_than_avg']),
                    format(self.results['poorer_than_avg_perc'], fmt))
        )
        print(
            ">>> Gini coefficient: " + format(self.graph.gini_coeff, fmt)
        )

    def read(self):
        return {
            attr_name: getattr(self, attr_name)
            for attr_name in sorted(list(self.get_population_attrs().keys()))
        }

    def round(self, value, force_precision=None):
        """ Rounds `value` according to `precision` attribute """
        return round(value, ndigits=force_precision or self.precision)

    def skip_person(self, person, others=None, done=None):
        """
        Checks whether given `person` should be skipped from current
        transaction.
        """

        # `done` should always be empty if `allow_duplicates` is True
        if self.allow_duplicates:
            done = []
        done = set(done or [])
        if person in done:
            return True
        # Check if every other person in given population has already partaken
        # in some transaction
        elif set(self.people_wealth.keys()) - done == {person}:
            return True

        # Avoid using twice the same person within the same transaction
        others = tuple(others or [])
        if person in others:
            return True

        return False

    def set_population_wealth(self):
        """ Generates population wealth before starting any transactions """
        if self.equal_start:
            w = self.round(
                (self.starting_min_wealth + self.starting_max_wealth) / 2
            )
            self.people_wealth = {
                x: w
                for x in range(1, self.num + 1)
            }

        else:
            if self.compare_floats(
                self.starting_min_wealth, self.starting_max_wealth
            ) > 1:
                raise ValueError(
                    "Starting minimum wealth can't be lower than starting"
                    " maximum wealth."
                )
            self.people_wealth = {
                x: randint(self.starting_min_wealth, self.starting_max_wealth)
                for x in range(1, self.num + 1)
            }

        # Copy starting wealth to allow comparison after transactions
        self.people_wealth_start = FrozenDict(self.people_wealth)
        self.update_graph()
        self.graph.show()

    def set_results(self):
        """
        Compute population results after all transactions are done.
        """
        wealths = tuple(sorted(self.people_wealth.values()))
        total_wealth = sum(wealths)

        # Get average and median wealth
        avg_wealth = total_wealth / self.num
        pos = floor(self.num / 2)
        if self.num % 2:
            # Median for an odd number of values
            median_wealth = wealths[pos]
        else:
            # Median for an even number of values
            median_wealth = (wealths[pos - 1] + wealths[pos]) / 2

        poorer_than_avg = len([w for w in wealths if w <= avg_wealth])
        poorer_than_avg_perc = poorer_than_avg / self.num * 100

        poorer_than_begin = len([
            p for p, w in self.people_wealth.items()
            if w <= self.people_wealth_start[p]
        ])
        poorer_than_begin_perc = poorer_than_begin / self.num * 100

        self.results = FrozenDict(
            avg_wealth=avg_wealth,
            median_wealth=median_wealth,
            people_by_wealth=self.graph.y_quants_data,
            poorer_than_avg=poorer_than_avg,
            poorer_than_avg_perc=poorer_than_avg_perc,
            poorer_than_begin=poorer_than_begin,
            poorer_than_begin_perc=poorer_than_begin_perc,
            wealth_by_people=self.graph.x_quants_data,
        )

    def update_graph(self):
        self.graph = Graph(data=self.people_wealth, quants=10)


Population.__init__.__doc__ = Population.__doc__
