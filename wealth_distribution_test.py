import sys

from collections import Counter, OrderedDict
from math import floor, sqrt
from random import randint, uniform

POPULATION_ATTRS = {
    'allow_duplicates': (bool,),
    'days': (int,),
    'equal_start': (bool,),
    'loss': (float, int),
    'max_actors': (int,),
    'max_winners': (int,),
    'meritocracy': (bool,),
    'num': (int,),
    'precision': (int,),
    'random_loss': (bool,),
    'random_win': (bool,),
    'starting_max_wealth': (float, int),
    'starting_min_wealth': (float, int),
    'win': (float, int),
}


def get_population_vals():
    """
    Finds vals for Population constructor when launching script from terminal
    """
    vals = {}
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
    > 'loss': float or int, defines loss percentage for losing transactions
    > 'max_actors': int, max number of people taking part into a transaction
    > 'max_winners': int, max number of winners in a transaction
    > 'meritocracy': bool, gives wealthier persons more chance to win
        transactions
    > 'num': int, population number
    > 'precision': int, number of allowed decimal digits
    > 'random_loss': bool, use random loss percentage (lower than fixed loss)
    > 'random_win': bool, use random win percentage (lower than fixed win)
    > 'starting_max_wealth': float or int, max wealth allowed at start
    > 'starting_min_wealth': float or int, min wealth allowed at start
    > 'win': float or int, defines win percentage for losing transactions
    """

    def __init__(self, **kwargs):
        self.allow_duplicates = kwargs.get('allow_duplicates', True)
        self.days = kwargs.get('days', 365)
        self.loss = kwargs.get('loss', 100)
        self.max_actors = kwargs.get('max_actors', 2)
        self.max_winners = kwargs.get('max_winners', 1)
        self.meritocracy = kwargs.get('meritocracy', True)
        self.num = kwargs.get('num', 1000)
        self.precision = kwargs.get('precision', 2)
        self.random_loss = kwargs.get('random_loss', True)
        self.random_win = kwargs.get('random_win', True)
        self.starting_max_wealth = kwargs.get('starting_max_wealth', 100)
        self.starting_min_wealth = kwargs.get('starting_min_wealth', 100)
        self.win = kwargs.get('win', 100)

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
        self.people_wealth = {}
        self.people_wealth_start = {}
        self.results = {}

        # Check attributes
        if self.days < 1:
            raise ValueError("Days can't be lower than 1.")
        if not (0 <= self.loss <= 100) or self.win < 0 or self.win < self.loss:
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
        win = self.win
        if self.random_win:
            win = Population.get_random_perc(win)
        loss = self.loss
        if self.random_loss:
            loss = Population.get_random_perc(loss)
        return win, loss

    def get_winners(self, people):
        winners = ()

        # Winners num is upper-bounded
        winners_num = randint(1, self.max_winners)
        total_wealth = self.round(sum(self.people_wealth[p] for p in people))

        if not self.meritocracy or self.is_zero(total_wealth):
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
            sup = inf + (self.people_wealth[person] / total_wealth)
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
        return abs(self.round(value)) < 10 ** -self.precision

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

    def print_results(self):
        res = sorted([self.round(w) for w in self.people_wealth.values()])

        def fmt(s):
            f = '.{}f'.format(self.precision) if self.precision > 0 else ''
            return format(s, f)

        pop_by_deciles_list = [
            ('Decile', 'Range', '', 'Abs', 'Perc', 'Total wealth')
        ]
        col_lenghts = {x: len(pop_by_deciles_list[0][x]) + 1 for x in range(6)}
        for (min_d, max_d), d_vals in self.results['population_by_deciles'].items():
            descr = d_vals['decile']
            if len(descr) > col_lenghts[0]:
                col_lenghts[0] = len(descr) + 1
            min_wealth = fmt(self.round(min_d))
            if len(min_wealth) > col_lenghts[1]:
                col_lenghts[1] = len(min_wealth) + 1
            max_wealth = fmt(self.round(max_d))
            if len(max_wealth) > col_lenghts[2]:
                col_lenghts[2] = len(min_wealth) + 1
            abs_val = str(int(self.round(d_vals['abs'])))
            if len(abs_val) > col_lenghts[3]:
                col_lenghts[3] = len(abs_val) + 1
            perc_val = fmt(self.round(d_vals['perc']))
            if len(perc_val) > col_lenghts[4]:
                col_lenghts[4] = len(perc_val) + 1
            wealth = fmt(self.round(d_vals['tot_w']))
            if len(wealth) > col_lenghts[5]:
                col_lenghts[5] = len(wealth) + 1
            pop_by_deciles_list += [
                (descr, min_wealth, max_wealth, abs_val, perc_val, wealth),
            ]
        pop_by_deciles_str = ""
        title = True
        for array in pop_by_deciles_list:
            new_array = ()
            if title:
                for x in range(6):
                    item = array[x]
                    new_array += (item + " " * (col_lenghts[x] - len(item)),)
                    if x == 1:
                        new_array += ('  ',)
                title = False
            else:
                for x in range(6):
                    item = array[x]
                    new_array += (" " * (col_lenghts[x] - len(item)) + item,)
                    if x == 1:
                        new_array += ('->',)
            pop_by_deciles_str += "\n {} | {} {} {} | {} | {} | {}" \
                .format(*new_array)

        lower_percenter = self.results['lower_percenter']
        one_percenter = self.results['one_percenter']
        if len(lower_percenter) + len(one_percenter) == self.num:
            one_percenter_str = "The 1% has more wealth than anyone else" \
                                " combined."
        else:
            one_percenter_str = "The 1% has as much wealth as the lower {}%."\
                .format(
                    fmt(self.round(len(lower_percenter) / self.num * 100))
                )

        print("\n* RESULTS *\n")
        print(
            ">>> Ordered distribution:\n    " + '; '.join(
                "{}: {}".format(fmt(k), v)
                for k, v in dict(Counter(res)).items()
            )
        )
        print(">>> " + str(self))
        print(">>> Total wealth: {}".format(fmt(sum(res))))
        print(">>> Average wealth: " + fmt(self.results['avg_wealth']))
        print(">>> Median wealth: " + fmt(self.results['median_wealth']))
        print(
            ">>> Number of people who are poorer than the beginning: {} ({}%)"
            .format(str(self.results['poorer_than_begin']),
                    fmt(self.results['poorer_than_begin_perc']))
        )
        print(
            ">>> Number of people who are poorer than the average: {} ({}%)"
            .format(str(self.results['poorer_than_avg']),
                    fmt(self.results['poorer_than_avg_perc']))
        )
        print(
            ">>> Number of people who are poorer than the median: {} ({}%)"
            .format(str(self.results['poorer_than_med']),
                    fmt(self.results['poorer_than_med_perc']))
        )
        print(">>> " + one_percenter_str)
        print(">>> Population by deciles:" + pop_by_deciles_str)

    def read(self):
        return {
            attr_name: getattr(self, attr_name)
            for attr_name in sorted(list(self.get_population_attrs().keys()))
        }

    def round(self, value):
        """ Rounds `value` according to `precision` attribute """
        return round(value, self.precision)

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
            wealth = self.round(
                (self.starting_min_wealth + self.starting_max_wealth) / 2
            )
            self.people_wealth = {
                x: wealth
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
        self.people_wealth_start = self.people_wealth.copy()

    def set_results(self):
        """ Compute population results after all transactions are done """
        res = sorted([self.round(w) for w in self.people_wealth.values()])

        avg_wealth = sqrt(sum(w ** 2 for w in res) / self.num)

        if self.num % 2:
            pos = floor(self.num / 2)
            median_wealth = res[pos]
        else:
            pos = int(self.num / 2)
            median_wealth = (res[pos - 1] + res[pos]) / 2

        poorer_than_avg = len([w for w in res if w <= avg_wealth])
        poorer_than_avg_perc = self.round(poorer_than_avg / self.num * 100)

        poorer_than_begin = len([
            p for p, w in self.people_wealth.items()
            if w <= self.people_wealth_start[p]
        ])
        poorer_than_begin_perc = self.round(poorer_than_begin / self.num * 100)

        poorer_than_med = len([w for w in res if w <= median_wealth])
        poorer_than_med_perc = self.round(poorer_than_med / self.num * 100)

        min_res, max_res = min(res), max(res)
        delta_res = max_res - min_res

        def get_decile_key(k):
            inf = min_res + k * delta_res / 10
            if k < 9:
                sup = min_res + (k + 1) * delta_res / 10 - self.precision
            else:
                sup = max_res
            return inf, sup

        population_by_deciles = {
            get_decile_key(k): OrderedDict({
                'decile': "{}%-{}%".format(str(k * 10), str((k + 1) * 10)),
                'abs': 0,
                'perc': 0,
                'tot_w': 0
            })
            for k in range(10)
        }
        for w in res:
            for d in population_by_deciles.keys():
                if d[0] <= w <= d[1]:
                    population_by_deciles[d]['abs'] += 1
                    population_by_deciles[d]['perc'] += 100 / self.num
                    population_by_deciles[d]['tot_w'] += w

        one_percenter = sorted(res, reverse=True)[:floor(self.num / 100)]
        if not one_percenter:
            one_percenter = [res[-1]]
        one_percenter_wealth = sum(one_percenter)
        lower_percenter = []
        lower_percenter_wealth = 0
        for w in res:
            if w in one_percenter \
                    or lower_percenter_wealth > one_percenter_wealth:
                break
            lower_percenter_wealth += w
            lower_percenter.append(w)

        self.results.update(
            avg_wealth=avg_wealth,
            lower_percenter=lower_percenter,
            lower_percenter_wealth=lower_percenter_wealth,
            median_wealth=median_wealth,
            one_percenter=one_percenter,
            one_percenter_wealth=one_percenter_wealth,
            poorer_than_avg=poorer_than_avg,
            poorer_than_avg_perc=poorer_than_avg_perc,
            poorer_than_begin=poorer_than_begin,
            poorer_than_begin_perc=poorer_than_begin_perc,
            poorer_than_med=poorer_than_med,
            poorer_than_med_perc=poorer_than_med_perc,
            population_by_deciles=population_by_deciles,
        )


Population.__init__.__doc__ = Population.__doc__


if __name__ == '__main__':
    population = Population(**get_population_vals())
    population.set_population_wealth()
    population.make_transactions()
    population.set_results()
    population.print_results()
