import sys

from collections import Counter, OrderedDict
from math import floor, sqrt
from random import randint, uniform

POPULATION_ATTRS = {
    'allow_duplicates': (bool,),
    'days': (int,),
    'equal_start': (bool,),
    'loss': (float, int),
    'meritocracy': (bool,),
    'num': (int,),
    'precision': (float, int),
    'random_loss': (bool,),
    'random_win': (bool,),
    'starting_max_wealth': (float, int),
    'starting_min_wealth': (float, int),
    'win': (float, int),
}


def get_population_vals():
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
        self.meritocracy = kwargs.get('meritocracy', True)
        self.num = kwargs.get('num', 1000)
        self.precision = kwargs.get('precision', 2)
        self.random_loss = kwargs.get('random_loss', True)
        self.random_win = kwargs.get('random_win', True)
        self.starting_max_wealth = kwargs.get('starting_max_wealth', 100)
        self.starting_min_wealth = kwargs.get('starting_min_wealth', 100)
        self.win = kwargs.get('win', 100)

        self.people_wealth = {}
        self.people_wealth_start = {}
        self.results = {}

        self.equal_start = kwargs.get('equal_start', False) or (
            self.compare_vals(
                self.starting_max_wealth, self.starting_min_wealth
            ) == 0
        )

        self.name = "Population data:\n    {}".format(
            "\n    ".join([
                "{}: {}".format(attr_name, attr)
                for attr_name, attr in self.read().items()
            ])
        )
        self.check_vals()

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)

    def check_vals(self):
        if self.days < 1:
            raise ValueError("Days can't be lower than 1.")
        if not (0 <= self.loss <= 100) or self.win < 0:
            raise ValueError(
                "Loss percentage can't be lower than 0 or higher than 100."
                " Win percentage can't be lower than 0."
            )
        if self.num < 2:
            raise ValueError("Population can't be lower than 2.")
        if self.precision < 0:
            raise ValueError(
                "Precision can't be lower than 0. Set precision to 0 if you"
                " want only integers, else set the number of decimal places"
                " you want to see."
            )
        if not 0 <= self.starting_min_wealth <= self.starting_max_wealth:
            raise ValueError(
                "Starting max and min wealth must both be >= 0, and max wealth"
                " can't be lower than min wealth."
            )

    def compare_vals(self, value1, value2):
        if self.is_zero(value1 - value2):
            return 0
        elif value1 > value2:
            return 1
        else:
            return -1

    def get_people_wealth(self, people):
        wealth = ()
        for person in people:
            wealth += (self.people_wealth[person],)
        return wealth

    @staticmethod
    def get_population_attrs():
        return POPULATION_ATTRS

    @staticmethod
    def get_random_perc(perc):
        floor_digits = len(str(perc - floor(perc)).split('.')[-1])
        exp_perc = perc * (10 ** floor_digits)
        return randint(1, int(exp_perc)) / (10 ** floor_digits)

    def get_transaction_value(self, person1, person2, winner):
        wealth1, wealth2 = self.get_people_wealth((person1, person2))
        win, loss = self.get_win_loss_perc()

        if self.is_zero(wealth1 - wealth2):
            return self.round((wealth1 + wealth2) * (win + loss) / 400)

        base = min(wealth1, wealth2)
        poor = {wealth1: person1, wealth2: person2}[base]
        if winner == poor:
            multiplier = win
        else:
            multiplier = loss

        return self.round(base * multiplier / 100)

    def get_win_loss_perc(self):
        win = self.win
        if self.random_win:
            win = Population.get_random_perc(win)
        loss = self.loss
        if self.random_loss:
            loss = Population.get_random_perc(loss)
        return win, loss

    def get_winner(self, person1, person2):
        if not self.meritocracy:
            return {0: person1, 1: person2}[randint(0, 1)]

        wealth1, wealth2 = self.get_people_wealth((person1, person2))
        if self.is_zero(wealth1 + wealth2):
            return person1

        prob = {
            person1: (0, wealth1 / (wealth1 + wealth2)),
            person2: (wealth1 / (wealth1 + wealth2), 1)
        }

        res = uniform(0, 1)

        def is_in(r, p):
            return min(prob[p]) <= r <= max(prob[p])

        while (is_in(res, person1) and is_in(res, person2)) \
                or not (is_in(res, person1) or is_in(res, person2)):
            res = uniform(0, 1)
        if is_in(res, person1):
            return person1
        else:
            return person2

    def is_zero(self, value):
        return abs(self.round(value)) < 10 ** -self.precision

    def make_transactions(self):
        for day in range(1, self.days + 1):
            print("Day {}".format(day))
            done = []
            for person1, wealth1 in self.people_wealth.items():
                if not self.allow_duplicates and (
                        person1 in done
                        or (self.num % 2 and self.num - len(done) == 1)
                ):
                    continue

                person2 = 0
                while not person2 or person2 == person1 \
                        or (not self.allow_duplicates and person2 in done):
                    person2 = randint(1, self.num)
                wealth2 = self.people_wealth[person2]

                winner = self.get_winner(person1, person2)
                value = self.get_transaction_value(person1, person2, winner)

                sign1 = 1 if person1 == winner else -1
                sign2 = 1 if person2 == winner else -1
                self.people_wealth.update({
                    person1: wealth1 + (sign1 * value),
                    person2: wealth2 + (sign2 * value),
                })

                done += [person1, person2]
                print(
                    "Day {} - transactions {}, {} completed"
                    .format(day, len(done) - 1, len(done))
                )

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
        if value == 0:
            return value
        return round(value * (10 ** self.precision)) / (10 ** self.precision)

    def set_population_wealth(self):
        if self.equal_start:
            wealth = self.round(
                (self.starting_min_wealth + self.starting_max_wealth) / 2
            )
            self.people_wealth = {
                x: wealth
                for x in range(1, self.num + 1)
            }

        else:
            if self.compare_vals(
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
        self.people_wealth_start = self.people_wealth.copy()

    def set_results(self):
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
            p
            for p, w in self.people_wealth.items()
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


if __name__ == '__main__':
    population = Population(**get_population_vals())
    population.set_population_wealth()
    population.make_transactions()
    population.set_results()
    population.print_results()
