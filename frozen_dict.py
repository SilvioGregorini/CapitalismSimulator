# Author(s): Silvio Gregorini (silviogregorini@openforce.it)
# Copyright 2019 Silvio Gregorini (github.com/SilvioGregorini)
# License LGPL-3.0 or later (https://www.gnu.org/licenses/lgpl).


class FrozenDict(dict):
    """
    An implementation of an immutable dictionary.
    Thanks to the guys at Odoo S.A. for this class.
    """

    def __delitem__(self, key):
        raise NotImplementedError("'__delitem__' not supported on FrozenDict")

    def __setitem__(self, key, val):
        raise NotImplementedError("'__setitem__' not supported on FrozenDict")

    def clear(self):
        raise NotImplementedError("'clear' not supported on FrozenDict")

    def pop(self, key, default=None):
        raise NotImplementedError("'pop' not supported on FrozenDict")

    def popitem(self):
        raise NotImplementedError("'popitem' not supported on FrozenDict")

    def setdefault(self, key, default=None):
        raise NotImplementedError("'setdefault' not supported on FrozenDict")

    def update(self, *args, **kwargs):
        raise NotImplementedError("'update' not supported on FrozenDict")
