"""The Lewis-structure (bond order and formal charge) constraint model, built once.

Two solvers in the platform assign bond orders and formal charges to a connectivity graph by
integer programming: ``atb_helpers.bond_orders.assign_bond_orders`` (scipy ``milp``/HiGHS, the
standard-InChI and SMILES path) and ``fragment_capping``'s three pulp models
(``helpers/ilp_constraints.py``: bond orders, capping, tautomers).  They state the same chemistry
-- every bond an integer order, every atom's orders consistent with its valence and formal
charge, the charges summing to the net charge, no cumulated double bonds where they are not
wanted -- and until D6 (duplication plan, owner decision 2026-10-20) each wrote its own rows.

This module holds the rows.  It is solver-agnostic and pure (stdlib; numpy only inside
:meth:`LinearModel.milp_arrays`): a :class:`LinearModel` is named variables with bounds and
integrality, linear :class:`Row` s and an objective, from which a scipy ``milp`` matrix
(:meth:`LinearModel.milp_arrays`) or a pulp problem (iterate ``variables`` and ``rows``; the
adapter is fragment_capping's) is generated.  Every row builder here returns :class:`Row` s
and never adds them: *which* atoms and bonds a row covers, the order rows are added in, the
constraint names, the objective and the tie-breaking are each consumer's, as D6 requires.

**Two encodings of the same structure.**  The consumers encode a bond order differently, and
the module serves both through :class:`Expr` (a linear expression with a constant), so every
row is written once in terms of a bond's *order* and *extra order* (order - 1) and an atom's
*charge*, whatever variables those expand to:

- *electron bookkeeping* (fragment_capping): per bond one integer ``B`` in [1, 3] (the order
  itself); per atom an integer charge ``C`` in [-9, 9], its magnitude ``Z`` >= |C| (minimised
  first), and ``N`` non-bonded electron pairs (or electrons, with radicals).  The chemistry is
  the electron balance (:func:`electron_balance_row`: ``C = V - sum(orders) - m N``, ``V`` the
  valence electrons) and the octet (:func:`octet_row`: ``2 sum(orders) + m N = 8``, 2 for H/He).
  Any charge an element can carry is allowed; the objective picks the least charged.
- *state selection* (atb_helpers): per heavy-heavy bond two binaries, its second and its third
  order (``extra = x2 + x3``, :func:`third_only_on_second_row`); per heavy atom one binary per
  allowed ``(valence, charge)`` state, exactly one chosen (:func:`one_state_row`), the atom's
  bond orders summing to the chosen valence (:func:`state_valence_row`).  Hydrogens and their
  bonds are fixed (order 1) and appear only through the atom's degree.  Octet, hypervalence and
  which charges an element may carry are all in the state table, which stays with the consumer
  together with its penalties.

Rows that read the same in both encodings: :func:`net_charge_row`, :func:`cumulene_row`
(at most one extra order among an atom's bonds: no =C= or =[N+]= at that atom; fragment_capping's
"no allenes in short rings", atb_helpers' rule (A) for 3-7 rings), the switch form
:func:`allene_switch_rows` (forbid exactly the (2, 2) pair, with a binary) and
:func:`fixed_order_row`.  Encoding-specific: :func:`electron_balance_row`, :func:`octet_row`,
:func:`absolute_charge_rows` (bookkeeping); :func:`one_state_row`, :func:`state_valence_row`,
:func:`third_only_on_second_row`, :func:`no_multiple_bond_row` (states).  NXMol's CP solver
(cpmpy, live via atb_microstates tier 0) is the bookkeeping encoding with the octet as a hard
domain and the same electron balance; it is not migrated here (D6 scope).

Rows carry an optional ``name``; a range (both bounds finite and different) is one row for
scipy and two for pulp -- builders that must be two pulp constraints return two rows.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INF = float('inf')

#: Bounds and multipliers the electron-bookkeeping encoding has always used (fragment_capping
#: ``helpers/parameters.py``; the consumers pass their own, these are the documented defaults).
MIN_BOND_ORDER, MAX_BOND_ORDER = 1, 3
MAX_ABSOLUTE_CHARGE = 9
MAX_NONBONDED_ELECTRONS = 18
ELECTRONS_PER_BOND = 2


class Expr:
    """A linear expression: ``{variable name: coefficient}`` plus a constant.  Immutable in use;
    ``+``, ``-`` and scalar ``*`` return new expressions.  Coefficients stay the Python numbers
    they were given (ints stay ints), so a matrix or an LP text built from them is what a
    hand-written one would be."""
    __slots__ = ('coefficients', 'constant')

    def __init__(self, coefficients: Optional[Dict[str, float]] = None, constant: float = 0):
        self.coefficients = dict(coefficients or {})
        self.constant = constant

    @classmethod
    def var(cls, name: str, coefficient: float = 1) -> 'Expr':
        return cls({name: coefficient})

    @classmethod
    def const(cls, value: float) -> 'Expr':
        return cls({}, value)

    @classmethod
    def sum(cls, terms: Iterable['Expr']) -> 'Expr':
        total = cls()
        for term in terms:
            total = total + term
        return total

    def __add__(self, other: 'Expr') -> 'Expr':
        if not isinstance(other, Expr):
            other = Expr.const(other)
        coefficients = dict(self.coefficients)
        for name, value in other.coefficients.items():
            coefficients[name] = coefficients.get(name, 0) + value
        return Expr(coefficients, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self) -> 'Expr':
        return Expr({name: -value for name, value in self.coefficients.items()}, -self.constant)

    def __sub__(self, other: 'Expr') -> 'Expr':
        return self + (-other if isinstance(other, Expr) else Expr.const(-other))

    def __rsub__(self, other) -> 'Expr':
        return Expr.const(other) - self

    def __mul__(self, factor: float) -> 'Expr':
        return Expr({name: factor * value for name, value in self.coefficients.items()}, factor * self.constant)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return 'Expr(%r, %r)' % (self.coefficients, self.constant)


class Variable:
    __slots__ = ('name', 'lower', 'upper', 'integer')

    def __init__(self, name: str, lower: float, upper: float, integer: bool = True):
        self.name, self.lower, self.upper, self.integer = name, lower, upper, integer

    def expr(self, coefficient: float = 1) -> Expr:
        return Expr.var(self.name, coefficient)

    def __repr__(self) -> str:
        return 'Variable(%r, %r, %r, integer=%r)' % (self.name, self.lower, self.upper, self.integer)


class Row:
    """``lower <= sum(coefficient * variable) <= upper``.  The expression's constant is already
    folded into the bounds (:func:`row`), so ``coefficients`` is what a matrix row or an LP line
    holds.  ``INF``/``-INF`` for a one-sided row; ``lower == upper`` for an equality."""
    __slots__ = ('coefficients', 'lower', 'upper', 'name')

    def __init__(self, coefficients: Dict[str, float], lower: float, upper: float, name: Optional[str] = None):
        self.coefficients, self.lower, self.upper, self.name = dict(coefficients), lower, upper, name

    @property
    def is_equality(self) -> bool:
        return self.lower == self.upper

    def __repr__(self) -> str:
        return 'Row(%r, %r, %r, name=%r)' % (self.coefficients, self.lower, self.upper, self.name)


def row(expr: Expr, lower: float = -INF, upper: float = INF, name: Optional[str] = None) -> Row:
    """``lower <= expr <= upper`` as a :class:`Row`, the constant moved to the bounds."""
    return Row(expr.coefficients, lower - expr.constant if lower != -INF else -INF,
               upper - expr.constant if upper != INF else INF, name)


def equality(expr: Expr, value: float, name: Optional[str] = None) -> Row:
    return row(expr, value, value, name)


class LinearModel:
    """Variables in insertion order (the matrix column order), rows in insertion order (the row
    order) and a linear objective.  Pure; :meth:`milp_arrays` is the only place numpy is used."""

    def __init__(self):
        self.variables: List[Variable] = []
        self.column: Dict[str, int] = {}
        self.rows: List[Row] = []
        self.objective: Dict[str, float] = {}

    def add_variable(self, name: str, lower: float, upper: float, integer: bool = True) -> Variable:
        if name in self.column:
            raise ValueError('duplicate variable %r' % name)
        variable = Variable(name, lower, upper, integer)
        self.column[name] = len(self.variables)
        self.variables.append(variable)
        return variable

    def add_row(self, row_: Row) -> Row:
        for name in row_.coefficients:
            if name not in self.column:
                raise ValueError('row %r uses unknown variable %r' % (row_.name, name))
        self.rows.append(row_)
        return row_

    def add_rows(self, rows: Iterable[Row]) -> None:
        for row_ in rows:
            self.add_row(row_)

    def set_cost(self, name: str, cost: float) -> None:
        if name not in self.column:
            raise ValueError('unknown variable %r' % name)
        self.objective[name] = cost

    def milp_arrays(self):
        """The model as ``scipy.optimize.milp`` takes it: ``(cost, (rows, cols, values),
        row_lower, row_upper, var_lower, var_upper, integrality)``; the COO triplets are in row
        order, the coefficient values as given (an all-integer model stays integer)."""
        import numpy as np
        n = len(self.variables)
        cost = np.zeros(n)
        for name, value in self.objective.items():
            cost[self.column[name]] = value
        rows, cols, values = [], [], []
        row_lower, row_upper = np.zeros(len(self.rows)), np.zeros(len(self.rows))
        for r, row_ in enumerate(self.rows):
            for name, value in row_.coefficients.items():
                rows.append(r)
                cols.append(self.column[name])
                values.append(value)
            row_lower[r], row_upper[r] = row_.lower, row_.upper
        var_lower = np.array([v.lower for v in self.variables], dtype=float)
        var_upper = np.array([v.upper for v in self.variables], dtype=float)
        integrality = np.array([1 if v.integer else 0 for v in self.variables])
        return cost, (rows, cols, values), row_lower, row_upper, var_lower, var_upper, integrality


# --- variables: the two encodings ------------------------------------------------------------

def bookkeeping_bond_variable(model: LinearModel, name: str, upper: int = MAX_BOND_ORDER,
                              lower: int = MIN_BOND_ORDER) -> Expr:
    """An integer bond order ``B`` (the *order* expression)."""
    return model.add_variable(name, lower, upper).expr()


def bookkeeping_atom_variables(model: LinearModel, charge_name: str, absolute_name: str, nonbonded_name: str,
                               electron_multiplier: int, max_absolute_charge: int = MAX_ABSOLUTE_CHARGE,
                               max_nonbonded_electrons: int = MAX_NONBONDED_ELECTRONS) -> Tuple[Expr, Expr, Expr]:
    """``C`` in [-max, max], ``Z`` in [0, max] and ``N`` in [0, max_electrons // multiplier]."""
    charge = model.add_variable(charge_name, -max_absolute_charge, max_absolute_charge).expr()
    absolute = model.add_variable(absolute_name, 0, max_absolute_charge).expr()
    nonbonded = model.add_variable(nonbonded_name, 0, max_nonbonded_electrons // electron_multiplier).expr()
    return charge, absolute, nonbonded


def state_bond_variables(model: LinearModel, second_name: str, third_name: str) -> Expr:
    """Two binaries, a bond's second and its third order; returns the *extra order* ``x2 + x3``
    (the order is that plus one)."""
    second = model.add_variable(second_name, 0, 1)
    third = model.add_variable(third_name, 0, 1)
    return second.expr() + third.expr()


def state_atom_variables(model: LinearModel, names: Sequence[str]) -> List[Expr]:
    """One binary per allowed state of an atom."""
    return [model.add_variable(name, 0, 1).expr() for name in names]


# --- rows shared by both encodings -------------------------------------------------------------

def net_charge_row(charges: Iterable[Expr], net_charge: int, name: Optional[str] = None) -> Row:
    """The formal charges sum to the net charge."""
    return equality(Expr.sum(charges), net_charge, name)


def cumulene_row(extra_orders: Iterable[Expr], name: Optional[str] = None) -> Row:
    """At most one extra order among these bonds of one atom: no =X= there (and no triple bond
    beside a double one).  In order terms, for two bonds, ``B1 + B2 <= 3``."""
    return row(Expr.sum(extra_orders), upper=1, name=name)


def allene_switch_rows(order_1: Expr, order_2: Expr, switch: Expr) -> Tuple[Row, Row]:
    """Forbid exactly the (2, 2) pair of orders at an atom, every other pair allowed (a triple
    beside a single is fine), with a binary switch: ``3 <= 2 B1 - B2 + 4 A <= 5``.  Two rows, as
    the pulp models have always written it."""
    expr = 2 * order_1 - order_2 + 4 * switch
    return row(expr, lower=3), row(expr, upper=5)


def fixed_order_row(order: Expr, value: int, name: Optional[str] = None) -> Row:
    return equality(order, value, name)


# --- electron bookkeeping --------------------------------------------------------------------

def electron_balance_row(charge: Expr, valence_electrons: int, orders: Iterable[Expr], nonbonded: Expr,
                         electron_multiplier: int, name: Optional[str] = None) -> Row:
    """``charge = valence_electrons - sum(orders) - multiplier * nonbonded``."""
    return equality(charge - (valence_electrons - Expr.sum(orders) - electron_multiplier * nonbonded), 0, name)


def octet_row(orders: Iterable[Expr], nonbonded: Expr, electron_multiplier: int, electrons: int = 8,
              electrons_per_bond: int = ELECTRONS_PER_BOND, name: Optional[str] = None) -> Row:
    """``electrons_per_bond * sum(orders) + multiplier * nonbonded = electrons`` (8; 2 for H, He)."""
    return equality(electrons_per_bond * Expr.sum(orders) + electron_multiplier * nonbonded, electrons, name)


def absolute_charge_rows(charge: Expr, absolute: Expr, names: Tuple[Optional[str], Optional[str]] = (None, None)
                         ) -> Tuple[Row, Row]:
    """``charge <= Z`` and ``-charge <= Z``: ``Z >= |charge|``."""
    return row(charge - absolute, upper=0, name=names[0]), row(-charge - absolute, upper=0, name=names[1])


# --- state selection ----------------------------------------------------------------------------

def one_state_row(states: Iterable[Expr], name: Optional[str] = None) -> Row:
    """Exactly one state chosen."""
    return equality(Expr.sum(states), 1, name)


def state_valence_row(extra_orders: Iterable[Expr], states: Iterable[Tuple[Expr, int]], degree: int,
                      name: Optional[str] = None) -> Row:
    """The atom's extra orders sum to the chosen state's valence minus its degree (every bond,
    hydrogens included, contributes one order already): ``sum(extra) - sum((v_s - degree) x_s) = 0``,
    written in that form (with :func:`one_state_row`) rather than ``sum(extra) - sum(v_s x_s) = -degree``."""
    return equality(Expr.sum(extra_orders) - Expr.sum(x * (valence - degree) for x, valence in states), 0, name)


def third_only_on_second_row(second: Expr, third: Expr, name: Optional[str] = None) -> Row:
    """``x3 <= x2``: a third order only on top of a second."""
    return row(third - second, upper=0, name=name)


def no_multiple_bond_row(extra_orders: Sequence[Expr], unless_states: Iterable[Expr], name: Optional[str] = None
                         ) -> Row:
    """These bonds stay single unless none of ``unless_states`` is chosen (big-M with
    M = 2 * number of bonds, the most extra order they can hold): atb_helpers' rule that an
    expanded-octet S/Se/Te bonds multiply only to heteroatoms."""
    big_m = 2 * len(extra_orders)
    return row(Expr.sum(extra_orders) + big_m * Expr.sum(unless_states), upper=big_m, name=name)
