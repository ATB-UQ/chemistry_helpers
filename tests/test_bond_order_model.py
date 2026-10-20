"""chemistry_helpers.bond_order_model: the rows of both encodings and the milp arrays."""
import numpy as np
import pytest

from chemistry_helpers.bond_order_model import (
    INF, Expr, LinearModel, absolute_charge_rows, allene_switch_rows, bookkeeping_atom_variables,
    bookkeeping_bond_variable, cumulene_row, electron_balance_row, equality, fixed_order_row, net_charge_row,
    no_multiple_bond_row, octet_row, one_state_row, row, state_atom_variables, state_bond_variables,
    state_valence_row, third_only_on_second_row)


def test_expr_arithmetic_keeps_ints_and_zero_coefficients():
    e = 2 * Expr.var('a') - Expr.var('b') + 3
    assert e.coefficients == {'a': 2, 'b': -1} and e.constant == 3
    assert all(isinstance(v, int) for v in e.coefficients.values())
    z = Expr.var('a') * 0
    assert z.coefficients == {'a': 0}  # an explicit zero stays (the matrix keeps the entry)
    assert Expr.sum([Expr.var('a'), Expr.var('a', 2)]).coefficients == {'a': 3}


def test_row_folds_the_constant_into_the_bounds():
    r = row(Expr.var('B1') + Expr.var('B2') - 2, upper=1)
    assert r.coefficients == {'B1': 1, 'B2': 1} and r.lower == -INF and r.upper == 3
    assert equality(Expr.var('C') - 5, 0).lower == 5


def test_bookkeeping_rows_read_as_the_fragment_capping_lp():
    m = LinearModel()
    b1 = bookkeeping_bond_variable(m, 'B_0')
    b2 = bookkeeping_bond_variable(m, 'B_1', upper=2)
    c, z, n = bookkeeping_atom_variables(m, 'C_1', 'Z_1', 'N_1', electron_multiplier=2)
    assert [v.upper for v in m.variables] == [3, 2, 9, 9, 9]
    assert m.variables[2].lower == -9 and m.variables[3].lower == 0
    balance = electron_balance_row(c, 6, [b1, b2], n, 2)  # oxygen: C = 6 - B - 2N
    assert balance.coefficients == {'C_1': 1, 'B_0': 1, 'B_1': 1, 'N_1': 2} and balance.lower == balance.upper == 6
    octet = octet_row([b1, b2], n, 2)
    assert octet.coefficients == {'B_0': 2, 'B_1': 2, 'N_1': 2} and octet.lower == 8
    assert octet_row([b1], n, 1, electrons=2).upper == 2
    left, right = absolute_charge_rows(c, z, names=('l', 'r'))
    assert left.coefficients == {'C_1': 1, 'Z_1': -1} and left.upper == 0 and left.name == 'l'
    assert right.coefficients == {'C_1': -1, 'Z_1': -1} and right.upper == 0
    allene = cumulene_row(b - 1 for b in (b1, b2))  # B1 + B2 <= 3
    assert allene.coefficients == {'B_0': 1, 'B_1': 1} and allene.upper == 3
    a = m.add_variable('A_1', 0, 1).expr()
    lo, hi = allene_switch_rows(b1, b2, a)
    assert lo.coefficients == hi.coefficients == {'B_0': 2, 'B_1': -1, 'A_1': 4}
    assert (lo.lower, lo.upper, hi.lower, hi.upper) == (3, INF, -INF, 5)
    assert fixed_order_row(b1, 2).lower == 2
    assert net_charge_row([c], -1).coefficients == {'C_1': 1}


def test_state_rows_read_as_the_atb_helpers_matrix():
    m = LinearModel()
    extra = state_bond_variables(m, 'x2_0', 'x3_0')
    states = state_atom_variables(m, ['s_0_0', 's_0_1'])  # (4, 0) and (3, -1) on a carbon of degree 3
    valence = state_valence_row([extra], [(states[0], 4), (states[1], 3)], degree=3)
    assert valence.coefficients == {'x2_0': 1, 'x3_0': 1, 's_0_0': -1, 's_0_1': 0} and valence.lower == 0
    assert one_state_row(states).coefficients == {'s_0_0': 1, 's_0_1': 1}
    chain = third_only_on_second_row(m.variables[0].expr(), m.variables[1].expr())
    assert chain.coefficients == {'x2_0': -1, 'x3_0': 1} and chain.upper == 0
    guard = no_multiple_bond_row([extra], [states[1]])
    assert guard.coefficients == {'x2_0': 1, 'x3_0': 1, 's_0_1': 2} and guard.upper == 2
    assert cumulene_row([extra]).upper == 1


def test_milp_arrays_follow_insertion_order_and_keep_zeros():
    m = LinearModel()
    x = m.add_variable('x', 0, 1).expr()
    y = m.add_variable('y', 0, 3, integer=False).expr()
    m.add_row(equality(x + 0 * y, 1, 'eq'))
    m.add_row(row(2 * x - y, upper=4))
    m.set_cost('y', 0.5)
    cost, (rows, cols, values), lower, upper, vl, vu, integrality = m.milp_arrays()
    assert cost.tolist() == [0, 0.5]
    assert (rows, cols, values) == ([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 2, -1])
    assert lower.tolist() == [1, -np.inf] and upper.tolist() == [1, 4]
    assert vl.tolist() == [0, 0] and vu.tolist() == [1, 3] and integrality.tolist() == [1, 0]


def test_unknown_or_duplicate_names_are_refused():
    m = LinearModel()
    m.add_variable('x', 0, 1)
    with pytest.raises(ValueError):
        m.add_variable('x', 0, 1)
    with pytest.raises(ValueError):
        m.add_row(row(Expr.var('nope'), upper=1))
    with pytest.raises(ValueError):
        m.set_cost('nope', 1)


def test_the_two_encodings_agree_on_a_solved_structure():
    """Both encodings, solved with scipy, give acetate's O=C-[O-]: the model is one chemistry."""
    scipy = pytest.importorskip('scipy.optimize')
    from scipy.sparse import coo_matrix
    # Bookkeeping: atoms C1, O2, O3 (+ CH3 carbon as a fixed single bond partner, dropped: use H-less acetate core
    # C(=O)[O-] with the methyl as a single bond to a 'C0' whose own rows we omit by fixing its order).
    m = LinearModel()
    orders = {('C0', 'C1'): bookkeeping_bond_variable(m, 'B_0'), ('C1', 'O2'): bookkeeping_bond_variable(m, 'B_1'),
              ('C1', 'O3'): bookkeeping_bond_variable(m, 'B_2')}
    atoms = {}
    for name, ve in (('C1', 4), ('O2', 6), ('O3', 6)):
        c, z, n = bookkeeping_atom_variables(m, 'C_' + name, 'Z_' + name, 'N_' + name, 2)
        atoms[name] = (c, z, n)
        mine = [o for pair, o in orders.items() if name in pair]
        m.add_row(electron_balance_row(c, ve, mine, n, 2))
        m.add_row(octet_row(mine, n, 2))
        m.add_rows(absolute_charge_rows(c, z))
        m.set_cost('Z_' + name, 1)
    m.add_row(fixed_order_row(orders[('C0', 'C1')], 1))
    m.add_row(net_charge_row([atoms[a][0] for a in atoms], -1))
    cost, (r, c_, v), lo, hi, vl, vu, integ = m.milp_arrays()
    res = scipy.milp(cost, constraints=scipy.LinearConstraint(coo_matrix((v, (r, c_)), shape=(len(m.rows), len(m.variables))), lo, hi),
                     integrality=integ, bounds=scipy.Bounds(vl, vu))
    x = dict(zip(m.column, np.round(res.x).astype(int)))
    assert sorted([x['B_1'], x['B_2']]) == [1, 2] and x['C_C1'] == 0 and sorted([x['C_O2'], x['C_O3']]) == [-1, 0]
    # State selection: same molecule, the O states (2, 0) / (1, -1) / (3, +1), C (4, 0) / (3, -1) / (3, +1).
    m = LinearModel()
    extra = {pair: state_bond_variables(m, 'x2_%d' % k, 'x3_%d' % k) for k, pair in enumerate(orders)}
    table = {'C1': ((4, 0, 0), (3, -1, 4), (3, 1, 4)), 'O2': ((2, 0, 0), (1, -1, 1), (3, 1, 2)), 'O3': ((2, 0, 0), (1, -1, 1), (3, 1, 2))}
    degree = {'C1': 3, 'O2': 1, 'O3': 1}
    charges = []
    for a, (name, states) in enumerate(table.items()):
        xs = state_atom_variables(m, ['s_%d_%d' % (a, k) for k in range(len(states))])
        mine = [e for pair, e in extra.items() if name in pair]
        m.add_row(state_valence_row(mine, [(xv, s[0]) for xv, s in zip(xs, states)], degree[name]))
        m.add_row(one_state_row(xs))
        for xv, (valence, charge, penalty) in zip(xs, states):
            m.set_cost(next(iter(xv.coefficients)), penalty)
            charges.append(xv * charge)
    m.add_row(net_charge_row(charges, -1))
    m.add_row(fixed_order_row(extra[('C0', 'C1')], 0))
    for k in range(3):
        m.add_row(third_only_on_second_row(m.variables[2 * k].expr(), m.variables[2 * k + 1].expr()))
    cost, (r, c_, v), lo, hi, vl, vu, integ = m.milp_arrays()
    res = scipy.milp(cost, constraints=scipy.LinearConstraint(coo_matrix((v, (r, c_)), shape=(len(m.rows), len(m.variables))), lo, hi),
                     integrality=integ, bounds=scipy.Bounds(vl, vu))
    x = dict(zip(m.column, np.round(res.x).astype(int)))
    assert sorted([x['x2_1'] + x['x3_1'], x['x2_2'] + x['x3_2']]) == [0, 1] and x['s_0_0'] == 1
    assert sorted([x['s_1_1'], x['s_2_1']]) == [0, 1]  # exactly one O- state
