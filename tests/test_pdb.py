"""Pins the current behaviour of chemistry_helpers.pdb with synthetic PDB lines."""
import pytest

from chemistry_helpers import pdb
from chemistry_helpers.pdb import PDB_Atom

# Columns: serial 7-11, name 13-16, coords 31-54, element 77-78, charge 79-80.
LINE = 'HETATM    1  C1  RES     1       0.000   1.500  -2.250  1.00  0.00           C  '
LINE_H = 'HETATM    2  H1  RES     1       1.000   0.000   0.000  1.00  0.00           H  '
LINE_CL_MINUS = 'ATOM      3 CL1  RES     1      -1.000   0.000   0.000  1.00  0.00          CL1-'
LINE_N_PLUS = 'HETATM    4  N1  RES     1       0.000   0.000   0.000  1.00  0.00           N2+'


def test_line_is_80_columns():
    assert len(LINE) == 80 and len(LINE_CL_MINUS) == 80


def test_atom_record_detection():
    assert pdb.is_pdb_atom_line(LINE)
    assert pdb.is_pdb_atom_line(LINE_CL_MINUS)
    assert not pdb.is_pdb_atom_line('REMARK   1')
    assert not pdb.is_pdb_atom_line('CONECT    1    2')
    assert not pdb.is_pdb_atom_line('')
    assert not pdb.is_pdb_atom_line('ATOM')  # needs the full 6 characters ('ATOM  ')
    assert pdb.is_pdb_connect_line('CONECT    1    2')
    assert not pdb.is_pdb_connect_line(LINE)


def test_pdb_atoms_in_fixed_columns():
    atoms = pdb.pdb_atoms_in('\n'.join(['REMARK x', LINE, LINE_CL_MINUS, LINE_N_PLUS, 'END']))
    assert atoms == [
        PDB_Atom(1, 'C1', (0.0, 1.5, -2.25), 'C', None),
        PDB_Atom(3, 'CL1', (-1.0, 0.0, 0.0), 'CL', -1),
        PDB_Atom(4, 'N1', (0.0, 0.0, 0.0), 'N', 2),
    ]


def test_charge_is_digit_then_sign_and_sign_first_raises():
    assert pdb.pdb_atoms_in(LINE[:78] + '1-')[0].charge == -1
    assert pdb.pdb_atoms_in(LINE[:78] + '3+')[0].charge == 3
    # REVIEW: '-1' (sign first, which many tools write) is not PDB v3.3 but is common; the
    # parser raises a bare Exception instead of returning None/-1.
    with pytest.raises(Exception, match='Unexpected charge_sign'):
        pdb.pdb_atoms_in(LINE[:78] + '-1')


def test_element_comes_from_columns_77_78_never_inferred_from_name():
    # REVIEW: with a blank element column the element is '' (no inference from the atom name),
    # so pdb_formula_string of such a file is the empty string.
    blank = LINE[:76] + '    '
    assert pdb.pdb_atoms_in(blank)[0].element == ''
    assert pdb.pdb_formula_string(blank) == ''


def test_alternate_locations_are_not_filtered():
    # REVIEW: column 17 (altLoc) is ignored, so both A and B conformers are returned as atoms.
    a = LINE[:16] + 'A' + LINE[17:]
    b = LINE[:16] + 'B' + LINE[17:]
    assert len(pdb.pdb_atoms_in(a + '\n' + b)) == 2


def test_bad_coordinates_raise_valueerror():
    with pytest.raises(ValueError):
        pdb.pdb_atoms_in(LINE[:30] + '     abc' + LINE[38:])
    with pytest.raises(ValueError):  # truncated record
        pdb.pdb_atoms_in(LINE[:40])


def test_coords_from_pdbstr_keeps_none_for_non_atom_lines():
    text = 'REMARK x\n' + LINE
    assert pdb.get_coords_from_pdbstr(text) == [None, (0.0, 1.5, -2.25)]
    assert pdb.get_coords_from_pdbstr(text, filter_empty=True) == [(0.0, 1.5, -2.25)]


def test_templates():
    assert pdb.PDB_TEMPLATE.startswith('{0:>6}{1:>5}{2:>5} {3:>4}{4:>1}{5:>4}')
    assert pdb.CONECT_TEMPLATE == '{0:>6}{1:>5}{2:>5}{3:>5}{4:>5}{5:>5}'


def test_pdb_str_from_roundtrip_and_layout():
    text = pdb.pdb_str_from(['C1', 'H1'], ['C', 'H'], [(0.0, 1.5, -2.25), (1.0, 0.0, 0.0)])
    lines = text.splitlines()
    assert lines[0].startswith('ATOM      1')
    # REVIEW: names are right-justified in the 5-wide field (cols 12-16), so 'C1' lands in cols
    # 15-16 instead of the PDB-standard cols 13-14; the reader strips, so it round-trips.
    assert lines[0][11:16] == '   C1'
    atoms = pdb.pdb_atoms_in(text)
    assert [(a.index, a.name, a.element, a.coordinates) for a in atoms] == [
        (1, 'C1', 'C', (0.0, 1.5, -2.25)), (2, 'H1', 'H', (1.0, 0.0, 0.0))]


def test_str_for_pdb_atom_charge_formatting():
    atom = PDB_Atom(1, 'C1', (0.0, 1.0, 2.0), 'C', 2)
    assert pdb.str_for_pdb_atom(atom).endswith('C2+')
    assert pdb.str_for_pdb_atom(atom._replace(charge=0)).endswith('C0+')
    assert pdb.str_for_pdb_atom(atom._replace(charge=None)).endswith('C  ')


def test_str_for_pdb_atom_negative_charge_is_malformed():
    # REVIEW (P2): str(-1) + '-' gives '-1-': three characters in a two-column field, so the
    # line is 81 columns and pdb_atoms_in cannot read it back. Fix would be abs(charge).
    s = pdb.str_for_pdb_atom(PDB_Atom(1, 'C1', (0.0, 1.0, 2.0), 'C', -1))
    assert s.endswith('C-1-') and len(s) == 81
    with pytest.raises(Exception, match='Unexpected charge_sign'):
        pdb.pdb_atoms_in(s)


def test_substitute_coordinates_in_keeps_other_fields():
    out = pdb.substitute_coordinates_in(LINE, (9.0, 8.0, 7.0))
    assert out[30:54] == '   9.000   8.000   7.000'
    assert out[:30] == LINE[:30] and out[54:] == LINE[54:]


def test_formula_and_charge():
    text = '\n'.join([LINE, LINE_H, LINE_H, LINE_CL_MINUS, LINE_N_PLUS])
    assert pdb.pdb_formula(text) == [('C', 1), ('CL', 1), ('H', 2), ('N', 1)]
    assert pdb.pdb_formula_string(text) == 'C' + 'Cl' + 'H2' + 'N'
    assert pdb.pdb_total_charge(text) == 1  # -1 (Cl) + 2 (N)
    assert pdb.pdb_formula_string(text, add_charge=True).endswith('+1')
    assert pdb.pdb_formula_string(LINE_H, add_charge=True) == 'H'


def test_pdb_charge_str_requires_80_columns():
    # REVIEW (P2): a line right-stripped to <80 columns makes pdb_total_charge raise IndexError
    # (line[79]); pdb_atoms_in / pdb_formula accept the same line.
    stripped = LINE.rstrip()
    assert len(stripped) == 78
    assert pdb.pdb_atoms_in(stripped)[0].element == 'C'
    with pytest.raises(IndexError):
        pdb.pdb_total_charge(stripped)


def test_pdb_charge_str_rejects_two_digit_charge():
    with pytest.raises(AssertionError, match='larger than 9'):
        pdb.pdb_total_charge(LINE[:78] + '10')


def test_remove_pdb_charges_truncates_atom_lines_only():
    text = 'REMARK x   ' + '\n' + LINE_CL_MINUS
    out = pdb.remove_pdb_charges(text)
    assert out.splitlines()[0] == 'REMARK x   '
    assert out.splitlines()[1] == LINE_CL_MINUS[:78]
    assert pdb.pdb_atoms_in(out)[0].charge is None


def test_replace_atom_by_hetatm():
    assert pdb.replace_ATOM_by_HETATM('ATOM      1').startswith('HETATM    ')
    assert pdb.replace_ATOM_by_HETATM(LINE) == LINE


def test_pdb_conect_line():
    assert pdb.pdb_conect_line([1, 2]) == 'CONECT    1    2               '
    assert pdb.pdb_conect_line([1, 2, 3, 4, 5]) == 'CONECT    1    2    3    4    5'
    # REVIEW (P3): a sixth serial is silently dropped; callers must split at 4 bonded atoms.
    assert pdb.pdb_conect_line([1, 2, 3, 4, 5, 6]) == 'CONECT    1    2    3    4    5'
    # REVIEW: annotated '-> List[str]' but returns str.
    assert isinstance(pdb.pdb_conect_line([1]), str)


def test_pdb_atom_lines_number():
    assert pdb.pdb_atom_lines_number('\n'.join([LINE, 'REMARK', LINE_H, 'CONECT    1    2'])) == 2
