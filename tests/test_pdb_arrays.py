"""read_pdb_arrays and the D5 missing-element rule (owner decision 2026-10-20)."""
import pytest

from chemistry_helpers.pdb import PdbElementError, infer_element, read_pdb_arrays


def line(name, resname='LIG', element='', serial=1, rec='HETATM', charge=''):
    return '{0:<6}{1:>5} {2:<4} {3:<3}  {4:>4}    {5:8.3f}{6:8.3f}{7:8.3f}  1.00  0.00          {8:>2}{9:<2}'.format(
        rec, serial, name, resname, 1, 1.0, 2.0, 3.0, element, charge)


def test_element_column_wins_and_is_normalised():
    names, elements, coords, extras = read_pdb_arrays(line(' CL1', element='CL', charge='1-'))
    assert (names, elements, coords) == (['CL1'], ['Cl'], [(1.0, 2.0, 3.0)])
    assert extras['charges'] == [-1] and extras['inferred'] == [False] and extras['resnames'] == ['LIG']


@pytest.mark.parametrize('name,res,expected', [
    (' C1 ', 'LIG', 'C'), (' H12', 'LIG', 'H'), ('1HB ', 'ALA', 'H'), (" O5'", 'LIG', 'O'),
    (' CA ', 'ALA', 'C'), (' NE ', 'ARG', 'N'), (' HG ', 'SER', 'H'), (' OG ', 'SER', 'O'),
    (' SD ', 'MET', 'S'), (' SE ', 'MSE', 'Se'), (' OW ', 'HOH', 'O'),
    (' CL1', 'LIG', None), (' BR ', 'LIG', None), (' CA ', 'CA', None), (' NA ', 'NA', None),
    (' ZN ', 'ZN', None), (' MG ', 'MG', None), (' HG ', 'HG', None), (' K  ', 'K', None),
    (' FE ', 'HEM', None), (' CD1', 'LIG', None), (' X1 ', 'LIG', None), ('    ', 'LIG', None),
])
def test_infer_element(name, res, expected):
    assert infer_element(name, res) == expected


def test_missing_element_inferred_and_flagged():
    text = '\n'.join([line(' C1 ', serial=1), line(' H1 ', serial=2)])
    _, elements, _, extras = read_pdb_arrays(text)
    assert elements == ['C', 'H'] and extras['inferred'] == [True, True]


def test_missing_element_ambiguous_raises_with_line_number():
    text = '\n'.join(['REMARK', line(' C1 ', serial=1), line('CL1 ', serial=2)])
    with pytest.raises(PdbElementError, match='line 3'):
        read_pdb_arrays(text)


def test_conect_pairs_deduplicated_and_ordered():
    text = '\n'.join([line(' C1 ', element='C', serial=1), line(' C2 ', element='C', serial=2),
                     'CONECT    1    2', 'CONECT    2    1    3'])
    assert read_pdb_arrays(text)[3]['conect'] == [(1, 2), (2, 3)]


def test_require_elements_false_gives_blank_and_icode():
    text = line('CL1 ', serial=1)
    names, elements, _, extras = read_pdb_arrays(text, require_elements=False)
    assert names == ['CL1'] and elements == [''] and extras['icodes'] == ['']
