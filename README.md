# chemistry_helpers

Small, dependency-free chemistry primitives: PDB-line parsing/formatting, bytes/str casting, and
a hardened OpenBabel subprocess wrapper. Author: Bertrand Caron. Library, **live-support**:
imported by `atb_helpers` (which wraps it with platform config) and by `atb_outputs`,
`Blind_RMSD`, `gamess_interface`, `pyscf_interface`, `xTB_interface`, `fragment_merger`,
`fragment_capping`, `cyana_lib_join`, `dihedral_scan_service`, `protomer_pipeline`,
`atb_graph_helpers` and `website`.

## Modules (`src/chemistry_helpers/`)
* `pdb.py`: PDB format v3.3 handling. `PDB_Atom`, `pdb_atoms_in`, `str_for_pdb_atom`,
  `pdb_str_from`, `get_coords_from_pdbstr`, `pdb_formula(_string)`, `pdb_total_charge`,
  `remove_pdb_charges`, `substitute_coordinates_in`.
* `babel.py`: `babel_output(data, in_format=..., out_format=..., timeout=...)` runs `babel` in a
  killable child (communicate timeout; orphaned children killed with `PR_SET_PDEATHSIG`); raises
  `BabelTimeoutError` / `BabelFailure` / `Babel_Screw_Up`; `dump_babel_failure` writes failing
  inputs under `logs/`. Defaults to `/usr/local/bin/babel`; `atb_helpers.babel` passes the
  configured path and timeout instead.
* `io.py`: `decode_if_necessary`, `encode_if_necessary`, `can_encode_and_decode`.

```python
from chemistry_helpers.babel import babel_output
inchi = babel_output('CCC', in_format='smiles', out_format='inchi')
```

## Configuration
None of its own: `babel_executable`, `babel_libdir` and `timeout` are arguments. Needs an
OpenBabel `babel` executable (platform: `/usr/local/bin/babel`).

## Packaging and tests
setuptools `pyproject.toml` + `setup.cfg` (`src` layout), no dependencies. **No tests.** The
`Makefile` is a py3.5-era pylint/mypy relic.

## Duplication
`pdb.py` helpers are copied (same names, same code) in `NXMol`'s
`chemistry_data_structure/parsing/pdb.py`; `atb_helpers` has its own `pdb.py`, `babel.py`, `io.py`
that layer on this package. See `atb_helpers/README.md`.
