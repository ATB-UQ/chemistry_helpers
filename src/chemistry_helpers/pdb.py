from typing import List, Optional, Tuple, Union, NamedTuple, Any
from itertools import groupby
from operator import itemgetter

Coordinate = Tuple[float, float, float]

S = ('>', '')
CHARGE = ('<', '')
F = ('', '.3f')
I = ('>', '')

# Columns definition are inclusive on both ends
PDB_ATOM_INDEX_FIELD = (7, 11)
PDB_ATOM_NAME_FIELD = (12, 16)
PDB_COORD_FIELDS = ((31,38), (39, 46), (47, 54)) # Taken from ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_Letter.pdf, page 194
PDB_ELEMENT_FIELD = (77, 78)
PDB_CHARGE = (79, 80)

PDB_INT = int
PDB_STR = lambda x: str(x).strip()
def PDB_MAYBE_INT_WITH_SIGN(x: str) -> int:
    if x.strip() == '':
        return None
    else:
        charge_int, charge_sign = x
        if charge_sign == '-':
            sign = -1
        elif charge_sign == '+':
            sign = +1
        else:
            raise Exception('Unexpected charge_sign: "{0}" (x="{1}")'.format(charge_sign, x))
        return sign * int(charge_int)

PDB_FIELD_AND_FORMATTER_FOR_ATTRIBUTE = {
    'atom_index': (PDB_ATOM_INDEX_FIELD, PDB_INT),
    'atom_name': (PDB_ATOM_NAME_FIELD, PDB_STR),
    'element': (PDB_ELEMENT_FIELD, PDB_STR),
    'charge': (PDB_CHARGE, PDB_MAYBE_INT_WITH_SIGN),
}

# Taken from: ftp://ftp.wwpdb.org/pub/pdb/doc/format_descriptions/Format_v33_A4.pdf, p183
HETATM_SPECS = (
    (1, 6) + S,   # Record name
    PDB_ATOM_INDEX_FIELD + S,  # Atom serial number
    PDB_ATOM_NAME_FIELD + S, # Atom name
#    (17, 17) + S, # Alternate location indicator
    (18, 21) + S, # Residue name
    (22, 22) + S, # Chain identifier
    (23, 26) + S, # Residue sequence number
#    (27, 27) + S, # Code for insertion of residues
    PDB_COORD_FIELDS[0] + F, # Orthogonal coordinates for X
    PDB_COORD_FIELDS[1] + F, # Orthogonal coordinates for Y
    PDB_COORD_FIELDS[2] + F, # Orthogonal coordinates for Z
    (55, 60) + S, # Occupancy
    (61, 66) + S, # Temperature factor
    PDB_ELEMENT_FIELD + S, # Element symbol; right-justified
    PDB_CHARGE + CHARGE, # Charge on the atom
)

FIRST_COORDS_INDEX = HETATM_SPECS.index(PDB_COORD_FIELDS[0] + F)

PDB_CHARGE_START = HETATM_SPECS[-1][0] - 1

CONECT_SPECS = (
    (1, 6) + S, # Record name
    (7, 11) + I, # Atom serial number
    (12, 16) + I, # Serial number of bonded atom
    (17, 21) + I, # Serial number of bonded atom
    (22, 26) + I, # Serial number of bonded atom
    (27, 31) + I, # Serial number of bonded atom
)

def PDB_FORMAT_STR(pdb_specs):
    all_fields = []

    for i, fields in enumerate(pdb_specs):
        if i == 0:
            diff = 0
        else:
            diff = (fields[0] -1) - (pdb_specs[i - 1][1])
        if diff != 0:
            all_fields.append(' ' * diff)

        all_fields.append(
            '{' + '{i}:{left_or_right}{len}{formatter}'.format(
            i=i,
            len=(fields[1] - fields[0] + 1),
            left_or_right=fields[2],
            formatter=fields[3],
        ) + '}'
        )
    return ''.join(all_fields)

PDB_TEMPLATE = PDB_FORMAT_STR(HETATM_SPECS)

CONECT_TEMPLATE = PDB_FORMAT_STR(CONECT_SPECS)

NO_CHAIN = ' '

PDB_ATOM_RECORDS = ('ATOM  ', 'HETATM')

PDB_CONNECT_RECORDS = ('CONECT',)

PDB_Atom = NamedTuple(
    'PDB_Atom',
    [
        ('index', int),
        ('name', str),
        ('coordinates', Tuple[float, float, float]),
        ('element', str),
        ('charge', Optional[int]),
    ]
)

assert list(map(len, PDB_ATOM_RECORDS)) == list(map(len, [ PDB_ATOM_RECORDS[0] ]*len(PDB_ATOM_RECORDS))), '{0} != {1}'.format(list(map(len, PDB_ATOM_RECORDS)), list(map(len, [ PDB_ATOM_RECORDS[0] ]*len(PDB_ATOM_RECORDS))))

def is_pdb_atom_line(line: str) -> bool:
    return line[0:len(PDB_ATOM_RECORDS[0])] in PDB_ATOM_RECORDS

def is_pdb_connect_line(line: str) -> bool:
    return line[0:len(PDB_CONNECT_RECORDS[0])] in PDB_CONNECT_RECORDS

def pdb_atoms_in(pdb_str: str) -> List[PDB_Atom]:
    return [
        PDB_Atom(
            index=get_attribute_from_pdb_line('atom_index', line),
            name=get_attribute_from_pdb_line('atom_name', line),
            coordinates=get_coords_from_pdbline(line),
            element=get_attribute_from_pdb_line('element', line),
            charge=get_attribute_from_pdb_line('charge', line),
        )
        for line in pdb_str.splitlines()
        if is_pdb_atom_line(line)
    ]

class PdbElementError(ValueError):
    """An ATOM/HETATM record has no element (cols 77-78) and the name does not settle it."""


_ELEMENT_SYMBOLS = frozenset(
    """H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN GA GE AS SE BR KR
    RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU
    HF TA W RE OS IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF ES FM MD NO LR RF DB SG
    BH HS MT DS RG CN NH FL MC LV TS OG""".split())

# Residues whose atom names follow the PDB polymer convention (NE, CA, CD, HG, OG, SD ... are an
# element letter plus a remoteness letter, never Ne/Ca/Cd/Hg/Og/Sd). Anywhere else a name whose
# first two letters form an element symbol is ambiguous and is refused.
_POLYMER_RESIDUES = frozenset(
    """ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
    HID HIE HIP HSD HSE HSP HISA HISB HISH HISD HISE HISP CYX CYM CYS2 ASH GLH LYN ACE NME NMA NH2 NHE
    A C G U T DA DC DG DT RA RC RG RU ADE CYT GUA URA THY
    HOH WAT SOL TIP3 TIP SPC""".split())

_ONE_LETTER_ELEMENTS = 'HCNOSPFIB'

# Selenomethionine's SE really is selenium, even though MSE is a polymer-like residue.
_NAME_EXCEPTIONS = {('MSE', 'SE'): 'Se'}


def infer_element(name: str, resname: str = '') -> Optional[str]:
    """Element for an atom that has no element column, from its name (owner decision D5).

    Returns the symbol, or ``None`` when the name does not settle it. The name, stripped and
    with digits and punctuation removed, must start with one of H C N O S P F I B. If its first
    two letters also form an element symbol (CL, BR, NA, MG, ZN, SE, CA, CD, HG, FE ...) the name
    is only accepted when ``resname`` is a standard amino acid, nucleotide or water, where the
    PDB name convention makes CA/CB/CD/CE/CG/CZ/HG/HE/NE/OG/SD ... alpha/beta/... atoms; a ligand
    or ion with such a name (``CL1``, ``CA``, ``HG``) is ambiguous and returns None. Any other
    first letter (K, Y, W, U ...) or no letters also returns None. The residue name decides, not
    ATOM vs HETATM: many writers use HETATM for every atom.
    """
    resname = resname.strip().upper()
    letters = ''.join(ch for ch in name.upper() if ch.isalpha())
    if (resname, letters) in _NAME_EXCEPTIONS:
        return _NAME_EXCEPTIONS[(resname, letters)]
    if not letters or letters[0] not in _ONE_LETTER_ELEMENTS:
        return None
    if len(letters) >= 2 and letters[:2] in _ELEMENT_SYMBOLS and resname not in _POLYMER_RESIDUES:
        return None
    return letters[0]


def read_pdb_arrays(text: str, require_elements: bool = True) -> Tuple[List[str], List[str], List[Coordinate], dict]:
    """``(names, elements, coords, extras)`` from the ATOM/HETATM records of ``text``.

    Names are columns 13-16 stripped; elements are columns 77-78 normalised to ``Cl`` form, or,
    when blank, inferred from the name by :func:`infer_element`, else :class:`PdbElementError`
    (message names the 1-based line number). Coordinates are Angstrom tuples (columns 31-54).
    ``extras`` holds per-atom lists ``resnames`` (18-21, the 4-column form ATB writes), ``resseqs`` (23-26, int, or the raw stripped string if not an integer),
    ``serials`` (7-11), ``charges`` (79-80 as in ``pdb_atoms_in``, None if blank),
    ``inferred`` (True where the element was inferred), and ``conect``: sorted unique
    ``(serial, serial)`` pairs, smaller first, from the CONECT records; also ``icodes`` (col 27).
    With ``require_elements=False`` an atom whose element cannot be inferred gets ``''`` instead of
    raising (for callers that only need names and coordinates).
    """
    names, elements, coords = [], [], []
    extras = {'resnames': [], 'resseqs': [], 'serials': [], 'charges': [], 'inferred': [], 'icodes': [], 'conect': []}
    bonds = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if is_pdb_connect_line(line):
            ids = [int(line[a:a + 5]) for a in range(6, 31, 5) if line[a:a + 5].strip()]
            for other in ids[1:]:
                bonds.add((min(ids[0], other), max(ids[0], other)))
            continue
        if not is_pdb_atom_line(line):
            continue
        name = line[12:16].strip()
        resname = line[17:21].strip()
        element = line[76:78].strip()
        inferred = False
        if element:
            element = element[0].upper() + element[1:].lower()
        else:
            symbol = infer_element(name, resname)
            if symbol is None and not require_elements:
                symbol = ''
            if symbol is None:
                raise PdbElementError(
                    'line {0}: atom {1!r} (residue {2!r}) has no element column and the name '
                    'does not settle it'.format(number, name, resname))
            element, inferred = symbol, True
        resseq = line[22:26].strip()
        names.append(name)
        elements.append(element)
        coords.append(get_coords_from_pdbline(line))
        extras['resnames'].append(resname)
        extras['resseqs'].append(int(resseq) if resseq.lstrip('-').isdigit() else resseq)
        extras['serials'].append(get_attribute_from_pdb_line('atom_index', line))
        extras['charges'].append(get_attribute_from_pdb_line('charge', line.ljust(80)))
        extras['inferred'].append(inferred)
        extras['icodes'].append(line[26:27].strip())
    extras['conect'] = sorted(bonds)
    return names, elements, coords, extras


def str_for_pdb_atom(atom: PDB_Atom, residue_name: str = 'RES', chain_id: str = '', residue_number: int = 1) -> str:
    return PDB_TEMPLATE.format(
        PDB_ATOM_RECORDS[0],
        atom.index,
        atom.name[:5],
        residue_name,
        chain_id,
        residue_number,
        *atom.coordinates,
        '',
        '',
        atom.element,
        '' if atom.charge is None else (str(atom.charge) + ('+' if atom.charge >= 0 else '-')),
    )

def get_coords_from_pdbstr(pdb_str: str, filter_empty=False) -> List[Coordinate]:
    optional_filtering = lambda list_of_lines: list_of_lines if not filter_empty else list(filter(bool, list_of_lines))

    return optional_filtering(
        [
            get_coords_from_pdbline(line)
            for line in pdb_str.splitlines()
        ]
    )

def pdb_str_from(atom_names: List[str], element_types: List[str], coords: List[Tuple]) -> str:
    return "\n".join(
        [PDB_TEMPLATE.format(
            PDB_ATOM_RECORDS[0],
            i + 1,
            atom_name,
            "RES",
            "",
            1,
            *coord,
            "",
            "",
            element_type,
            "",
            )
        for i, (atom_name, element_type, coord) in enumerate(zip(atom_names, element_types, coords))
        ]
    )

def get_coords_from_pdbline(line: str) -> Optional[Coordinate]:
    if is_pdb_atom_line(line):
        return tuple(
            map(
                float,
                [
                    line[pdb_coord_field[0] - 1: pdb_coord_field[1]]
                    for pdb_coord_field in PDB_COORD_FIELDS
                ],
            ),
        )
    else:
        return None

def get_attribute_from_pdb_line(attribute: str, line: str) -> Any:
    pdb_field, formatter = PDB_FIELD_AND_FORMATTER_FOR_ATTRIBUTE[attribute]
    return formatter(line[pdb_field[0] - 1: pdb_field[1]])

def get_elements_from_pdbstr(pdb_str: str) -> List[str]:
    return [
        get_attribute_from_pdb_line('element', line)
        for line in pdb_str.splitlines()
        if is_pdb_atom_line(line)
    ]

def pdb_fields(line: str) -> List[str]:
    return [
        line[field_start - 1: field_stop]
        for (field_start, field_stop, _, _) in HETATM_SPECS
    ]

def pdb_conect_line(fields: List[Union[str, int]]) -> str:
    return CONECT_TEMPLATE.format(
        *list(PDB_CONNECT_RECORDS) + fields + [''] * (len(CONECT_SPECS) - len(fields) - 1)
    )

def substitute_coordinates_in(pdb_line: str, coords: Coordinate) -> str:
    assert len(coords) == 3, coords
    fields =  tuple(pdb_fields(pdb_line))
    return PDB_TEMPLATE.format(*fields[0:FIRST_COORDS_INDEX] + coords + fields[FIRST_COORDS_INDEX + 3:])

def pdb_atom_lines_number(pdb_str: str) -> int:
    return len([ 1 for line in pdb_str.splitlines() if is_pdb_atom_line(line) ])

def normalised_element(line: str) -> str:
    return line[76:78].replace(' ', '').upper()

def pdb_charge_str(line: str) -> str:
    assert line[79] in '+- ', 'Detected formal atomic charge larger than 9. Line was: "{0}"'.format(line)
    charge_chars = line[78:80]
    if charge_chars.strip() == '':
        return '0+'
    else:
        return charge_chars

def pdb_formula(pdb_str: str) -> List[Tuple[str, int]]:
    return [
        (element, len(list(occurences)))
        for (element, occurences) in
        groupby(
            sorted([normalised_element(line) for line in pdb_str.splitlines() if is_pdb_atom_line(line)]),
        )
    ]

on_sign = itemgetter(1)

def pdb_total_charge(pdb_str: str) -> int:
    return sum([
        int(sign + '1') * sum([int(x[0]) for x in occurences])
        for (sign, occurences) in
        groupby(
            sorted(
                [pdb_charge_str(line) for line in pdb_str.splitlines() if is_pdb_atom_line(line)],
                key=on_sign,
            ),
            key=on_sign,
        )
    ])

def pdb_formula_string(pdb_str: str, add_charge: bool = False) -> str:
    if add_charge:
        total_charge = pdb_total_charge(pdb_str)
        if total_charge == 0:
            add_charge = False

    return ''.join(
        [
            '{element}{occurence}'.format(element=element.title(), occurence=(occurence if occurence != 1 else ''))
            for (element, occurence) in pdb_formula(pdb_str)
        ]
        +  [ '{sign}{total_charge}'.format(total_charge=abs(total_charge), sign=('+' if total_charge > 0 else '-')) if add_charge else '']
    )

def remove_charge_from(line: str) -> str:
    if is_pdb_atom_line(line):
        return line[0:PDB_CHARGE_START]
    else:
        return line

def remove_pdb_charges(pdb_str: str) -> str:
    return '\n'.join([remove_charge_from(line)for line in pdb_str.splitlines()])

def replace_ATOM_by_HETATM(x: str) -> str:
    return x.replace(*PDB_ATOM_RECORDS)

if __name__ == '__main__':
    print(PDB_TEMPLATE)
    print(CONECT_TEMPLATE)
    print(pdb_conect_line([1, 2, 3]))

    pdb_str = '''
HETATM   36  H15 UNK  1120      -3.379  -2.942  -1.191  1.00  0.00           B1-
HETATM   36  H15 UNK     0      -3.379  -2.942  -1.191  1.00  0.00           A1-
HETATM   36  H15 UNK     0      -3.379  -2.942  -1.191  1.00  0.00          CL1-
HETATM   36  H15 UNK     0      -3.379  -2.942  -1.191  1.00  0.00           H9-
HETATM   36  H15 UNK     0      -3.379  -2.942  -1.191  1.00  0.00          CL3+
HETATM   36  H15 UNK     0      -3.379  -2.942  -1.191  1.00  0.00          CL5+
HETATM   18  C17 RTL A 201      14.349  -6.384 -14.255  1.00  0.00           C  
'''
    print(pdb_formula_string(pdb_str))
    print(pdb_total_charge(pdb_str))

    for line in pdb_str.splitlines():
        if is_pdb_atom_line(line):
            print(line)
            print(substitute_coordinates_in(line, (0., 0., 0.)))
            print()

    print(remove_pdb_charges(pdb_str))

    print(pdb_atoms_in(pdb_str))
