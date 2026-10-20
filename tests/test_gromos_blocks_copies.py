"""Cross-copy table for chemistry_helpers.gromos_blocks (duplication.md groups 3/14).

Each ``copy_*`` is another repo's splitter reproduced verbatim (read-only snapshot,
2026-10-20; the repos are not imported because most are not importable in isolation).
Every copy is run on the same inputs and its answer compared with the decided policy.
``AGREES`` pins, per copy, the inputs where it gives the same blocks (as whitespace-token
streams); the rest are the behaviour changes a migration introduces and are listed in
``DIFFERS`` with the reason. If a snapshot is wrong the table is wrong: re-derive from the
source before relying on it.
"""
import io
import os
import tempfile

import pytest

from chemistry_helpers.gromos_blocks import GromosBlockError, split_blocks


# ---- verbatim copies ------------------------------------------------------------------

class GromosFormatError(Exception):
    pass


def copy_gromos2amber(text):             # gromos2amber/gromos_format.py parse_blocks (+ GromosTopologyParser.__init__)
    fh = io.StringIO(text)
    blocks = {}
    lines = [line for line in fh.readlines()
             if line.strip() and not line[0] == '#']
    start = 0
    for l, line in enumerate(lines):
        if l == start:
            blockname = line.strip()
        elif line == "END\n":
            blocks[blockname] = lines[start:l + 1]
            start = l + 1
        else:
            continue
    if len(blocks) == 0:
        raise GromosFormatError("No blocks found in file.")
    return blocks


def copy_lammps_forcefield(text):        # atb_lammps/lib/GromosForcefield.py _parse_blocks (test/gromos_format.py parse_blocks == gromos2amber's)
    blocks = {}
    lines = [line for line in text.splitlines(True)
             if line.strip() and not line[0] == '#']
    start = 0
    for l, line in enumerate(lines):
        if l == start:
            blockname = line.strip()
        elif line == "END\n":
            blocks[blockname] = [x.strip().split() for x in lines[start:l + 1]]
            start = l + 1
        else:
            continue
    if len(blocks) == 0:
        raise Exception("No blocks parsed.")
    return blocks


def copy_interlocks(text):               # atb_condensed_phase/engines/_interlocks.py _read_blocks (path -> text)
    blocks, name, lines = {}, None, []
    for raw in text.splitlines(True):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if name is None:
            if stripped and not stripped.startswith("#"):
                name, lines = stripped, []
            continue
        if stripped == "END":
            blocks.setdefault(name, []).extend(lines)
            name = None
            continue
        if name == "TITLE":
            continue
        content = line.split("#")[0].strip()
        if content:
            lines.append(content)
    if name is not None:
        raise GromosFormatError("Unterminated block {0}".format(name))
    return blocks


def copy_protein_ff(text):               # atb_protein_ff/gromos_top_energy.py _blocks (path -> text)
    blocks, name, toks = {}, None, []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if name is None:
            name, toks = line.split()[0], []
        elif line == "END":
            blocks[name] = toks
            name = None
        else:
            toks.extend(line.split())
    return blocks


def copy_core_parse_gromos(string):      # core/atb/helpers/parse_gromos.py (gen_valid_lines is not used by parse_GROMOS)
    gromosDict = {}
    lines = string.splitlines()
    for eoh in range(len(lines)):
        if not lines[eoh].strip().startswith('#'):
            break
    gromosDict['head'] = lines[:eoh]
    new_block = None
    for line in lines[eoh:]:
        if not line.strip():
            continue
        if new_block is None:
            new_block = True
        if new_block:
            if line.startswith('#'):
                continue
            key = line.strip()
            gromosDict[key] = []
            new_block = False
        else:
            if line.strip() == 'END':
                new_block = True
                continue
            if line != '':
                gromosDict[key].append(line.strip())
    return gromosDict


def copy_atb_helpers_get_blocks(content, filter_comments=True):   # atb_helpers/gromos/blocks.py get_blocks
    from re import search
    END_KEYWORD_PATTERN = '\nEND\n'

    def block_header(lines):
        return [line for line in lines if line[0] != '#'][0]

    def block_body(lines):
        return (lines[1:] if not filter_comments
                else list(filter(lambda x: search('^#', x) is None, lines[1:])))

    if not content[-1] == '\n':
        content += '\n'
    return [(block_header(bl), block_body(bl))
            for bl in [list(filter(bool, s.splitlines())) for s in content.split(END_KEYWORD_PATTERN)]
            if len(bl) > 0]


def copy_job_wrapper_split_end(text):    # gromos_job_wrapper build_gromos_run_config / ene_ana: content.split('END')
    return text.split('END')


def copy_bonded_motifs_block_text(text, block):   # atb_bonded_motifs/ifp.py block_text (keeps comments; no END check on tails)
    lines, inside = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if not inside:
            if stripped == block:
                inside = True
            continue
        if stripped == "END":
            return lines
        lines.append(line)
    return lines if inside else None


# ---- normalisation: every copy -> [(name, tokens)] or an exception name ---------------

def tokens(lines):
    return [t for line in lines for t in line.split()]


def ours(text):
    try:
        return [(n, tokens(ls)) for n, ls in split_blocks(text)]
    except GromosBlockError:
        return "GromosBlockError"


def _dict_form(d, strip_frame):
    out = []
    for name, lines in d.items():
        if name == "head":
            continue
        if strip_frame:                       # name line first, END last
            lines = lines[1:-1]
        out.append((name, lines))
    return out


def run(copy, text):
    try:
        if copy == "gromos2amber":
            return [(n, tokens(ls[1:-1])) for n, ls in copy_gromos2amber(text).items()]
        if copy == "atb_lammps":
            return [(n, [t for row in rows[1:-1] for t in row]) for n, rows in copy_lammps_forcefield(text).items()]
        if copy == "atb_condensed_phase":
            return [(n, tokens(ls)) for n, ls in copy_interlocks(text).items()]
        if copy == "atb_protein_ff":
            return [(n, t) for n, t in copy_protein_ff(text).items()]
        if copy == "core":
            return [(n, tokens(ls)) for n, ls in copy_core_parse_gromos(text).items() if n != "head"]
        if copy == "atb_helpers":
            return [(h.strip(), tokens(ls)) for h, ls in copy_atb_helpers_get_blocks(text)]
    except IndexError:
        return "IndexError"
    except Exception as exc:                  # noqa: BLE001 -- the table records whatever a copy raises
        return type(exc).__name__
    raise KeyError(copy)


COPIES = ["gromos2amber", "atb_lammps", "atb_condensed_phase", "atb_protein_ff", "core", "atb_helpers"]
DICT_COPIES = {"gromos2amber", "atb_lammps", "atb_condensed_phase", "atb_protein_ff", "core"}

# ---- inputs ---------------------------------------------------------------------------

INPUTS = {
    "topology": "# comment\nTITLE\nA water box\nEND\nPHYSICALCONSTANTS\n# FPEPSI\n138.9354\nEND\n"
                "ATOMTYPENAME\n2\nC\nH\nEND\n",
    "ifp_counter_comments": "# ifp\nFORCEFIELD\n54A7\nEND\nMASSATOMTYPECODE\n# NRMATY NMATY\n   3   2\n"
                            "# MATY\n  1  12.011\n  2  1.008 # H\nEND\n",
    "cnf": "TITLE\nframe 1\nEND\nPOSITION\n    1 SOLV  OW    1   0.1   0.2   0.3\nEND\nBOX\n   2.0   2.0   2.0\nEND\n",
    "comment_tail": "BOX\n 1.0 2.0 # tail\n3.0\nEND\n",
    "hash_line_in_block": "BOX\n# whole line\n1.0\nEND\n",
    "blank_lines": "\n\nBOX\n\n1.0\n\n\n2.0\n\nEND\n\n",
    "title_with_hash": "TITLE\nrun # 5\nEND\nBOX\n1.0\nEND\n",
    "unterminated_final": "BOX\n1.0\nEND\nTIMESTEP\n0 0.0\n",
    "unterminated_only": "BOX\n1.0\n",
    "end_in_value_line": "BOX\n1 END 2\nSEND\nEND\nX\n3\nEND\n",
    "end_with_tail": "BOX\n1.0\nEND # done\nX\n2\nEND\n",
    "end_trailing_space": "BOX\n1.0\nEND  \nX\n2\nEND\n",
    "indented_name_and_end": "  BOX  \n  1.0  \n  END\n",
    "no_trailing_newline": "BOX\n1.0\nEND",
    "crlf": "BOX\r\n1.0 # c\r\nEND\r\n",
    "duplicate_blocks": "MTB\n1\nEND\nX\n2\nEND\nMTB\n3\nEND\n",
    "empty": "",
    "comments_only": "# a\n# b\n",
    "stray_end": "BOX\n1\nEND\nEND\n",
}


def table():
    rows = {}
    for iname, text in INPUTS.items():
        want = ours(text)
        rows[iname] = {}
        for c in COPIES:
            got = run(c, text)
            ref = want
            if c in DICT_COPIES and isinstance(want, list):     # a dict keeps one entry per name
                ref = list({n: t for n, t in want}.items())
            if c == "atb_condensed_phase" and isinstance(want, list):   # merges duplicates, skips TITLE
                merged = {}
                for n, t in want:
                    if n != "TITLE":
                        merged.setdefault(n, []).extend(t)
                ref_ic = list(merged.items())
                rows[iname][c] = (got == ref_ic, got)
                continue
            if c == "atb_protein_ff" and isinstance(want, list):
                pass
            rows[iname][c] = (got == ref, got)
    return rows


TABLE = table()

# (input, copy) pairs whose output differs from the decided policy, as observed on the
# snapshots above. Reasons, by copy, are in the phase-2 report; the set pins the table.
DIFFERS = {
    ('topology', 'atb_condensed_phase'),
    ('topology', 'core'),
    ('topology', 'atb_helpers'),
    ('ifp_counter_comments', 'gromos2amber'),
    ('ifp_counter_comments', 'atb_lammps'),
    ('ifp_counter_comments', 'core'),
    ('ifp_counter_comments', 'atb_helpers'),
    ('cnf', 'atb_condensed_phase'),
    ('comment_tail', 'gromos2amber'),
    ('comment_tail', 'atb_lammps'),
    ('comment_tail', 'core'),
    ('comment_tail', 'atb_helpers'),
    ('hash_line_in_block', 'core'),
    ('title_with_hash', 'atb_condensed_phase'),
    ('title_with_hash', 'atb_protein_ff'),
    ('unterminated_final', 'gromos2amber'),
    ('unterminated_final', 'atb_lammps'),
    ('unterminated_final', 'atb_condensed_phase'),
    ('unterminated_final', 'atb_protein_ff'),
    ('unterminated_final', 'core'),
    ('unterminated_final', 'atb_helpers'),
    ('unterminated_only', 'gromos2amber'),
    ('unterminated_only', 'atb_lammps'),
    ('unterminated_only', 'atb_condensed_phase'),
    ('unterminated_only', 'atb_protein_ff'),
    ('unterminated_only', 'core'),
    ('unterminated_only', 'atb_helpers'),
    ('end_with_tail', 'gromos2amber'),
    ('end_with_tail', 'atb_lammps'),
    ('end_with_tail', 'atb_condensed_phase'),
    ('end_with_tail', 'core'),
    ('end_with_tail', 'atb_helpers'),
    ('end_trailing_space', 'gromos2amber'),
    ('end_trailing_space', 'atb_lammps'),
    ('end_trailing_space', 'atb_helpers'),
    ('indented_name_and_end', 'gromos2amber'),
    ('indented_name_and_end', 'atb_lammps'),
    ('indented_name_and_end', 'atb_helpers'),
    ('no_trailing_newline', 'gromos2amber'),
    ('no_trailing_newline', 'atb_lammps'),
    ('crlf', 'gromos2amber'),
    ('crlf', 'atb_lammps'),
    ('crlf', 'core'),
    ('crlf', 'atb_helpers'),
    ('empty', 'gromos2amber'),
    ('empty', 'atb_lammps'),
    ('empty', 'core'),
    ('empty', 'atb_helpers'),
    ('comments_only', 'gromos2amber'),
    ('comments_only', 'atb_lammps'),
    ('comments_only', 'atb_helpers'),
    ('stray_end', 'gromos2amber'),
    ('stray_end', 'atb_lammps'),
    ('stray_end', 'atb_condensed_phase'),
    ('stray_end', 'atb_protein_ff'),
    ('stray_end', 'core'),
    ('stray_end', 'atb_helpers'),
}


@pytest.mark.parametrize("iname", sorted(INPUTS))
def test_cross_copy_table_is_pinned(iname):
    observed = {c for c in COPIES if not TABLE[iname][c][0]}
    pinned = {c for (i, c) in DIFFERS if i == iname}
    assert observed == pinned, (iname, sorted(observed ^ pinned))


def test_atb_helpers_leading_comment_duplicates_the_first_header():
    """Quirk of atb_helpers.gromos.blocks.get_blocks: header = first non-comment line but
    the body is lines[1:], so a comment before the first block puts the header in its body."""
    got = copy_atb_helpers_get_blocks("# c\nFORCEFIELD\n54A7\nEND\n")
    assert got == [("FORCEFIELD", ["FORCEFIELD", "54A7"])]


def test_core_parse_gromos_keeps_comments_inside_blocks():
    got = copy_core_parse_gromos("A\n# c\n1 # t\nEND\n")
    assert got["A"] == ["# c", "1 # t"]       # needed by ifp readers: use keep_comments=True


def test_ifp_block_text_equals_keep_comments_up_to_blank_lines():
    text = "# h\nMASS\n# NR\n 3 2\n  1 12.0 # c\nEND\n"
    assert copy_bonded_motifs_block_text(text, "MASS") == dict(split_blocks(text, keep_comments=True))["MASS"]
    blank = "MASS\n 1\n\n 2\nEND\n"
    assert copy_bonded_motifs_block_text(blank, "MASS") != dict(split_blocks(blank, keep_comments=True))["MASS"]


def test_job_wrapper_split_end_cuts_inside_words():
    assert copy_job_wrapper_split_end("BOX\nSEND\nEND\n") == ["BOX\nS", "\n", "\n"]

