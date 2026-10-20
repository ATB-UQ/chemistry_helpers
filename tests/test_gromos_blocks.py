"""chemistry_helpers.gromos_blocks: policy tests (owner decision D1, 2026-10-20)."""
import pytest

from chemistry_helpers.gromos_blocks import GromosBlockError, blocks_dict, split_blocks

TOP = """\
# a leading comment
TITLE
A title # with a hash
END
PHYSICALCONSTANTS
# FPEPSI
138.9354
# HBAR
0.0635078
END
ATOMTYPENAME
2
C
H
END
"""


def test_normal_topology_in_order():
    got = split_blocks(TOP)
    assert [n for n, _ in got] == ["TITLE", "PHYSICALCONSTANTS", "ATOMTYPENAME"]
    assert dict(got)["PHYSICALCONSTANTS"] == ["138.9354", "0.0635078"]
    assert dict(got)["ATOMTYPENAME"] == ["2", "C", "H"]


def test_comment_lines_inline_tails_and_blank_lines():
    text = "BOX\n\n  # indented comment\n 1.0  2.0 # tail\n\n3.0\t# x\nEND\n"
    assert split_blocks(text) == [("BOX", ["1.0  2.0", "3.0"])]


def test_title_is_verbatim_and_hash_is_text():
    text = "TITLE\n# not a comment here\n  run 5 # of 10\nEND\nBOX\n1 # c\nEND\n"
    assert split_blocks(text) == [("TITLE", ["# not a comment here", "  run 5 # of 10"]),
                                  ("BOX", ["1"])]


def test_title_ended_by_end_line_only():
    assert split_blocks("TITLE\nsend END here\nEND\n") == [("TITLE", ["send END here"])]


def test_end_recognised_after_tail_and_whitespace_but_not_inside_a_value_line():
    text = "A\n1 END 2\nSEND\n  END  \nB\nx\nEND # done\n"
    assert split_blocks(text) == [("A", ["1 END 2", "SEND"]), ("B", ["x"])]


def test_unterminated_final_block_raises():
    with pytest.raises(GromosBlockError, match="unterminated block 'B'"):
        split_blocks("A\n1\nEND\nB\n2\n")
    with pytest.raises(ValueError):                  # GromosBlockError is a ValueError
        split_blocks("A\n")


def test_unterminated_title_raises():
    with pytest.raises(GromosBlockError):
        split_blocks("TITLE\nsome text\n")


def test_stray_end_raises():
    with pytest.raises(GromosBlockError, match="END with no open block"):
        split_blocks("A\n1\nEND\nEND\n")


def test_empty_and_comment_only_give_no_blocks():
    assert split_blocks("") == []
    assert split_blocks("# only\n\n   \n# comments\n") == []


def test_crlf_and_missing_final_newline():
    assert split_blocks("A\r\n1 # c\r\nEND") == [("A", ["1"])]


def test_duplicate_names_allowed_and_order_kept():
    text = "M\n1\nEND\nX\n2\nEND\nM\n3\nEND\n"
    assert split_blocks(text) == [("M", ["1"]), ("X", ["2"]), ("M", ["3"])]


def test_strip_false_keeps_fixed_width_columns():
    text = "POSITION\n    1 RES   C      1   0.000   1.000   2.000   # tail\nEND\n"
    (name, lines), = split_blocks(text, strip=False)
    assert lines == ["    1 RES   C      1   0.000   1.000   2.000   "]
    assert split_blocks(text)[0][1] == ["1 RES   C      1   0.000   1.000   2.000"]


def test_keep_comments_returns_block_as_written():
    text = "# head\nTORS\n# NRTY NTY\n   3 1\n# 1\n  1 2.0 # tail\n\nEND\n"
    assert split_blocks(text, keep_comments=True) == [
        ("TORS", ["# NRTY NTY", "   3 1", "# 1", "  1 2.0 # tail"])]


def test_blocks_dict_duplicate_policies():
    text = "M\n1\nEND\nX\n2\nEND\nM\n3\nEND\n"
    with pytest.raises(GromosBlockError, match="duplicate block 'M'"):
        blocks_dict(text)
    assert blocks_dict(text, on_duplicate="last") == {"M": ["3"], "X": ["2"]}
    assert blocks_dict(text, on_duplicate="extend") == {"M": ["1", "3"], "X": ["2"]}
    assert list(blocks_dict(text, on_duplicate="extend")) == ["M", "X"]
    with pytest.raises(ValueError):
        blocks_dict(text, on_duplicate="nope")
    assert blocks_dict("A\n x \nEND\n", strip=False) == {"A": [" x "]}


def test_imports_nothing_configured():
    import subprocess, sys
    code = ("import sys, chemistry_helpers.gromos_blocks, chemistry_helpers.geometry;"
            "bad=[m for m in sys.modules if m.split('.')[0] in "
            "('atb_server_settings','atb_helpers','keyring','pymysql','mysql')];"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
