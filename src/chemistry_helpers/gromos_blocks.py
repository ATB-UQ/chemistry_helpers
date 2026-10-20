"""Split GROMOS block-structured text (topology, ifp, mtb, cnf/g96, md/imd input) into
named blocks. Pure string handling: no I/O, no configuration, no third-party imports.

Policy (owner decision D1, 2026-10-20):

* A block is a name line, value lines, and a line that is exactly ``END``.
* Comments: a line whose first non-blank character is ``#`` is dropped; on every other
  line the text from the first ``#`` is stripped (inline tail). Blank lines are dropped.
* ``TITLE`` is the exception: its lines are returned verbatim (only the line ending is
  removed; blank lines are still dropped), so a ``#`` in a title is text, not a comment.
  Only a line that is exactly ``END`` (after surrounding whitespace is stripped) ends it.
* ``END`` is recognised on the stripped line, after the inline tail is removed, so
  ``END # done`` terminates but ``1 END 2`` and ``SEND`` do not.
* A block still open at the end of the text raises :class:`GromosBlockError`, as does an
  ``END`` with no open block. Nothing is silently dropped.
* Block order is kept and duplicate names are allowed (an mtb repeats ``MTBUILDBLSOLUTE``).
* Value lines are returned stripped of surrounding whitespace, unless ``strip=False``
  (fixed-width readers): then the line keeps its leading and trailing blanks (after the
  tail is cut) and only the line ending is removed.
* ``keep_comments=True`` is for readers that need the file as written (ifp counter lines
  are located by their comments): every line inside a block is kept verbatim, comments
  and tails included; only blank lines are dropped, and ``END`` is the stripped line.

``split_blocks`` is the primitive; ``blocks_dict`` is the convenience for callers that
want ``{name: lines}``.
"""
from typing import Dict, List, Tuple

__all__ = ["GromosBlockError", "split_blocks", "blocks_dict"]

END = "END"
COMMENT = "#"
TITLE = "TITLE"


class GromosBlockError(ValueError):
    """The text is not well-formed GROMOS block structure (unterminated block, stray END)."""


def split_blocks(text: str, strip: bool = True, keep_comments: bool = False
                 ) -> List[Tuple[str, List[str]]]:
    """``[(block_name, [value lines])]`` in file order. See the module docstring for the
    comment, TITLE, END and error policy. Empty or comment-only text gives ``[]``."""
    blocks: List[Tuple[str, List[str]]] = []
    name = None
    start = 0
    lines: List[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if name is None:
            if stripped.startswith(COMMENT):
                continue
            head = stripped.split(COMMENT, 1)[0].strip()
            if head == END:
                raise GromosBlockError("line {0}: END with no open block".format(number))
            name, start, lines = head, number, []
            continue
        if keep_comments or name == TITLE:
            if stripped == END:
                blocks.append((name, lines))
                name = None
            else:
                lines.append(raw)
            continue
        if stripped.startswith(COMMENT):
            continue
        content = raw.split(COMMENT, 1)[0]
        if content.strip() == END:
            blocks.append((name, lines))
            name = None
            continue
        if content.strip():
            lines.append(content.strip() if strip else content.rstrip("\r\n"))
    if name is not None:
        raise GromosBlockError("unterminated block {0!r} (opened at line {1})".format(name, start))
    return blocks


def blocks_dict(text: str, on_duplicate: str = "error", **kwargs) -> Dict[str, List[str]]:
    """``{block_name: lines}``. ``on_duplicate``: ``"error"`` (default, raises
    :class:`GromosBlockError`), ``"last"`` (a later block replaces) or ``"extend"``
    (lines are concatenated in file order). Other keywords go to :func:`split_blocks`."""
    if on_duplicate not in ("error", "last", "extend"):
        raise ValueError("on_duplicate must be 'error', 'last' or 'extend'")
    result: Dict[str, List[str]] = {}
    for name, lines in split_blocks(text, **kwargs):
        if name in result:
            if on_duplicate == "error":
                raise GromosBlockError("duplicate block {0!r}".format(name))
            if on_duplicate == "extend":
                result[name].extend(lines)
                continue
        result[name] = list(lines)
    return result
