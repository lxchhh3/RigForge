"""ASCII FBX structural lexer.

ASCII FBX is a tree of `Name: args { body }` blocks (with body) and `Name: args`
statements (leaves whose args may wrap across many lines, e.g. vertex arrays).
This lexer identifies the structure and records byte offsets — it does NOT
interpret arg values (numbers, strings, etc.). Higher layers (sections.py,
edits.py) consume the offsets to extract semantic data and perform text edits.

Byte-aware (not str-aware): source files we've seen include non-UTF8 fragments
(e.g. Korean Windows-1252 paths). Structural chars `:`, `{`, `}`, `"`, `;` are
ASCII so we scan over raw bytes safely.

Tokens emitted (structural):
    IDENT_COLON(name, start, end)   identifier immediately followed by `:`
    LBRACE(offset)                  unquoted `{`
    RBRACE(offset)                  unquoted `}`

Strings (`"..."`) and comments (`; ... \\n`) are scanned-past silently so
braces inside them cannot fool the parser. FBX ASCII strings have no `\\"`
escape mechanism — backslashes inside `"..."` are literal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


_IDENT_FIRST = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_REST = _IDENT_FIRST | set(b"0123456789")


class TokenKind(Enum):
    IDENT_COLON = "IDENT_COLON"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    start: int
    end: int            # exclusive
    name: Optional[str] = None  # populated only for IDENT_COLON


def tokenize(source: bytes) -> list[Token]:
    """Scan structural tokens. Skips strings and comments."""
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if c in b" \t\r\n":
            i += 1
            continue
        if c == 0x3B:  # ';'
            while i < n and source[i] != 0x0A:
                i += 1
            continue
        if c == 0x22:  # '"'
            i += 1
            while i < n and source[i] != 0x22:
                i += 1
            if i < n:
                i += 1
            continue
        if c == 0x7B:  # '{'
            tokens.append(Token(TokenKind.LBRACE, i, i + 1))
            i += 1
            continue
        if c == 0x7D:  # '}'
            tokens.append(Token(TokenKind.RBRACE, i, i + 1))
            i += 1
            continue
        if c in _IDENT_FIRST:
            j = i + 1
            while j < n and source[j] in _IDENT_REST:
                j += 1
            k = j
            while k < n and source[k] in b" \t":
                k += 1
            if k < n and source[k] == 0x3A:  # ':'
                name = source[i:j].decode("ascii")
                tokens.append(Token(TokenKind.IDENT_COLON, i, k + 1, name=name))
                i = k + 1
                continue
            # Identifier not followed by ':': opaque content. Skip identifier.
            i = j
            continue
        # Any other byte (digits, signs, commas, dots, '*', etc.): opaque content.
        i += 1
    return tokens


@dataclass
class FBXNode:
    """A node in the ASCII FBX tree.

    All offsets are byte offsets into the original source.
    """
    name: str
    name_span: tuple[int, int]    # offsets of the identifier itself
    intro_end: int                 # offset just after the `:`
    body_open: Optional[int]       # offset of `{`, or None for leaf
    body_close: Optional[int]      # offset of `}`, or None for leaf
    extent_end: int                # exclusive end of this node's text
    children: list["FBXNode"] = field(default_factory=list)
    parent: Optional["FBXNode"] = field(default=None, repr=False)

    @property
    def has_body(self) -> bool:
        return self.body_open is not None

    @property
    def extent_span(self) -> tuple[int, int]:
        return (self.name_span[0], self.extent_end)

    @property
    def args_span(self) -> tuple[int, int]:
        """Byte span of the args text (between `:` and `{` for body nodes,
        between `:` and extent_end for leaves)."""
        if self.body_open is not None:
            return (self.intro_end, self.body_open)
        return (self.intro_end, self.extent_end)

    def args_bytes(self, source: bytes) -> bytes:
        s, e = self.args_span
        return source[s:e]

    def walk(self) -> Iterator["FBXNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def child(self, name: str) -> Optional["FBXNode"]:
        for c in self.children:
            if c.name == name:
                return c
        return None

    def child_all(self, name: str) -> list["FBXNode"]:
        return [c for c in self.children if c.name == name]


@dataclass
class FBXDocument:
    source: bytes
    roots: list[FBXNode] = field(default_factory=list)

    def walk(self) -> Iterator[FBXNode]:
        for r in self.roots:
            yield from r.walk()

    def serialize(self) -> bytes:
        """Round-trip identity: an unmodified document re-serializes to its
        original bytes verbatim. Edits go through edits.py, which returns a
        new bytes object."""
        return self.source

    def root(self, name: str) -> Optional[FBXNode]:
        for r in self.roots:
            if r.name == name:
                return r
        return None


class LexError(RuntimeError):
    """Raised when the ASCII FBX source cannot be parsed structurally."""


def parse(source: bytes) -> FBXDocument:
    tokens = tokenize(source)
    pos = [0]
    n = len(source)

    def _parse_nodes(closing_required: bool, parent: Optional[FBXNode]) -> list[FBXNode]:
        nodes: list[FBXNode] = []
        while pos[0] < len(tokens):
            tok = tokens[pos[0]]

            if tok.kind is TokenKind.RBRACE:
                if not closing_required:
                    raise LexError(f"unexpected '}}' at offset {tok.start}")
                return nodes

            if tok.kind is TokenKind.LBRACE:
                raise LexError(f"unexpected '{{' at offset {tok.start} (no node intro)")

            assert tok.kind is TokenKind.IDENT_COLON and tok.name is not None
            name_start = tok.start
            name_end = name_start + len(tok.name)
            intro_end = tok.end
            pos[0] += 1

            if pos[0] < len(tokens) and tokens[pos[0]].kind is TokenKind.LBRACE:
                body_open = tokens[pos[0]].start
                pos[0] += 1
                node = FBXNode(
                    name=tok.name,
                    name_span=(name_start, name_end),
                    intro_end=intro_end,
                    body_open=body_open,
                    body_close=None,
                    extent_end=0,
                    parent=parent,
                )
                node.children = _parse_nodes(closing_required=True, parent=node)
                if pos[0] >= len(tokens) or tokens[pos[0]].kind is not TokenKind.RBRACE:
                    raise LexError(f"missing '}}' for body opened at {body_open}")
                node.body_close = tokens[pos[0]].start
                node.extent_end = tokens[pos[0]].end  # past the '}'
                pos[0] += 1
                nodes.append(node)
            else:
                # Leaf node: extent goes to start of next token (or EOF).
                leaf_end = tokens[pos[0]].start if pos[0] < len(tokens) else n
                node = FBXNode(
                    name=tok.name,
                    name_span=(name_start, name_end),
                    intro_end=intro_end,
                    body_open=None,
                    body_close=None,
                    extent_end=leaf_end,
                    parent=parent,
                )
                nodes.append(node)
        if closing_required:
            raise LexError("unexpected EOF before '}'")
        return nodes

    roots = _parse_nodes(closing_required=False, parent=None)
    return FBXDocument(source=source, roots=roots)
