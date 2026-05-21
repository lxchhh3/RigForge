"""Tests for the ASCII FBX structural lexer."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import (
    FBXDocument,
    FBXNode,
    LexError,
    Token,
    TokenKind,
    parse,
    tokenize,
)


# --- Tokenizer tests ---------------------------------------------------------


def test_tokenize_simple_leaf():
    src = b"Count: 1\n"
    toks = tokenize(src)
    assert len(toks) == 1
    assert toks[0].kind is TokenKind.IDENT_COLON
    assert toks[0].name == "Count"


def test_tokenize_body():
    src = b"Foo: {\n  Bar: 2\n}\n"
    toks = tokenize(src)
    kinds = [t.kind for t in toks]
    assert kinds == [
        TokenKind.IDENT_COLON,  # Foo:
        TokenKind.LBRACE,
        TokenKind.IDENT_COLON,  # Bar:
        TokenKind.RBRACE,
    ]


def test_tokenize_skips_comments():
    src = b"; comment with { fake } braces\nFoo: 1\n"
    toks = tokenize(src)
    assert len(toks) == 1
    assert toks[0].name == "Foo"


def test_tokenize_skips_string_literals():
    """Braces and identifiers inside `"..."` must not be tokenized."""
    src = b'P: "Name", "{not_a_brace}", "Trailing: 1"\nReal: 2\n'
    toks = tokenize(src)
    assert [t.name for t in toks if t.kind is TokenKind.IDENT_COLON] == ["P", "Real"]
    assert not any(t.kind is TokenKind.LBRACE for t in toks)


def test_tokenize_whitespace_before_colon():
    """Properties70 in real FBX uses `Properties70:  {` (double space)."""
    src = b"Properties70 : 100\n"
    toks = tokenize(src)
    assert len(toks) == 1
    assert toks[0].name == "Properties70"


def test_tokenize_ignores_asterisk_count():
    """Vertices: *4782 { ... } — the *N form should be opaque content."""
    src = b"Vertices: *4782 {\n  a: 1.0,2.0\n}\n"
    toks = tokenize(src)
    kinds = [t.kind for t in toks]
    assert kinds == [
        TokenKind.IDENT_COLON,
        TokenKind.LBRACE,
        TokenKind.IDENT_COLON,
        TokenKind.RBRACE,
    ]


# --- Parser tests ------------------------------------------------------------


def test_parse_empty():
    doc = parse(b"")
    assert doc.roots == []


def test_parse_single_leaf():
    src = b"Count: 1\n"
    doc = parse(src)
    assert len(doc.roots) == 1
    root = doc.roots[0]
    assert root.name == "Count"
    assert not root.has_body
    assert root.name_span == (0, 5)
    assert root.intro_end == 6
    assert root.extent_end == len(src)


def test_parse_nested_body():
    src = b"Outer: {\n  Inner: 42\n}\n"
    doc = parse(src)
    assert len(doc.roots) == 1
    outer = doc.roots[0]
    assert outer.name == "Outer"
    assert outer.has_body
    assert len(outer.children) == 1
    inner = outer.children[0]
    assert inner.name == "Inner"
    assert inner.parent is outer
    assert not inner.has_body


def test_parse_sibling_leaves():
    src = b"Foo: {\n  A: 1\n  B: 2\n  C: 3\n}\n"
    doc = parse(src)
    foo = doc.roots[0]
    names = [c.name for c in foo.children]
    assert names == ["A", "B", "C"]


def test_parse_multi_root():
    src = b"FBXHeaderExtension: {\n  Version: 7\n}\nGlobalSettings: {\n  Up: 1\n}\n"
    doc = parse(src)
    assert [r.name for r in doc.roots] == ["FBXHeaderExtension", "GlobalSettings"]


def test_parse_leaf_extent_covers_args():
    """A leaf's extent_end must cover its args text, even if wrapped."""
    src = b"Outer: {\n\ta: 1.0,2.0,\n3.0,4.0\n}\n"
    doc = parse(src)
    outer = doc.roots[0]
    a = outer.children[0]
    assert a.name == "a"
    leaf_text = src[a.intro_end : a.extent_end]
    # The leaf's text should include all the comma-separated numbers, up to
    # just before the closing brace.
    assert b"3.0,4.0" in leaf_text


def test_parse_raises_on_unbalanced_brace():
    with pytest.raises(LexError):
        parse(b"Foo: {\n  Bar: 1\n")  # missing closing brace


def test_parse_raises_on_orphan_close_brace():
    with pytest.raises(LexError):
        parse(b"}\n")


def test_args_span_for_leaf_excludes_colon():
    src = b"Count: 42\n"
    doc = parse(src)
    leaf = doc.roots[0]
    args = src[leaf.args_span[0] : leaf.args_span[1]]
    # Should be everything after the `:` (typically " 42\n").
    assert args.strip() == b"42"


def test_args_span_for_body_node():
    """For a body node, args_span ends at the `{`."""
    src = b'Document: 123, "Scene", "Scene" {\n  Foo: 1\n}\n'
    doc = parse(src)
    doc_node = doc.roots[0]
    args = src[doc_node.args_span[0] : doc_node.args_span[1]]
    assert b"123" in args
    assert b'"Scene"' in args
    assert b"{" not in args


# --- Round-trip on the real Maya_ascii.fbx fixture ---------------------------


def test_roundtrip_real_maya_ascii(maya_fbx_ascii: Path):
    raw = maya_fbx_ascii.read_bytes()
    doc = parse(raw)
    assert doc.serialize() == raw
    # Sanity check on top-level structure
    top_names = [r.name for r in doc.roots]
    expected = {"FBXHeaderExtension", "GlobalSettings", "Documents",
                "References", "Definitions", "Objects", "Connections"}
    assert expected.issubset(set(top_names)), f"missing top-level nodes: got {top_names}"


def test_real_maya_has_many_model_nodes(maya_fbx_ascii: Path):
    """Sanity: a rigged character file should have hundreds of Model nodes
    inside Objects."""
    raw = maya_fbx_ascii.read_bytes()
    doc = parse(raw)
    objects = doc.root("Objects")
    assert objects is not None
    models = objects.child_all("Model")
    assert len(models) > 50, f"expected many Model nodes, got {len(models)}"


def test_real_maya_model_name_extraction(maya_fbx_ascii: Path):
    """Verify name_span pulls the actual identifier text out byte-exact."""
    raw = maya_fbx_ascii.read_bytes()
    doc = parse(raw)
    for node in doc.walk():
        s, e = node.name_span
        assert raw[s:e].decode("ascii") == node.name
