"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — LEXICAL ANALYZER (TOKENIZER)
==============================================================================
Module: krystal_kernel/ksl/ksl_lexer.py
==============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


class KslSyntaxError(Exception):
    """Raised on lexical or syntactic violations in KSL source code."""
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"KSL Syntax Error [{line}:{column}]: {message}")
        self.message = message
        self.line = line
        self.column = column


@dataclass
class Token:
    type: str  # "KEYWORD", "IDENTIFIER", "NUMBER", "STRING", "OPERATOR", "PUNCTUATION", "EOF"
    value: str
    line: int
    column: int


KEYWORDS = {
    "device_group", "constraint", "state", "transition", "ring", "fn",
    "zeroalloc", "let", "observe", "candidates", "predict", "commit",
    "validate", "requires", "effect", "on", "latency", "deadline",
    "within", "prefer", "minimize", "maximize", "hold", "route",
    "batch_size", "workers", "performance", "true", "false",
    "new", "malloc", "free", "return", "if", "else",
    # Epistemic type constructors:
    "Measured", "Predicted", "Validated", "Derived"
}


class KslLexer:
    """Tokenizes KSL source code into a clean stream of Token objects."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            char = self.source[self.pos]

            # Whitespace handling
            if char in " \t\r":
                self.pos += 1
                self.column += 1
                continue
            if char == "\n":
                self.pos += 1
                self.line += 1
                self.column = 1
                continue

            # Single-line comment: // ...
            if char == "/" and self.pos + 1 < self.length and self.source[self.pos + 1] == "/":
                while self.pos < self.length and self.source[self.pos] != "\n":
                    self.pos += 1
                continue

            # Multi-line comment: /* ... */
            if char == "/" and self.pos + 1 < self.length and self.source[self.pos + 1] == "*":
                self.pos += 2
                self.column += 2
                while self.pos + 1 < self.length and not (self.source[self.pos] == "*" and self.source[self.pos + 1] == "/"):
                    if self.source[self.pos] == "\n":
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += 1
                    self.pos += 1
                self.pos += 2
                self.column += 2
                continue

            # Identifiers & Keywords
            if char.isalpha() or char == "_":
                start_col = self.column
                start_pos = self.pos
                while self.pos < self.length and (self.source[self.pos].isalnum() or self.source[self.pos] == "_"):
                    self.pos += 1
                    self.column += 1
                word = self.source[start_pos:self.pos]
                token_type = "KEYWORD" if word in KEYWORDS else "IDENTIFIER"
                tokens.append(Token(token_type, word, self.line, start_col))
                continue

            # Numbers (including suffixes like 20ms, 50us, 128B)
            if char.isdigit():
                start_col = self.column
                start_pos = self.pos
                has_dot = False
                while self.pos < self.length:
                    c = self.source[self.pos]
                    if c.isdigit():
                        self.pos += 1
                        self.column += 1
                    elif c == "." and not has_dot and self.pos + 1 < self.length and self.source[self.pos + 1].isdigit():
                        has_dot = True
                        self.pos += 1
                        self.column += 1
                    elif c.isalpha():  # suffix e.g. ms, us, ns, B, KiB, MB
                        self.pos += 1
                        self.column += 1
                    else:
                        break
                val = self.source[start_pos:self.pos]
                tokens.append(Token("NUMBER", val, self.line, start_col))
                continue

            # Strings: "..."
            if char == '"':
                start_col = self.column
                self.pos += 1
                self.column += 1
                start_pos = self.pos
                while self.pos < self.length and self.source[self.pos] != '"':
                    if self.source[self.pos] == "\n":
                        self.line += 1
                        self.column = 1
                    else:
                        self.column += 1
                    self.pos += 1
                if self.pos >= self.length:
                    raise KslSyntaxError("Unterminated string literal", self.line, start_col)
                str_val = self.source[start_pos:self.pos]
                self.pos += 1  # Skip closing quote
                self.column += 1
                tokens.append(Token("STRING", str_val, self.line, start_col))
                continue

            # Multi-character operators: ->, ==, !=, <=, >=, +=, -=
            start_col = self.column
            two_char = self.source[self.pos:self.pos + 2]
            if two_char in {"->", "==", "!=", "<=", ">=", "+=", "-="}:
                tokens.append(Token("OPERATOR", two_char, self.line, start_col))
                self.pos += 2
                self.column += 2
                continue

            # Single-character operators & punctuation
            if char in {"<", ">", "=", "+", "-", "*", "/"}:
                tokens.append(Token("OPERATOR", char, self.line, start_col))
                self.pos += 1
                self.column += 1
                continue

            if char in {"(", ")", "{", "}", "[", "]", ";", ",", ":", "."}:
                tokens.append(Token("PUNCTUATION", char, self.line, start_col))
                self.pos += 1
                self.column += 1
                continue

            raise KslSyntaxError(f"Unexpected character: {char!r}", self.line, self.column)

        tokens.append(Token("EOF", "", self.line, self.column))
        return tokens
