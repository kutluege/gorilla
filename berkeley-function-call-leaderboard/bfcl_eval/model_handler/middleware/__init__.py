"""Middleware utilities that sit between the model handler and the BFCL tool-execution boundary.

These components are intentionally non-invasive: they operate on the JSON tool-result
strings and decoded tool-call strings that already flow through the handler, never on the
benchmark data, the memory backend internals, or the scoring code.
"""
