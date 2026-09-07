"""Strict native-backend qualification tools.

The package is intentionally independent of the production cache runner.  Its
golden-cache reader authenticates a frozen, possibly incomplete cache directly
and never asks production code to resume or complete it.
"""
