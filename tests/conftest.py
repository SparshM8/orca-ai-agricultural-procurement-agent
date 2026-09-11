"""Pytest configuration and test fixtures."""

import pytest
import sys
import os

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Ensure unit and integration tests run hermetically with rule_based provider by default
os.environ["AI_PROVIDER"] = "rule_based"

