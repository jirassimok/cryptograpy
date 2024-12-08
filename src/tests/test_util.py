"""
Tests for the few cases in the util module that aren't incidentally covered
anywhere else.

These tests are here for coverage, not unit testing; we asssume the uses in
other locations are sufficient to test the module's various helpers.
"""
import unittest

from homework import util

class TestUtilModule(unittest.TestCase):
    def test_supstr_zero(self):
        self.assertEqual(util.supstr(0), '\N{superscript zero}')

    def test_substr_zero(self):
        self.assertEqual(util.substr(0), '\N{subscript zero}')

    def test_copy_args(self):
        # This decorator should only work for the type checker.
        @util.copy_args(print)
        def like_print(a, b):
            return a + b
        self.assertEqual(like_print(4, 7), 11)
