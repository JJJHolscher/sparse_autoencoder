#! /usr/bin/env python3
# vim:fenc=utf-8

"""
Example asserts:

self.assertEqual(code(), 'Hello World!')
self.assertTrue(code())
self.assertFalse(code())
with self.assertRaises(TypeError):
    code()
"""

import unittest

from src.__main__ import *


class TestMain(unittest.TestCase):
    def test_main(self):
        self.assertEqual(UNITTESTS(), 0)

if __name__ == "__main__":
    unittest.main()
