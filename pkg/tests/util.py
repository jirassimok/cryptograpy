import unittest

# class TestResult(unittest.TestResult):
#     def __init__(self, *a, **k):
#         super().__init__(*a, **k)
#         print('init')
#
#     def addSubTest(self, test, subtest, outcome):
#         result = super().addSubTest(self, test, subtest, outcome)
#         self.testsRun += 1
#         return result


# class TestCase(unittest.TestCase):
#     def shortDescription(self):
#         desc = super().shortDescription()
#         note = 'Due to subtests, the number of tests will be inaccurate.'
#         return note if desc is None else f'{desc} {note}'
