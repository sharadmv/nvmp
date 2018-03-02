import logging
logging.getLogger('tensorflow').disabled = True
from deepx import T
import unittest

class BaseTest(unittest.TestCase):

    def setUp(self):
        self.session = T.interactive_session()

    def tearDown(self):
        self.session.close()
