import unittest
from run import grade
class GradingTests(unittest.TestCase):
 def setUp(self):
  self.task={'questions':{'d':{'type':'choice','criteria':{'a':'A','b':'B'}}},'expected':{'d':'a'}}
 def test_valid_correct(self):
  r=grade(self.task,{'d':{'type':'choice','choice':'a','probabilities':{'a':.9,'b':.1}}})['d']
  self.assertTrue(r['correct']);self.assertAlmostEqual(r['brier'],.02)
 def test_reject_malformed(self):
  for a in [{},{'type':'choice','choice':'a','probabilities':{'a':1.0}},{'type':'choice','choice':'a','probabilities':{'a':float('nan'),'b':0}},{'type':'choice','choice':'a','probabilities':{'a':.1,'b':.9}}]:
   self.assertFalse(grade(self.task,{'d':a})['d']['valid'])
 def test_error_is_wrong(self):self.assertFalse(grade(self.task,{})['d']['correct'])
if __name__=='__main__':unittest.main()
