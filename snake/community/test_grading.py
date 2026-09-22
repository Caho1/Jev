import unittest
from run import grade
class GradingTests(unittest.TestCase):
 def test_noul_tie_matches_original_false_rule(self):
  t={'questions':{'q':{'type':'noul'}},'expected':{'q':False},'scoring':'reviews'}
  self.assertTrue(grade(t,{'q':{'type':'noul','noul':.5}})['q']['correct'])
 def test_invalid_distribution_is_wrong(self):
  t={'questions':{'q':{'type':'choice','criteria':{'a':'a','b':'b'}}},'expected':{'q':'a'},'scoring':'jevbench'}
  self.assertFalse(grade(t,{'q':{'type':'choice','probabilities':{'a':.8,'b':.8}}})['q']['valid'])
 def test_rating_uses_original_expected_range(self):
  t={'questions':{'q':{'type':'score','criteria':['1','2','3','4','5']}},'expected':{'q':[4,5]},'scoring':'reviews'}
  a={'q':{'type':'score','score':3.1,'probabilities':{'0':0,'1':0,'2':0,'3':.9,'4':.1}}}
  self.assertTrue(grade(t,a)['q']['correct'])
 def test_ordinal_benchmark_uses_argmax_not_rounded_expectation(self):
  t={'questions':{'q':{'type':'score','criteria':['low','mid','high']}},'expected':{'q':'0'},'scoring':'jevbench'}
  a={'q':{'type':'score','score':.9,'probabilities':{'0':.45,'1':.2,'2':.35}}}
  self.assertTrue(grade(t,a)['q']['correct'])
if __name__=='__main__':unittest.main()
