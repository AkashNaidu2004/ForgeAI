"""Offline regression tests for the verification path and judge safeguards."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.evalgen import Gen
from forge.docs import load_folder
from forge.judge import llm_check, judge_case
class Mock:
    model = 'fixture'
    def __init__(self, response): self.response = response
    def json(self, messages, **kw): return self.response
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
req = {'scope':'Leave policy', 'must_answer_topics':[{'topic':'carry-over'}]}
text = 'Annual leave is 24 days. Carry-over is 5 days.'
r = {'question':'What is the carry-over limit?', 'key_facts':['24 days','5 days'], 'reference_answer':'5 days'}
v = {'valid':True,'duplicate':False,'key_facts':['5 days'],'reference_answer':'5 days','topic':'carry-over'}
m = Mock(v)
g = Gen(req, load_folder(os.path.join(root,'samples','hr_policy')), m)
assert g._check(r,[text])[1] == ['5 days']
g._add('answerable',r['question'],['5 days'],['leave.txt'],'5 days',{'leave.txt':text})
assert g.covered['carry-over'] == 1
m.response = dict(v,duplicate=True)
assert g._check(dict(r,question='How much leave rolls over?'),[text]) is None
m.response = dict(v,key_facts=['999 days'])
assert g._check(dict(r,question='How much leave rolls over?'),[text]) is None
m.response = dict(v,reference_answer='No idea')
assert g._check(dict(r,question='How much leave rolls over?'),[text]) is None
case={'id':'x','type':'answerable','question':'What is the price?','key_facts':['420 USD'],'expected_behavior':'answer','expected_sources':['a']}
m.response={'facts':[{'fact':'420 USD','stated':True,'quote':'It is cheap'}],'unsupported_claims':['stock is 420 USD'],'flags_missing':False}
k=llm_check(case,'It is cheap. stock is 420 USD','price is 420 USD',m)
assert not k['facts']['420 USD'] and k['unsupported_claims']
m.response={'facts':[{'fact':'unrelated','stated':True,'quote':'420 USD'}],'unsupported_claims':[]}
assert not llm_check(case,'420 USD','price is 420 USD',m)['facts']['420 USD']
case['key_facts']=['99999']
res={'answer':'99999','sources':['a'],'refused':False,'context':[{'id':'a0','source':'a','text':'price is 420 USD'}]}
assert not judge_case(case,res,None,{'a'})['grounded']
print('STEP 4 V2 REGRESSION TESTS PASSED')
