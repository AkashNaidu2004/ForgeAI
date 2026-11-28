import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.evalgen import Gen
from forge.docs import Doc
class Mock:
    model='fixture'
    def json(self,messages,**kw):
        p=messages[0]['content']
        if 'TASK: evalgen_evidence_multi' in p:
            a,b=p.split('\nA: ')[1].split('\nB: '); b=b.split('\nPrior: ')[0]
            return {'cases':[{'question':'What is the leave entitlement and meal allowance?', 'fact_a':a,'fact_b':b},{'question':'Invented?', 'fact_a':'invented fact here','fact_b':b}]}
        return {'verdicts':[{'index':0,'valid':True,'duplicate':False}]}
g=Gen({'scope':'HR'},[Doc('a','a','text','Annual leave is 24 days.'),Doc('b','b','text','Meal allowance is 40 USD.')],Mock())
g.batched_multi(1)
assert len(g.cases)==1 and g.calls==2 and g.dropped['invalid_evidence_pair']==1
assert len(g.cases[0]['expected_sources'])==2
print('STEP 5 V5 EVIDENCE TEST PASSED')
