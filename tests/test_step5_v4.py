import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.evalgen import Gen
from forge.docs import load_folder
class Mock:
    model='fixture'
    def json(self,messages,**kw):
        if 'TASK: evalgen_batch' in messages[0]['content']:
            return {'cases':[{'question':'What is the annual allowance?', 'key_facts':['24 days'], 'reference_answer':'24 days'}, {'question':'How much unused leave rolls over?', 'key_facts':['5 days'], 'reference_answer':'5 days'}, {'question':'What is the fictional bonus?', 'key_facts':['999 days'], 'reference_answer':'999 days'}]}
        return {'verdicts':[{'index':i,'valid':True,'duplicate':False,'key_facts':[f],'reference_answer':f} for i,f in enumerate(['24 days','5 days'])]}
g=Gen({'scope':'Leave policy'},load_folder('samples/hr_policy'),Mock())
g.batch('answerable',3,{'leave.txt':'Annual allowance is 24 days. Carry-over is 5 days.'},'test')
assert len(g.cases)==2 and g.calls==2 and g.dropped['fact_not_in_source']==1
# Same evidence in both documents cannot establish distinct multi-document facts.
g=Gen({'scope':'Leave policy'},load_folder('samples/hr_policy'),Mock())
g.batch('multi_doc',3,{'a':'24 days and 5 days','b':'24 days and 5 days'},'test')
assert not g.cases and g.dropped['multi_not_spanning_both']==2
print('STEP 5 V4 BATCH TESTS PASSED')
