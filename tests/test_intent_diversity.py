import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.intake import draft_requirements
from forge.evalgen import Gen
from forge.docs import Doc
class Mock:
    model='fixture'
    def json(self, messages, **kw):
        return {'must_answer_topics':['borrowing rules'], 'tool_signals':{}}
req=draft_requirements('Build a library assistant. It should calculate overdue fines and refuse unsupported requests.', [], [], Mock())
assert req['tool_signals']['needs_calculation']
assert any('calculate overdue fines' in t['topic'] for t in req['must_answer_topics'])
assert not draft_requirements('Do not calculate fines.', [], [], Mock())['tool_signals']['needs_calculation']
g=Gen({'scope':'library'},[Doc('a','a','text','Students may borrow four books at once. Faculty may borrow eight books at once.')],Mock())
seen=[]
g.batch=lambda typ,n,evidence,prompt: seen.append(next(iter(evidence.values())))
g.batched_answerable(4,'direct')
assert len(set(seen))==2
print('INTENT AND DIVERSITY TESTS PASSED')
