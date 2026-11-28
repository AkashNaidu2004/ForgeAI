import sys, os
from unittest.mock import patch, Mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.llm import LLM, RateLimitError
from forge.evalgen import Gen
from forge.docs import load_folder
r=Mock(status_code=429, headers={'Retry-After':'2'})
l=LLM(); l.provider='openai'
with patch('forge.llm.requests.post', return_value=r) as post, patch('forge.llm.time.sleep') as sleep:
    try: l.chat([{'role':'user','content':'test'}]); raise AssertionError('expected rate limit')
    except RateLimitError: pass
    assert post.call_count==4 and sleep.call_count==3
r.headers={'Retry-After':'120'}
with patch('forge.llm.requests.post', return_value=r) as post, patch('forge.llm.time.sleep') as sleep:
    try: l.chat([{'role':'user','content':'test'}]); raise AssertionError('expected rate limit')
    except RateLimitError: pass
    assert post.call_count==1 and sleep.call_count==0
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
g=Gen({'scope':'HR','must_answer_topics':[]},load_folder(os.path.join(root,'samples','hr_policy')),l)
with patch.object(l,'json',side_effect=RateLimitError('quota')):
    try: g._llm('test'); raise AssertionError('must propagate quota failure')
    except RateLimitError: pass
seen=[]; g.checkpoint=lambda obj: seen.append(len(obj.cases))
g._add('answerable','question',['fact'],['doc'],'fact',{'doc':'fact'})
assert seen==[1]
print('API RELIABILITY TESTS PASSED')
