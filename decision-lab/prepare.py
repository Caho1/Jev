"""Frozen, synthetic paired decision cases. Labels are computed before inference."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=random.Random(260921)
TASKS=[]

def add(suite,case,state,criteria,expected,instructions,group=None,**meta):
 keys=list(criteria);R.shuffle(keys)
 q={'decision':{'type':'choice','instructions':instructions,'criteria':{k:criteria[k] for k in keys}}}
 TASKS.append(dict(id=f'{suite}-{case}',suite=suite,group=group or f'{suite}-{case}',state=state,questions=q,expected={'decision':expected},meta=meta))

INTENTS={
'billing': [('I was charged twice for one subscription. Please refund the duplicate.', '我的订阅被重复扣款，请退还多扣的一笔。'),('Please send the invoice for my latest payment.','请把最近付款的发票发给我。'),('I cancelled last month but you billed me again.','我上个月已经取消订阅，这个月却又扣费了。'),('My payment failed although the card has enough funds.','银行卡余额充足，但付款失败了。'),('How can I change my billing address?','我想修改账单地址，该怎么操作？'),('The receipt shows the wrong tax amount.','收据上的税额不对，请帮我核对。')],
'access':[('I forgot my password and cannot sign in.','我忘了密码，无法登录。'),('The two-factor code never arrives.','双重验证的验证码一直收不到。'),('My account is locked after too many login attempts.','登录错误次数太多，账号被锁定了。'),('Please help recover access to my account.','请帮我恢复账号的访问权限。'),('I lost my security key and cannot authenticate.','我的安全密钥丢了，无法完成身份验证。'),('The password reset link has expired.','重置密码的链接过期了。')],
'bug':[('The application crashes whenever I export a PDF.','每次导出 PDF，程序就崩溃。'),('The search box returns no results for existing records.','明明有这条记录，搜索框却查不到。'),('Clicking Save erases the text I entered.','点保存后，我输入的文字被清空了。'),('The chart displays negative counts for positive data.','数据都是正数，图表却显示成了负数。'),('The upload freezes at 70 percent on every attempt.','每次上传都卡在百分之七十。'),('The calendar duplicates events after a refresh.','刷新后，日历事件重复出现了。')],
'feature':[('Could you add a dark theme?','能增加深色主题吗？'),('I would like support for exporting to Markdown.','希望增加导出 Markdown 的功能。'),('Please add keyboard shortcuts for switching projects.','请增加切换项目的快捷键。'),('It would be useful to schedule reports automatically.','希望可以定时自动生成报告。'),('Can you add an offline editing mode?','能增加离线编辑模式吗？'),('Please support grouping tasks with custom tags.','希望支持用自定义标签给任务分组。')]}
crit={'billing':'Payments, invoices, charges or refunds.','access':'Login, credentials or account recovery.','bug':'An existing feature malfunctions.','feature':'Request a new feature.'}
for lang,col in [('en',0),('zh',1)]:
 for label,pairs in INTENTS.items():
  for i,pair in enumerate(pairs): add('intent_'+lang,f'{label}-{i}',pair[col],crit,label,'Route this support message to exactly one team.',group=f'intent-{label}-{i}')

polarity=[('I love the update. It works perfectly.','positive'),('The product is reliable and easy to use.','positive'),('Support solved the problem quickly. Thank you!','positive'),('I expected a mess, but it is actually excellent.','positive'),('Not bad at all; I am very happy with it.','positive'),('It no longer crashes, and I love using it.','positive'),('It is not perfect, but overall I recommend it.','positive'),('I had doubts. Now I am glad I bought it.','positive'),('This is awful and keeps crashing.','negative'),('I regret buying it. Nothing works.','negative'),('The support team ignored every message.','negative'),('Great, another crash right before the deadline.','negative'),('I cannot say I am happy with this purchase.','negative'),('It used to be excellent, but now it is unusable.','negative'),('I wanted to love it, but I cannot recommend it.','negative'),('Thanks for deleting all my work. Fantastic job.','negative'),('The package arrived on Tuesday.','neutral'),('The app has a menu and a search field.','neutral'),('I installed version 4.2 this morning.','neutral'),('The item weighs two kilograms.','neutral'),('The manual lists three setup steps.','neutral'),('I have neither praise nor complaints.','neutral'),('My colleague likes it; I have not tried it.','neutral'),('The receipt shows an order number.','neutral')]
for i,(s,y) in enumerate(polarity):add('sentiment',i,s,{k:k+' overall attitude' for k in ['positive','negative','neutral']},y,'Classify the speaker\'s overall sentiment, including negation and sarcasm.')

for i in range(32):
 days=[7,14,15,30][i%4];opened=bool(i&4);damaged=bool(i&8);receipt=bool(i&16)
 label='review' if not receipt else 'approve' if damaged or (days<=14 and not opened) else 'deny'
 state=f'Policy: Missing receipt always requires manual review. With a receipt, approve any damaged item. Otherwise approve only unopened items returned within 14 days inclusive. Deny the rest.\nReturn: days_since_delivery={days}; opened={str(opened).lower()}; damaged={str(damaged).lower()}; receipt={str(receipt).lower()}.'
 add('policy',i,state,{'approve':'Approve refund.','deny':'Deny refund.','review':'Manual review.'},label,'Apply the policy exactly. The missing-receipt rule has first priority.')
for i in range(32):
 status=['loading','ready','done','error'][i%4];saved=bool(i&4);match=bool(i&8);enabled=bool(i&16)
 label='wait' if status=='loading' else 'done' if saved and match else 'retry' if status=='error' else 'save' if match and enabled else 'blocked'
 state=f'Goal: save the requested customer record. Current status={status}; saved_indicator={saved}; exact_target_match={match}; save_button_enabled={enabled}. Rules: while loading choose WAIT. Otherwise a saved exact match means DONE. Otherwise an error requires RETRY. Otherwise save only an exact match with an enabled button. All other states are BLOCKED.'
 add('workflow',i,state,{k:k.upper() for k in ['wait','done','retry','save','blocked']},label,'Choose the next action using the ordered rules. Do not treat an unrelated saved record as completion.')
for i in range(32):
 stock=R.randint(0,30);reserved=R.randint(0,15);incoming=R.randint(0,20);order=R.randint(1,30)
 available=stock-reserved
 label='now' if available>=order else 'later' if available+incoming>=order else 'short'
 state=f'Warehouse: physical stock {stock}, already reserved {reserved}, arriving tomorrow {incoming}. New order needs {order}. Available now = physical stock minus reserved. Reserved units cannot serve this order.'
 add('arithmetic',i,state,{'now':'Can fill completely now.','later':'Cannot fill now; can fill after tomorrow\'s arrival.','short':'Still short after tomorrow\'s arrival.'},label,'Decide when the entire new order can be fulfilled.')
for n in [2,4,8,16]:
 for j in range(6):
  waits=R.sample(range(1,90),n);target=R.randrange(n)
  # 3 direct lookup, 3 ranking; every alternative is a genuine option.
  state='Queues: '+', '.join(f'Q{k}={w} minutes' for k,w in enumerate(waits))+'.'
  if j<3: instr=f'Choose queue Q{target} requested by the user. Ignore wait length.';answer=f'Q{target}';kind='lookup'
  else:instr='Choose the queue with the smallest waiting time.';answer=f'Q{waits.index(min(waits))}';kind='ranking'
  add('options',f'{n}-{j}',state,{f'Q{k}':f'Queue {k}' for k in range(n)},answer,instr,option_count=n,kind=kind)

for i in range(16):
 flags={k:bool(R.getrandbits(1)) for k in ['payment','login','crash','export','offline','invoice','upload','search']}
 positive={'payment':'I was charged twice.','login':'I cannot log in.','crash':'The app crashes.','export':'PDF export is broken.','offline':'Offline editing does not work.','invoice':'My invoice is incorrect.','upload':'Uploads fail.','search':'Search returns the wrong results.'}
 negative={'payment':'Payment is fine.','login':'Login works.','crash':'The app is stable.','export':'PDF export works.','offline':'Offline editing works.','invoice':'My invoice is correct.','upload':'Uploads work.','search':'Search works.'}
 state=' '.join(positive[k] if v else negative[k] for k,v in flags.items())
 questions={k:{'type':'choice','instructions':f'Is the customer reporting a problem with {k}?','criteria':{'yes':'A problem is reported.','no':'No problem is reported.'}} for k in flags}
 TASKS.append(dict(id=f'multifield-{i}',suite='multifield',group=f'multifield-{i}',state=state,questions=questions,expected={k:'yes' if v else 'no' for k,v in flags.items()},meta={'fields':8}))

# Identical evidence moved through growing context; group holds content constant.
for words in [0,128,512,1024]:
 for position in ['start','middle','end']:
  for j in range(4):
   answer=['north','south','east','west'][j]
   fact=f'Customer C17 has assigned region {answer.upper()}. '
   filler=('Background log: routine inventory sync completed without changes. '*((words+8)//9)).split()[:words]
   cut={'start':0,'middle':len(filler)//2,'end':len(filler)}[position]
   state=' '.join(filler[:cut])+' '+fact+' '.join(filler[cut:])
   add('context',f'{words}-{position}-{j}',state,{k:k.upper() for k in ['north','south','east','west']},answer,'Return only the assigned region of customer C17. Background inventory logs are irrelevant.',group=f'context-{j}',filler_words=words,position=position)

R.shuffle(TASKS)
(ROOT/'corpus.json').write_text(json.dumps(TASKS,ensure_ascii=False,indent=2)+'\n')
counts={s:sum(t['suite']==s for t in TASKS) for s in sorted({t['suite'] for t in TASKS})}
print(json.dumps(dict(cases=len(TASKS),questions=sum(len(t['questions']) for t in TASKS),suites=counts,sha256=hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest()),indent=2))
