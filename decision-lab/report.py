import json,statistics,html
from pathlib import Path
from run import aggregate
ROOT=Path(__file__).resolve().parent
NAMES={'english':'Laya English','typed':'Laya Typed','jev':'JEV'}
SUITES={'intent_en':'英文客服路由','intent_zh':'中文客服路由','sentiment':'否定与讽刺情绪','policy':'退款规则优先级','workflow':'工作流状态选择','arithmetic':'库存计算','options':'2–16 选项查找／排序','multifield':'一次 8 字段判断','context':'上下文长度与信息位置'}
MODELS=['english','typed','jev']

def main():
 tasks=json.loads((ROOT/'corpus.json').read_text());by_id={t['id']:t for t in tasks}
 acc={m:json.loads((ROOT/f'results/{m}-accuracy.json').read_text()) for m in MODELS}
 loads={m:json.loads((ROOT/f'results/{m}-load.json').read_text()) for m in MODELS}
 assert all(len(acc[m]['rows'])==len(tasks) for m in MODELS),'Incomplete accuracy run'
 assert all(len(loads[m]['records'])==(32 if m!='jev' else 16) for m in MODELS),'Incomplete load run'
 index={m:{r['id']:r for r in acc[m]['rows']} for m in MODELS}
 combined=[]
 for t in tasks:
  rs={m:index[m][t['id']] for m in MODELS}
  combined.append(dict(**t,models=rs))
 scaling=[]
 for m in MODELS:
  for w in ['short','mixed']:
   for mode in ['serial_queue','microbatch'] if m!='jev' else ['parallel_api']:
    for c in [1,4,8,16]:
     reps=[r for r in loads[m]['records'] if r['workload']==w and r['mode']==mode and r['concurrency']==c]
     rows=[r for r in reps for r in r['rows']];wall=sum(r['summary']['wall_s'] for r in reps);a=aggregate(rows)
     a.update(model=m,workload=w,mode=mode,concurrency=c,wall_s=wall,requests_per_s=sum(r['ok'] for r in rows)/wall,correct_requests_per_s=sum(r['ok'] and all(g['correct'] for g in r['grades'].values()) for r in rows)/wall,round_rates=[r['summary']['requests_per_s'] for r in reps],round_p95=[r['summary']['p95_ms'] for r in reps])
     scaling.append(a)
 full={}
 for m in ['english','typed']:
  full[m]={}
  for suite in SUITES:
   ids={r['id'] for r in acc[m]['rows'] if r['suite']==suite and not any(a['state_truncated'] or a['rubric_truncated'] for a in r['audit'].values())}
   full[m][suite]={k:aggregate([index[k][i] for i in ids]) for k in [m,'jev']}
 context=[]
 for words in [0,128,512,1024]:
  for pos in ['start','middle','end']:
   ids=[t['id'] for t in tasks if t['suite']=='context' and t['meta']['filler_words']==words and t['meta']['position']==pos]
   context.append(dict(words=words,position=pos,models={m:aggregate([index[m][i] for i in ids]) for m in MODELS}))
 # Exploratory cascade simulation, never advertised as validated deployment policy.
 cascade=[]
 for m in ['english','typed']:
  for threshold in [.8,.9,.95,.99]:
   local=[];final=[]
   for t in tasks:
    r=index[m][t['id']]
    use=r['ok'] and all(g.get('pmax',0)>=threshold for g in r['grades'].values()) and not any(a['state_truncated'] or a['rubric_truncated'] for a in r['audit'].values())
    chosen=r if use else index['jev'][t['id']]
    if use:local.append(r)
    final.append(chosen)
   cascade.append(dict(model=m,threshold=threshold,local_coverage=len(local)/len(tasks),local_exact_match=aggregate(local)['exact_match'],combined=aggregate(final)))
 failures=[json.loads(line) for line in (ROOT/'results/setup-failures.jsonl').read_text().splitlines()] if (ROOT/'results/setup-failures.jsonl').exists() else []
 summary=dict(setup_failures=failures,accuracy={m:acc[m]['summary'] for m in MODELS},scaling=scaling,full_context=full,context=context,cascade_exploratory=cascade,models={m:acc[m]['metadata'] for m in MODELS},counts=dict(cases=len(tasks),questions=sum(len(t['questions']) for t in tasks),accuracy_calls=sum(len(acc[m]['rows']) for m in MODELS),load_calls=sum(len(r['rows']) for m in MODELS for r in loads[m]['records']),errors=sum(not r['ok'] for m in MODELS for r in acc[m]['rows'])+sum(not r['ok'] for m in MODELS for rr in loads[m]['records'] for r in rr['rows'])))
 (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 lines=['# Laya × JEV：扩展场景与并发实测','', '日期：2026-09-21。设备：Apple M5 Pro，48GB。三模型全部为本轮新请求；未复用此前 JEV 答案。', '',f"本轮 {len(tasks)} 个合成输入、每模型 {sum(len(t['questions']) for t in tasks)} 个判定；准确性请求 {summary['counts']['accuracy_calls']} 次，负载测量请求 {summary['counts']['load_calls']} 次，另有排除统计的预热及批处理一致性检查。测量阶段错误 {summary['counts']['errors']} 次。",'', '## 场景准确率与单请求延迟','', '| 场景 | 判定数 | English | Typed | JEV | English / Typed / JEV P50 ms |','|---|---:|---:|---:|---:|---:|']
 findings=['## 本轮发现','','- 9 类场景的整体准确率均由 JEV 领先；本次没有发现 Laya 在某一完整套件准确率上胜出。逐题仍有 11 个输入至少一个 Laya 版本全对而 JEV 出错，可在交互页面筛选审查，不能据此推断可稳定路由的优势领域。','- Laya 更适合容许一定错误的短文本初筛：中英文路由准确率 91.7%–95.8%，单请求中位延迟约 8 ms；JEV 这两组全对，中位约 300 ms。','- API 并发确实有效：JEV 短任务从单并发约 3.2 请求/s 提升到 16 并发约 40.1 请求/s。但本地跨请求批处理也能扩展：Laya Typed 同时释放 16 个短请求时约 336.2 请求/s、311.7 正确请求/s，P95 49.8 ms。这个吞吐优势伴随错误率，不能代替准确率指标。','- 复杂任务改变结果：16 并发混合负载中，JEV 约 35.1 正确请求/s，超过 English 批处理的 29.5 和 Typed 的 27.4。Laya 原始吞吐更高，但错误抵消了速度优势。','- 多字段错误会累积：每字段约 71%–72% 的 Laya，在 16 条八字段请求上的整条全对率仅 0%／6.25%；JEV 为 100%。这批输入较少，仍足以说明只看字段平均分会掩盖实际交付失败。','- Typed 不是所有任务都优于 English；它在本轮工作流、退款规则和长上下文上更好，但库存计算更差。两者计算准确率都不足一半，JEV 也仅 62.5%，此类确定性计算应优先交给代码。','- 本轮结果支持把 Laya 用于延迟敏感的粗分类候选，把复杂约束交给更可靠的路径；尚不足以证明某个置信度阈值能安全自动分流，详见级联探索。','']
 lines[6:6]=findings
 for s,title in SUITES.items():
  vals=[acc[m]['summary'][s] for m in MODELS]
  lines.append(f"| {title} | {vals[0]['questions']} | "+' | '.join(f"{v['accuracy']:.1%}" for v in vals)+' | '+' / '.join(f"{v['p50_ms']:.1f}" for v in vals)+' |')
 lines += ['', '多字段准确率按字段统计；必须同时关注整条请求全对率：']
 for m in MODELS:lines.append(f"- {NAMES[m]}：8 字段全部正确 {acc[m]['summary']['multifield']['exact_match']:.1%}。")
 lines+=['','## 并发与批处理','', '每个负载 48 请求／轮、重复 2 轮；并发 1、4、8、16。按同步突发运行：一波请求同时释放，整波完成后才发下一波。P95 包含该波内的等待。吞吐使用两轮总成功数除以总墙钟时间，排除预热。', '', '短任务：中英文路由和情绪；混合任务：规则、工作流、计算及多字段。每条多字段请求仍计为一个请求，只有全部字段正确才计入“正确请求／秒”。', '', 'Laya 队列：同一 GPU 串行处理并发到达请求。Laya 批处理：不同输入通过公开 prepare/forward 合批，最多每批 16 个问题；多字段请求可能拆成多个计算批次。JEV：每个并发 worker 独立 HTTPS 连接，无全局锁。']
 for w,title in [('short','短任务'),('mixed','混合复杂任务')]:
  lines+=['',f'### {title}','','| 模型／方式 | 并发 | 请求/s | 正确请求/s | P50 / P95 ms | 100ms 内全对 | 500ms 内全对 |','|---|---:|---:|---:|---:|---:|---:|']
  for x in scaling:
   if x['workload']!=w:continue
   mode={'serial_queue':'串行队列','microbatch':'跨请求批处理','parallel_api':'API 并发'}[x['mode']]
   lines.append(f"| {NAMES[x['model']]} / {mode} | {x['concurrency']} | {x['requests_per_s']:.1f} | {x['correct_requests_per_s']:.1f} | {x['p50_ms']:.1f} / {x['p95_ms']:.1f} | {x['deadline_success']['100']:.1%} | {x['deadline_success']['500']:.1%} |")
 lines+=['','## 截断控制','','下表只保留对应 Laya 能完整读取状态、指令及选项的请求，JEV 使用完全相同子集。','','| 版本 | 场景 | 判定数 | Laya | JEV |','|---|---|---:|---:|---:|']
 for m in ['english','typed']:
  for s in SUITES:
   a,b=full[m][s][m],full[m][s]['jev']
   lines.append(f"| {NAMES[m]} | {SUITES[s]} | {a['questions']} | {a['accuracy']:.1%} | {b['accuracy']:.1%} |")
 lines+=['','长上下文是重复背景日志的定向压力测试，不等同于长文理解。在输入被截断时，仍可能保留前部的关键事实，因此“截断”不意味着每题必错；尾部事实更容易丢失。','','## 级联路由探索（离线模拟）','','若 Laya 每个字段的最高概率达到阈值且没有裁剪，则接收本地答案，其余使用本轮 JEV 答案。这是当前题集上的探索曲线，未用独立测试集验证，不能作为生产可靠性保证，也不代表已部署级联系统。','','| 本地模型 | 阈值 | 本地接收比例 | 被接收请求全对率 | 级联整体请求全对率 |','|---|---:|---:|---:|---:|']
 for x in cascade:lines.append(f"| {NAMES[x['model']]} | {x['threshold']:.2f} | {x['local_coverage']:.1%} | {x['local_exact_match']:.1%} | {x['combined']['exact_match']:.1%} |")
 lines += ['',f'连接预热／准备阶段另有 {len(failures)} 次失败，详情保存在 results/setup-failures.jsonl。失败轮次中止后显式恢复未完成轮次，已完成测量不覆盖；这些失败不混入成功测量延迟，故延迟数字不能代表含建连失败的冷启动体验。']
 lines+=['','## 方法边界与复现','','- 这是本机 MLX 与在线 API 的部署路径比较，不是同硬件纯模型算力比较；网络延迟包含在 JEV 结果中。','- 新增题集由本次脚本生成，标签由人工明确语义或确定性规则给出；全部输入和标签可审查。它是场景诊断集，不是独立、代表性生产数据集。','- 中英文翻译、长度／位置变体和重复负载并非独立样本，因此不将它们伪装为上千条独立事实，也不报告虚假的总体置信区间。','- 每次只发送 state 和 questions，标准答案和套件标签不发送给模型。选项顺序固定随机打乱；各模型拿到相同输入。','- 本地不缓存结果。JEV 不自动重试，API 失败会停止进一步调用；schema 无效和请求失败计错。','- 两个 Laya checkpoint、context 和 head budget 见 summary.json；批处理适配与标准 predict 的对照见各 load.json 的 parity。','- 负载波次顺序固定；温度、系统调度和网络波动可能影响结果。仅测到 16 并发，未探索 JEV 服务配额或更高并发饱和点，也未测试 Laya 多机横向扩展。','- 上一次贪吃蛇服务有全局锁，不能用它判断 JEV API 的并发能力。本次使用独立客户端连接。','', '```sh','cd /Users/bystanders/Desktop/pythonproject/Jev','source ~/.config/typesafe/env','HF_HUB_OFFLINE=1 snake/.venv/bin/python decision-lab/run.py --models english typed jev --load','snake/.venv/bin/python decision-lab/report.py','```','','结果目录已有完整同 corpus 文件时会复用已完成阶段，防止误覆盖和重复计费。要做新一轮测量，应先明确归档整个 results 目录。源码：prepare.py、run.py；原始输入：corpus.json；逐条响应、计时与审计：results/*.jsonl。']
 (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 payload=json.dumps(dict(summary=summary,cases=combined,suites=SUITES,names=NAMES),ensure_ascii=False).replace('</','<\\/')
 template=(ROOT/'explorer-template.html').read_text()
 (ROOT/'explorer.html').write_text(template.replace('__PAYLOAD__',payload))
 print(json.dumps(summary['counts'],indent=2))
if __name__=='__main__':main()
