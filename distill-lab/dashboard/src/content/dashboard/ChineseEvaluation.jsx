import React from "react";
import {DataComponent, DataTable, EvidenceChart, Section, Dropdown, Button, Dialog, useDataApp} from "../../data-app-public.jsx";

const pct=value=>Number.isFinite(value)?`${(value*100).toFixed(2)}%`:"—";
const fixed=value=>Number.isFinite(value)?value.toFixed(2):"—";
const pp=value=>Number.isFinite(value)?`${value>0?"+":""}${(value*100).toFixed(2)} pp`:"—";

export function ChineseEvaluation(){
  const {reviewedRows}=useDataApp();
  const runRows=reviewedRows("zh_run"),run=runRows[0]??{};
  const summary=reviewedRows("zh_summary"),performance=reviewedRows("zh_performance"),cases=reviewedRows("zh_cases"),classes=reviewedRows("zh_classes");
  const audit=reviewedRows("zh_audit");
  const [language,setLanguage]=React.useState("zh"),[intent,setIntent]=React.useState("all"),[outcome,setOutcome]=React.useState("all"),[search,setSearch]=React.useState("");
  const [textScope,setTextScope]=React.useState("all");
  const [selected,setSelected]=React.useState(null);
  const choices=["all",...new Set(cases.map(row=>row.expected))];
  const names=Object.fromEntries(cases.map(row=>[row.expected,row.expectedZh]));
  const filtered=cases.filter(row=>row.language===language && (intent==="all"||row.expected===intent) &&
    (textScope==="all" || (textScope==="han"?row.containsHan:!row.containsHan)) &&
    (outcome==="all"||(outcome==="wrong"?row.tunedCorrect===false:row.tunedCorrect===true)) &&
    [row.id,row.textZh,row.textEn,row.expected,row.expectedZh].some(value=>value.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())));
  const answer=(model,row)=><div className="jev-case-answer"><span className="jev-case-label">{row[`${model}AnswerZh`]??"待评测"}</span><span className={`jev-case-verdict ${row[`${model}Correct`]?"correct":"incorrect"}`}>{row[`${model}Correct`]===null?"":row[`${model}Correct`]?"✓ 正确":"× 错误"}<span className="jev-case-probability">{pct(row[`${model}Confidence`])}</span></span></div>;
  return <div>
    <DataComponent id="zh-run-summary" queryId="zh_run" title={run.complete?"中文本地评测已完成":"中文本地评测"} kind="metric" variant="card" sourceRows={runRows} displayRows={runRows}>
      <div className="jev-progress-row" data-reviewed-rows><strong>{run.records} 条中文子集记录</strong><span>{run.classes} 类 · Mac MPS · FP32 · batch 4{run.complete?` · 全流程 ${fixed(run.wallSeconds)} 秒`:` · ${run.completed} / ${run.total} 次预测`}</span></div>
      <p className="training-muted">MInDS-14 中文母语众包语音的转写及其英文机器译文。官方标为 train，本项目将全部留作外部评测，未参与微调、选模或校准。这里使用 14 类候选，与之前的 BANKING77 77 类测试单独计分。</p>
      <p className="training-muted" data-reviewed-rows>{run.records} 条原始记录，归一化中文去重后 {run.uniqueChinese} 条；保留全部原始记录评分。{run.error}</p>
    </DataComponent>
    {run.complete&&<>
      <Section id="zh-accuracy-section" title="同一道题，中文与英文译文的差距" columns={2}>
        <EvidenceChart id="zh-language-accuracy" queryId="zh_summary" title="中英文准确率" variant="card" height={285} rows={summary} sourceRows={summary}
          spec={{type:"bar",x:"model",y:"zhAccuracyRate",fields:["zhAccuracyRate","enAccuracyRate"],startAtZero:true,stackable:false,showLegend:true,valueDecimals:2,legend:{labels:{zhAccuracyRate:"中文原文",enAccuracyRate:"英文译文"}}}}/>
        <DataComponent id="zh-accuracy-table" queryId="zh_summary" title="配对评测结果" kind="table" variant="card" sourceRows={summary} displayRows={summary}>
          <DataTable rows={summary} searchable={false} caption="中文与英文的准确率和正确数" columns={[{key:"model",label:"模型"},{key:"zhAccuracyRate",label:"中文准确率",renderCell:pct},{key:"zhCorrect",label:"中文正确数"},{key:"enAccuracyRate",label:"英文准确率",renderCell:pct},{key:"enCorrect",label:"英文正确数"}]}/>
          <DataTable rows={summary} searchable={false} caption="中英文 Macro F1" columns={[{key:"model",label:"模型"},{key:"zhMacroF1Rate",label:"中文 Macro F1",renderCell:pct},{key:"enMacroF1Rate",label:"英文 Macro F1",renderCell:pct},{key:"delta",label:"中文 − 英文",renderCell:pp}]}/>
          <p className="training-muted">每种语言条件均为相同的 502 条样本。英文译文由数据源提供；本机不调用翻译服务或 Jev。14 个候选的英文定义及任务指令保持一致。</p>
        </DataComponent>
      </Section>
      {audit.length>0&&<DataComponent id="zh-language-audit" queryId="zh_audit" title="含汉字题目的补充检查" kind="table" variant="card" sourceRows={audit} displayRows={audit}>
        <p className="training-muted" data-reviewed-rows>来源的 zh-CN 子集中混有 {audit[0].excluded} 条英文原文。仅对其余 {audit[0].records} 条含汉字文本及配对英文译文计分，结果如下；上方保留全部 502 条的原协议结果。</p>
        <DataTable rows={audit} searchable={false} caption="492条含汉字文本的补充结果" columns={[{key:"model",label:"模型"},{key:"records",label:"样本"},{key:"zhAccuracyRate",label:"含汉字原文",renderCell:pct},{key:"enAccuracyRate",label:"对应英文译文",renderCell:pct}]}/>
      </DataComponent>}
      <Section id="zh-performance-section" title="本机推理速度" spacing="content">
        <DataComponent id="zh-performance-table" queryId="zh_performance" title="MPS 实测吞吐" kind="table" variant="card" sourceRows={performance} displayRows={performance}>
          <DataTable rows={performance} searchable={false} caption="本机 MPS 推理速度" columns={[{key:"model",label:"模型"},{key:"language",label:"输入"},{key:"examplesPerSecond",label:"条 / 秒",renderCell:fixed},{key:"seconds",label:"502 条耗时 / 秒",renderCell:fixed},{key:"medianTokens",label:"中位 tokens"},{key:"maxTokens",label:"最大 tokens"}]}/>
          <p className="training-muted">吞吐包含分词、组批、MPS 前向和结果回传，不含模型加载；上下文上限仍为 8k，本轮没有截断输入。批量吞吐不等同于单请求延迟。</p>
        </DataComponent>
      </Section>
    </>}
    <Section id="zh-cases-section" title="中文题目与本地模型回答" spacing="content">
      <DataComponent id="zh-case-table" queryId="zh_cases" title="逐题查看" kind="table" variant="card" sourceRows={filtered} displayRows={filtered}>
        <p className="training-muted" data-reviewed-rows>当前 {filtered.length} / {run.records} 题{run.complete?` · 原始 Laya ${filtered.filter(r=>r.baselineCorrect).length} 题正确 · 微调 Laya ${filtered.filter(r=>r.tunedCorrect).length} 题正确`:""}</p>
        <DataTable rows={filtered} searchable={false} caption="MInDS14 逐题本地预测" rowKey="id" onRowSelect={setSelected} rowActionLabel={row=>`查看中文评测第 ${row.number} 题`}
          toolbarControls={<><input className="jev-case-search" type="search" aria-label="搜索中文测试题" placeholder="搜索中文、英文或题目 ID" value={search} onChange={e=>setSearch(e.target.value)}/><Dropdown label="输入语言" showLabel value={language} onChange={setLanguage} choices={["zh","en"]} choiceLabels={{zh:"中文子集原文",en:"英文译文"}}/><Dropdown label="样本范围" showLabel value={textScope} onChange={setTextScope} choices={["all","han","noHan"]} choiceLabels={{han:"含汉字原文",noHan:"混入的英文原文"}}/><Dropdown label="标准类别" showLabel value={intent} onChange={setIntent} choices={choices} choiceLabels={names}/><Dropdown label="微调结果" showLabel value={outcome} onChange={setOutcome} choices={["all","wrong","correct"]} choiceLabels={{wrong:"答错",correct:"答对"}}/>{(search||intent!=="all"||outcome!=="all"||language!=="zh"||textScope!=="all")&&<Button onClick={()=>{setSearch("");setIntent("all");setOutcome("all");setLanguage("zh");setTextScope("all")}}>重置中文筛选</Button>}</>}
          columns={[{key:"number",label:"题号"},{key:"text",label:"测试原文",renderCell:v=><span className="jev-case-text">{v}</span>},{key:"expectedZh",label:"标准意图"},{key:"baselineAnswerZh",label:"原始 Laya",renderCell:(_v,row)=>answer("baseline",row)},{key:"tunedAnswerZh",label:"微调 Laya",renderCell:(_v,row)=>answer("tuned",row)}]}/>
        {!filtered.length&&<p className="training-muted" role="status">没有符合筛选条件的题目。</p>}
        <p className="training-muted">点击一题查看中文原文、英文译文及两种输入下的预测。显示的概率未针对本数据校准。</p>
      </DataComponent>
    </Section>
    {run.complete&&<Section id="zh-class-section" title="各业务意图的表现" spacing="content"><DataComponent id="zh-class-table" queryId="zh_classes" title="14 类召回率" kind="table" variant="card" sourceRows={classes} displayRows={classes}>
      <DataTable rows={classes} caption="MInDS14 各类召回率" columns={[{key:"intentZh",label:"意图"},{key:"samples",label:"样本"},{key:"baselineZhRate",label:"原始中文",renderCell:pct},{key:"tunedZhRate",label:"微调中文",renderCell:pct},{key:"tunedEnRate",label:"微调英文",renderCell:pct}]}/>
    </DataComponent></Section>}
    <Dialog open={Boolean(selected)} onClose={()=>setSelected(null)} title={selected?`第 ${selected.number} 题 · 中英文配对`:"题目详情"} expanded>
      {selected&&<DataComponent id="zh-case-detail" queryId="zh_cases" title="同一条来源记录" kind="table" sourceRows={cases.filter(r=>r.id===selected.id)} displayRows={cases.filter(r=>r.id===selected.id)}>
        <div data-reviewed-rows><p className="training-muted">{selected.id} · 标准意图：{selected.expectedZh} / {selected.expected}</p><blockquote className="jev-case-message">{selected.textZh}</blockquote><p className="training-callout">数据源英文译文：{selected.textEn}</p></div>
        <DataTable rows={cases.filter(r=>r.id===selected.id)} searchable={false} caption="同一道题的两种语言预测" columns={[{key:"language",label:"语言",renderCell:v=>v==="zh"?"中文原文":"英文译文"},{key:"baselineAnswerZh",label:"原始 Laya",renderCell:(_v,row)=>answer("baseline",row)},{key:"tunedAnswerZh",label:"微调 Laya",renderCell:(_v,row)=>answer("tuned",row)}]}/>
      </DataComponent>}
    </Dialog>
  </div>;
}
