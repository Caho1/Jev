import React from "react";
import {DataComponent, EvidenceChart, DataTable, Section, Dialog, Button, Switch, useDataApp} from "../../data-app-public.jsx";

const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : "待评测";
const number = value => Number.isFinite(value) ? value.toLocaleString("zh-CN") : "—";
const phases = {loading:"加载模型",baseline:"原始模型评测",baseline_complete:"原始模型评测完成",training:"本地微调中",gradient_verified:"梯度检查通过",tuned:"微调模型评测",complete:"本轮试验完成",failed:"运行失败"};

export function CustomerIntentTraining(){
  const {reviewedRows,snapshot,mode,chartExportActive} = useDataApp();
  const overview=reviewedRows("czi_run"),run=overview[0]??{},splits=reviewedRows("czi_splits"),curves=reviewedRows("czi_training"),scores=reviewedRows("czi_scores"),cases=reviewedRows("czi_cases");
  const [search,setSearch]=React.useState(""),[wrongOnly,setWrongOnly]=React.useState(false),[selected,setSelected]=React.useState(null),[auto,setAuto]=React.useState(true),[refreshError,setRefreshError]=React.useState(false);
  const finished=["complete","failed"].includes(run.phase);
  React.useEffect(()=>{
    if(finished||!auto||mode!=="view"||chartExportActive) return;
    let cancelled=false;
    const timer=setInterval(async()=>{
      try{const response=await fetch("/api/version",{cache:"no-store"});if(!response.ok)throw Error("unavailable");const version=await response.json();if(cancelled)return;setRefreshError(false);if(version.generatedAt&&version.generatedAt!==snapshot.generatedAt)window.location.reload();}
      catch{if(!cancelled)setRefreshError(true);}
    },15000);
    return()=>{cancelled=true;clearInterval(timer);};
  },[finished,auto,mode,chartExportActive,snapshot.generatedAt]);
  const filtered=cases.filter(r=>(!wrongOnly||r.tunedCorrect===false)&&[r.text,r.gold,r.id].some(v=>v.includes(search.trim())));
  const answer=(variant,row)=><span>{row[variant+"Answer"]??"待评测"}{row[variant+"Correct"]===true?" ✓":row[variant+"Correct"]===false?" ×":""}</span>;
  return <div>
    <DataComponent id="czi-status" queryId="czi_run" title={phases[run.phase]??"中文多场景客服"} kind="metric" variant="card" sourceRows={overview} displayRows={overview}>
      <div className="jev-progress-row" data-reviewed-rows><strong>{number(run.completed)} / {number(run.total)} 条 · {number(run.update)} / {number(run.updates)} 次更新</strong><span>MPS · FP32 · LoRA · 35 类</span></div>
      <div className="training-progress" data-reviewed-rows><div style={{width:`${Math.min(100,100*(run.progressRate??0))}%`}}/></div>
      <p className="training-muted" data-reviewed-rows>全量训练数据 {number(run.preparedTrain)} 条；本轮固定抽样 {number(run.pilotTrain)} 条训练、{number(run.devProbe)} 条开发评测。仅验证训练管线，独立测试仍封存。</p>
      {!finished&&<div className="jev-progress-row"><span data-reviewed-rows>{Number.isFinite(run.examplesPerSecond)?`${run.examplesPerSecond.toFixed(2)} 条/秒 · 预计剩余训练 ${Math.ceil(run.remainingSeconds/60)} 分钟` : "正在测量吞吐"}</span><Switch label="同步训练快照" checked={auto} onChange={setAuto}/></div>}
      <p className="training-muted" data-reviewed-rows>快照时间：{run.eventTime?new Date(run.eventTime).toLocaleTimeString("zh-CN",{hour12:false}):"—"}{refreshError?" · 同步暂不可用，显示最近快照":""}</p>
      {run.error&&<p role="alert" className="training-error">{run.error}</p>}
    </DataComponent>
    <Section id="czi-experiment" title="开发集诊断" columns={2}>
      <EvidenceChart id="czi-loss" queryId="czi_training" title="累计训练损失" variant="card" rows={curves} sourceRows={curves} height={260} spec={{type:"line",x:"step",y:"meanLoss",startAtZero:true,showLegend:false,valueDecimals:3}} description="交叉熵；用于观察优化是否正常，不代表泛化准确率。"/>
      <EvidenceChart id="czi-accuracy" queryId="czi_scores" title="同一开发样本：底座与微调" variant="card" rows={scores} sourceRows={scores} height={260} spec={{type:"bar",x:"model",y:"accuracyRate",fields:["accuracyRate","macroF1Rate"],stackable:false,startAtZero:true,showLegend:true,valueDecimals:2,legend:{labels:{accuracyRate:"准确率",macroF1Rate:"Macro F1"}}}} description="103 条固定开发探针。官方多语言 Laya 未在本批数据上微调，各底座使用自身 tokenizer。不是正式测试或 Jev 对比。"/>
    </Section>
    <Section id="czi-data" title="中文数据准备" spacing="content">
      <DataComponent id="czi-split-table" queryId="czi_splits" title="按会话隔离的单诉求数据" kind="table" variant="card" sourceRows={splits} displayRows={splits}>
        <DataTable rows={splits} searchable={false} caption="CrossWOZ 数据划分" columns={[{key:"split",label:"划分"},{key:"records",label:"样本",renderCell:number},{key:"dialogues",label:"会话",renderCell:number},{key:"medianTokens",label:"中位 tokens",renderCell:number},{key:"maxTokens",label:"最长 tokens",renderCell:number}]}/>
      </DataComponent>
      <DataComponent id="czi-data-scope" queryId="czi_run" title="当前覆盖范围" kind="table" variant="card" sourceRows={overview} displayRows={overview}>
        <p className="training-muted" data-reviewed-rows>酒店、餐馆、景点、地铁与出租。另保留 {number(run.multiple)} 条多诉求样本；{number(run.ecommerceReview)} 条电商对话正在复核，已批准进入本轮训练 {number(run.ecommerceApproved)} 条。</p>
        <p className="training-muted" data-reviewed-rows>上下文预算 8,192 tokens，本轮数据没有截断。CrossWOZ 是原生中文人工任务对话，尚不能代表真实电商客服流量。</p>
      </DataComponent>
    </Section>
    <Section id="czi-cases" title="开发题目与模型回答" spacing="content">
      <DataComponent id="czi-cases-table" queryId="czi_cases" title="逐题检查" kind="table" variant="card" sourceRows={filtered} displayRows={filtered}>
        <DataTable rows={filtered} searchable={false} caption="固定开发探针，非测试题" rowKey="id" onRowSelect={setSelected} rowActionLabel={r=>`查看中文客服第 ${r.number} 题`}
          toolbarControls={<><input className="jev-case-search" aria-label="搜索中文客服题目" type="search" placeholder="搜索问题或意图" value={search} onChange={e=>setSearch(e.target.value)}/><Switch label="只看微调答错" checked={wrongOnly} onChange={setWrongOnly}/>{(search||wrongOnly)&&<Button onClick={()=>{setSearch("");setWrongOnly(false);}}>重置筛选</Button>}</>}
          columns={[{key:"number",label:"题号"},{key:"text",label:"当前用户",renderCell:v=><span className="jev-case-text">{v}</span>},{key:"gold",label:"标准请求"},{key:"baselineAnswer",label:"原始 Laya",renderCell:(_v,r)=>answer("baseline",r)},{key:"tunedAnswer",label:"中文试验 LoRA",renderCell:(_v,r)=>answer("tuned",r)},{key:"multilingualAnswer",label:"官方多语言 Laya",renderCell:(_v,r)=>answer("multilingual",r)}]}/>
        <p className="training-muted" data-reviewed-rows>显示 {filtered.length} / {cases.length} 题。点击题目查看模型实际接收的对话历史。</p>
      </DataComponent>
    </Section>
    <Dialog open={Boolean(selected)} onClose={()=>setSelected(null)} title="中文客服输入与判断" expanded>
      {selected&&<DataComponent id="czi-case-detail" queryId="czi_cases" title={`第 ${selected.number} 题`} kind="table" sourceRows={[selected]} displayRows={[selected]}>
        <div data-reviewed-rows><p className="training-muted">{selected.id} · 原始 tokenizer：{selected.inputTokens} tokens · 多语言 tokenizer：{selected.multilingualTokens??"待评测"} · 标准请求：{selected.gold}</p><pre style={{whiteSpace:"pre-wrap",overflowWrap:"anywhere",fontFamily:"inherit"}}>{selected.state}</pre><p>原始 Laya：{answer("baseline",selected)}</p><p>中文试验 LoRA：{answer("tuned",selected)}</p><p>官方多语言 Laya（未在本批数据微调）：{answer("multilingual",selected)}</p></div>
      </DataComponent>}
    </Dialog>
  </div>;
}
