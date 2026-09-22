import React from "react";
import {DataComponent, EvidenceChart, DataTable, Section, useDataApp} from "../../data-app-public.jsx";
import {JevCases} from "./JevCases.jsx";

const pct=v=>Number.isFinite(v)?`${(v*100).toFixed(2)}%`:"—";
const pp=v=>Number.isFinite(v)?`${v>0?"+":""}${(v*100).toFixed(2)} pp`:"—";

export function JevComparison(){
  const {reviewedRows,snapshot,mode,chartExportActive}=useDataApp();
  const progress=reviewedRows("jev_progress"), state=progress[0]??{};
  const accuracy=reviewedRows("jev_accuracy"), comparison=reviewedRows("jev_pair"), classes=reviewedRows("jev_classes");
  const audit=reviewedRows("jev_label_audit");
  const pair=comparison[0];
  const [stale,setStale]=React.useState(false);
  React.useEffect(()=>{
    if(state.phase!=="running")return;
    let active=true;
    const timer=setInterval(async()=>{
      try{
        const response=await fetch("/api/version",{cache:"no-store"});
        if(!response.ok)throw new Error("unavailable");
        const value=await response.json();
        if(!active)return;
        setStale(Date.now()-new Date(state.updatedAt).valueOf()>90000);
        if(mode==="view"&&!chartExportActive&&value.generatedAt&&value.generatedAt!==snapshot.generatedAt)window.location.reload();
      }catch{if(active)setStale(true)}
    },10000);
    return()=>{active=false;clearInterval(timer)};
  },[state.phase,state.updatedAt,snapshot.generatedAt,mode,chartExportActive]);
  const complete=state.complete;
  const title=complete?"Jev 对照已完成":state.phase==="stopped_on_error"?"API 调用中止，已保留结果":"正在评测 Jev";
  return <div>
    <DataComponent id="jev-progress" queryId="jev_progress" title={title} kind="metric" variant="card" displayRows={progress} sourceRows={progress}>
      <div className="jev-progress-row" data-reviewed-rows><strong>{state.completed?.toLocaleString()??0} / {state.total?.toLocaleString()??"3,080"}</strong><span>有效回答 {state.success??0} · 无效/失败 {state.errors??0}</span></div>
      <div className="training-progress"><div style={{width:`${100*(state.progressRate??0)}%`}}/></div>
      <p className="training-muted" data-reviewed-rows>{stale?"连接延迟，显示最近快照。":complete?"同一测试集 · 同一原文、指令与 77 个类别 · 微调模型保持冻结":"每 15 秒同步进度；全部请求完成后显示最终准确率。"}</p>
    </DataComponent>
    {complete&&<>
      <Section id="jev-accuracy-section" title="同一测试集的准确性对比" columns={2}>
        <EvidenceChart id="jev-accuracy-chart" queryId="jev_accuracy" title="准确率与 Macro F1" rows={accuracy} sourceRows={accuracy} variant="card" height={290}
          spec={{type:"bar",x:"model",y:"accuracyRate",fields:["accuracyRate","macroF1Rate"],showLegend:true,startAtZero:true,stackable:false,valueDecimals:2,legend:{labels:{accuracyRate:"准确率",macroF1Rate:"Macro F1"}}}}/>
        <DataComponent id="jev-accuracy-table" queryId="jev_accuracy" title="完整测试结果" kind="table" variant="card" displayRows={accuracy} sourceRows={accuracy}>
          <DataTable rows={accuracy} searchable={false} caption="Jev 与 Laya 准确性对照" columns={[{key:"model",label:"模型"},{key:"accuracyRate",label:"准确率",renderCell:pct},{key:"macroF1Rate",label:"Macro F1",renderCell:pct},{key:"correct",label:"正确数"},{key:"failures",label:"无效/失败"}]}/>
          <p className="training-muted">Jev 为开箱 API；微调 Laya 已使用 BANKING77 训练集。两者使用同一 3,080 条独立测试集。</p>
          <p className="training-muted" data-reviewed-rows>Jev 有 {state.errors} 条回答未通过预定校验并计为错误。在相同的 {accuracy.find(r=>r.model==="Jev 1.13.0")?.validSubsetRecords?.toLocaleString()} 条有效回答子集上，Jev 为 {pct(accuracy.find(r=>r.model==="Jev 1.13.0")?.validSubsetAccuracyRate)}，微调 Laya 为 {pct(accuracy.find(r=>r.model==="微调 Laya")?.validSubsetAccuracyRate)}。</p>
        </DataComponent>
      </Section>
      <JevCases/>
      <Section id="jev-pair-section" title="逐条配对差异" spacing="content">
        <DataComponent id="jev-paired" queryId="jev_pair" title="Jev − 微调 Laya" kind="metric" variant="card" displayRows={comparison} sourceRows={comparison}>
          <div className="jev-progress-row" data-reviewed-rows><strong>{pp(pair?.delta)}</strong><span>95% 区间 {pp(pair?.lower)} 至 {pp(pair?.upper)}</span></div>
          <p className="training-muted">pp 为百分点；正值表示 Jev 更高。区间按重复组配对 bootstrap 2,000 次计算。</p>
          <DataTable rows={comparison} searchable={false} caption="逐条预测的共同与分歧" columns={[{key:"bothCorrect",label:"两者都正确"},{key:"jevOnly",label:"仅 Jev 正确"},{key:"layaOnly",label:"仅微调 Laya 正确"},{key:"bothWrong",label:"两者都错误"}]}/>
        </DataComponent>
      </Section>
      <Section id="jev-class-section" title="各意图的差距" spacing="content">
        <DataComponent id="jev-label-audit" queryId="jev_label_audit" title="类别名称不足以定义业务意图" kind="table" variant="card" displayRows={audit} sourceRows={audit}>
          <p className="training-callout" data-reviewed-rows>本轮只给模型原始类别名。数据源中 <code>get_physical_card</code> 的 {audit[0]?.samples} 条测试题实际都提到 PIN，例如“{audit[0]?.example}”。</p>
          <p className="training-muted">微调 Laya 学过这些标签的实际含义，Jev 仅根据类别名判断。这个差异影响结果解释；下一轮需要用训练数据制定明确的类别定义，重新冻结比较协议。</p>
        </DataComponent>
        <DataComponent id="jev-class-table" queryId="jev_classes" title="77 类召回对照" kind="table" variant="card" sourceRows={classes} displayRows={[...classes].sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta))} description="按召回率差的绝对值从大到小排列；可搜索意图，或点击表头排序。">
          <DataTable rows={[...classes].sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta))} caption="每类召回率的 Jev 与 Laya 对比" columns={[{key:"intent",label:"意图"},{key:"samples",label:"样本数"},{key:"layaRecallRate",label:"微调 Laya",renderCell:pct},{key:"jevRecallRate",label:"Jev",renderCell:pct},{key:"delta",label:"Jev − Laya",renderCell:pp}]}/>
        </DataComponent>
      </Section>
    </>}
    <p className="training-muted training-limit">这是英文银行短句的公开基准；Jev 是否曾在上游训练见过这些数据未知。结果不等同于真实线上流量表现。Jev 输出仅用于本次评测。</p>
  </div>;
}
