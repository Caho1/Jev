import React from "react";
import {DataComponent, DataTable, Section, Dropdown, Button, Dialog, useDataApp} from "../../data-app-public.jsx";

const models = [{key:"baseline",label:"原始 Laya"},{key:"tuned",label:"微调 Laya"},{key:"jev",label:"Jev 1.13.0"}];
const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : "—";
const outcomes = {
  all:"全部结果", disagreement:"模型答案有分歧", anyWrong:"至少一个模型答错",
  baselineWrong:"原始 Laya 答错", tunedWrong:"微调 Laya 答错", jevWrong:"Jev 答错或无效",
  tunedOnly:"微调 Laya 对 / Jev 错", jevOnly:"Jev 对 / 微调 Laya 错",
  allCorrect:"三个模型都答对", jevInvalid:"Jev 校验失败"
};

function matchesOutcome(row, value) {
  switch(value) {
    case "disagreement": return row.disagreement;
    case "anyWrong": return !row.baselineCorrect || !row.tunedCorrect || !row.jevCorrect;
    case "baselineWrong": return !row.baselineCorrect;
    case "tunedWrong": return !row.tunedCorrect;
    case "jevWrong": return !row.jevCorrect;
    case "tunedOnly": return row.tunedCorrect && !row.jevCorrect;
    case "jevOnly": return row.jevCorrect && !row.tunedCorrect;
    case "allCorrect": return row.baselineCorrect && row.tunedCorrect && row.jevCorrect;
    case "jevInvalid": return !row.jevValid;
    default: return true;
  }
}

function Prediction({row, model}) {
  const invalid = model === "jev" && !row.jevValid;
  const correct = row[`${model}Correct`];
  return <div className="jev-case-answer">
    <span className="jev-case-label">{invalid ? "回答未通过校验" : row[`${model}Answer`]}</span>
    <span className={`jev-case-verdict ${correct ? "correct" : "incorrect"}`}>
      {invalid ? "! 计为错误" : correct ? "✓ 正确" : "× 错误"}
      {!invalid && <span className="jev-case-probability">{pct(row[`${model}Confidence`])}</span>}
    </span>
  </div>;
}

export function JevCases() {
  const {reviewedRows} = useDataApp();
  const rows = reviewedRows("jev_cases");
  const [intent,setIntent] = React.useState("all");
  const [outcome,setOutcome] = React.useState("all");
  const [search,setSearch] = React.useState("");
  const [selected,setSelected] = React.useState(null);
  const choices = React.useMemo(()=>["all",...new Set(rows.map(row=>row.expected))].sort((a,b)=>a==="all"?-1:b==="all"?1:a.localeCompare(b)),[rows]);
  const filtered = React.useMemo(()=>{
    const needle=search.trim().toLocaleLowerCase();
    return rows.filter(row=>(intent==="all" || row.expected===intent) && matchesOutcome(row,outcome) &&
      (!needle || [row.id,row.text,row.expected,row.baselineAnswer,row.tunedAnswer,row.jevAnswer].some(value=>String(value??"").toLocaleLowerCase().includes(needle))));
  },[rows,intent,outcome,search]);
  const controls = <>
    <input className="jev-case-search" type="search" aria-label="搜索测试题目" placeholder="搜索原文、题目 ID 或预测类别" value={search} onChange={event=>setSearch(event.target.value)}/>
    <Dropdown label="标准意图" showLabel value={intent} choices={choices} choiceLabels={{all:"全部意图"}} onChange={setIntent}/>
    <Dropdown label="答题结果" showLabel value={outcome} choices={Object.keys(outcomes)} choiceLabels={outcomes} onChange={setOutcome}/>
    {(search || intent!=="all" || outcome!=="all") && <Button onClick={()=>{setSearch("");setIntent("all");setOutcome("all")}}>清除筛选</Button>}
  </>;
  const columns = [
    {key:"number",label:"题号"},
    {key:"text",label:"测试题目（原文）",renderCell:value=><span className="jev-case-text">{value}</span>},
    {key:"expected",label:"标准标签",renderCell:value=><span className="jev-case-gold">{value}</span>},
    ...models.map(model=>({key:`${model.key}Answer`,label:model.label,renderCell:(_value,row)=><Prediction row={row} model={model.key}/>}))
  ];
  return <Section id="jev-cases-section" title="测试题目与三模型回答" spacing="content">
    <DataComponent id="jev-case-table" queryId="jev_cases" title="逐题对照" kind="table" variant="card" sourceRows={filtered} displayRows={filtered}
      description="完整官方测试集，按原始顺序排列；题号从 1 开始，测试 ID 从 00000 开始。分类模型的回答是意图标签。点击一行查看原文、前三候选和失败原因。">
      <p className="training-muted" data-reviewed-rows>当前 {filtered.length.toLocaleString()} / {rows.length.toLocaleString()} 题 · {models.map(model=>`${model.label} ${filtered.filter(row=>row[`${model.key}Correct`]).length.toLocaleString()} 题正确`).join(" · ")}</p>
      <div className="jev-case-table">
        <DataTable key={`${intent}|${outcome}`} rows={filtered} columns={columns} caption="官方测试题与三个模型的逐题回答" searchable={false} toolbarControls={controls}
          rowKey="id" selectedRowKey={selected?.id} onRowSelect={setSelected} rowActionLabel={row=>`查看第 ${row.number} 题详情`}/>
      </div>
      {!filtered.length && <p className="training-muted" role="status">没有符合这些条件的测试题，请调整或清除筛选。</p>}
      <p className="training-muted">数字为预测类别的概率：Laya 已做温度校准，Jev 为 API 原始值，不能直接视为相同的正确率保证。点击题目查看前三候选。</p>
    </DataComponent>
    <Dialog open={Boolean(selected)} onClose={()=>setSelected(null)} title={selected?`第 ${selected.number} 题 · 三模型回答`:"测试题详情"} expanded>
      {selected && <DataComponent id="jev-case-detail" queryId="jev_cases" title="测试原文与预测详情" kind="table" displayRows={[selected]} sourceRows={[selected]}>
        <div className="jev-case-detail" data-reviewed-rows>
          <p className="training-muted">{selected.id} · BANKING77 官方 test.csv</p>
          <blockquote className="jev-case-message">{selected.text}</blockquote>
          <p className="jev-case-expected">标准标签：<strong>{selected.expected}</strong></p>
          <div className="jev-case-models">
            {models.map(model=><section key={model.key} className="jev-case-model">
              <h3>{model.label}</h3><Prediction row={selected} model={model.key}/>
              {model.key==="jev" && !selected.jevValid
                ? <p className="training-muted">{selected.jevError}。该次原始回答未保存，无法展示预测类别或候选概率；按冻结协议计为错误。</p>
                : <><p className="training-muted">概率最高的三个候选</p><ol className="jev-case-top3">{JSON.parse(selected[`${model.key}Top3`]).map(item=><li key={item.label}><span>{item.label}</span><strong>{pct(item.probability)}</strong></li>)}</ol></>}
            </section>)}
          </div>
          <p className="training-muted">三模型使用相同原文、任务指令及 77 个候选类别，标准标签仅用于本地评分。Jev 的最终回答以 API choice 为准；舍入后的概率可能出现并列。</p>
        </div>
      </DataComponent>}
    </Dialog>
  </Section>;
}
