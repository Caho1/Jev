import React from "react";
import { DataComponent, EvidenceChart, MetricCard, DataTable, Section, SectionHeader,
  SortableRegion, SortableItem, SegmentedControl, Switch, Button, useDataApp, useDashboardTabs
} from "../../data-app-public.jsx";
import "./training.css";
import {JevComparison} from "./JevComparison.jsx";
import {ChineseEvaluation} from "./ChineseEvaluation.jsx";
import {CustomerIntentTraining} from "./CustomerIntentTraining.jsx";

const phases = { loading:"加载模型", baseline_development:"评测原始模型", training:"正在训练", development:"开发集评测", epoch_complete:"本轮完成", checkpoint_reload:"校验最佳模型", calibration:"校准概率", final_test:"最终测试", complete:"训练与评测完成", failed:"运行失败" };
const pct = v => Number.isFinite(v) ? `${(v*100).toFixed(2)}%` : "待评测";
const dec = v => Number.isFinite(v) ? v.toFixed(3) : "—";
const num = v => Number.isFinite(v) ? v.toLocaleString("zh-CN") : "—";
const duration = v => Number.isFinite(v) ? `${Math.floor(v/60)} 分 ${Math.round(v%60)} 秒` : "估算中";
const clock = v => v ? new Date(v).toLocaleTimeString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false}) : "—";
const chartBase = { showXAxisLabel:false, showYAxisLabel:false, startAtZero:true, stackable:false, showLegend:true, valueDecimals:3 };

function LiveStatus({run}) {
  const {snapshot,mode,chartExportActive} = useDataApp();
  const [auto,setAuto] = React.useState(true);
  const [refreshError,setRefreshError] = React.useState(false);
  const [tick,setTick] = React.useState(Date.now());
  const finished = ["complete","failed"].includes(run.phase);
  React.useEffect(()=>{
    let cancelled=false;
    async function poll(){
      setTick(Date.now());
      if(finished) return;
      try {
        const response=await fetch("/api/version",{cache:"no-store"});
        if(!response.ok) throw new Error("unavailable");
        const version=await response.json();
        if(cancelled) return;
        setRefreshError(false);
        // 编辑和导出期间保留当前页面，避免丢失未保存的呈现修改。
        if(auto && mode==="view" && !chartExportActive && version.generatedAt && version.generatedAt!==snapshot.generatedAt) window.location.reload();
      } catch { if(!cancelled) setRefreshError(true); }
    }
    const interval=setInterval(poll,10000);
    return ()=>{cancelled=true;clearInterval(interval)};
  },[auto,mode,chartExportActive,snapshot.generatedAt,finished]);
  const stale=!finished && (refreshError || tick-new Date(run.syncTime).valueOf()>90000);
  return <div className="training-live" data-reviewed-rows>
    <div><span className={`training-dot ${stale||run.phase==="failed"?"warning":""}`} />
      <strong>{phases[run.phase]??"等待日志"}</strong>
      <span className="training-muted">{stale?"连接延迟 · 显示最近快照":`日志 ${clock(run.eventTime)} · 同步 ${clock(run.syncTime)}`}</span>
    </div>
    {!finished && <Switch label="自动更新（20 秒）" checked={auto} onChange={setAuto} size="compact"/>}
  </div>;
}

function Empty({children}) {return <div className="training-empty">{children}</div>}
function Metric({id,title,value,description,rows,children}) {
  return <SortableItem id={id} label={title} kind="metric"><MetricCard id={id} queryId="run" title={title} value={value} description={description} displayRows={rows} sourceRows={rows}>{children}</MetricCard></SortableItem>
}

export function DashboardContent(){
  useDashboardTabs([{id:"progress",label:"训练进度"},{id:"evaluation",label:"最终效果"},{id:"jev",label:"Jev 对比"},{id:"data",label:"数据与配置"},{id:"chinese",label:"中文本地评测"},{id:"customer-zh",label:"中文客服训练"}]);
  const {reviewedRows,activeTabId} = useDataApp();
  const runRows=reviewedRows("run"), run=runRows[0]??{};
  const training=reviewedRows("training"), development=reviewedRows("development"), test=reviewedRows("test"), splits=reviewedRows("splits");
  const classes=reviewedRows("classes"), selective=reviewedRows("selective");
  const [calibration,setCalibration]=React.useState("温度校准");
  const [threshold,setThreshold]=React.useState("0.9");
  const filteredTest=test.filter(r=>r.calibration===calibration);
  const selected=selective.filter(r=>r.threshold===Number(threshold));
  const devBest=development.reduce((best,r)=>!best||r["开发 NLL"]<best["开发 NLL"]?r:best,null);
  const done=run.phase==="complete";
  const evaluationPhases=["checkpoint_reload","calibration","final_test","complete"];
  const progressRows=training.map(r=>({...r,轮次:`第 ${r.epoch} 轮`}));
  const delta=run.accuracyDelta;
  return <article className="page training-page">
    <h1 className="training-visually-hidden">Laya 客服意图训练与效果</h1>
    {!["jev","chinese","customer-zh"].includes(activeTabId) && <LiveStatus run={run}/>}
    {activeTabId==="jev" && <JevComparison/>}
    {activeTabId==="chinese" && <ChineseEvaluation/>}
    {activeTabId==="customer-zh" && <CustomerIntentTraining/>}
    {run.error && <div role="alert" className="training-error">{run.error}</div>}
    {activeTabId==="progress" && <>
      <SortableRegion id="banking77:metrics" label="训练进度指标" variant="freeform" authoredRevision={1} className="training-metrics">
        <Metric id="run-progress" title="训练更新进度" value={pct(run.progressRate)} rows={runRows} description="优化器更新进度。训练完成后还需模型校验、校准与最终测试。">
          <div className="training-progress"><div style={{width:`${Math.min(100,100*(run.progressRate??0))}%`}}/></div>
          <p className="training-muted">{num(run.update)} / {num(run.totalUpdates)} 步 · 第 {run.epoch} / {run.epochs} 轮</p>
        </Metric>
        <Metric id="run-speed" title="训练吞吐" value={Number.isFinite(run.examplesPerSecond)?`${run.examplesPerSecond.toFixed(1)} 条/秒`:"等待采样"} rows={runRows} description="最近一个 epoch 的累计平均吞吐；不是推理速度。">
          <p className="training-muted">RTX 5090 · BF16 · 批量 {run.batchSize} × 累积 {run.accumulation}</p>
        </Metric>
        <Metric id="run-remaining" title={done?"全流程耗时":"预计剩余训练"} value={done?duration(run.elapsedSeconds):evaluationPhases.includes(run.phase)?"训练已完成":duration(run.remainingTrainingSeconds)} rows={runRows} description="估计仅含剩余训练，不含开发集评测、校准及测试。">
          <p className="training-muted">{done?`已选择第 ${run.selectedEpoch} 轮模型`:"按实测吞吐估算 · 不含评测"}</p>
        </Metric>
      </SortableRegion>
      <Section id="training-curves-section" title="训练过程" columns={2} spacing="after-metrics">
        <EvidenceChart id="training-loss" queryId="training" title="训练损失" rows={progressRows} sourceRows={training} variant="card" height={275}
          description="每个点为该轮截至该步的平均 NLL，越低越好。每轮重置累计，三轮不连线。横轴为优化器更新步数。"
          spec={{...chartBase,type:"line",x:"step",y:"训练 NLL",series:"轮次",colors:{"第 1 轮":"var(--chart-1)","第 2 轮":"var(--chart-2)","第 3 轮":"var(--chart-3)"}}}/>
        <EvidenceChart id="development-quality" queryId="development" title="开发集分类效果" rows={development} sourceRows={development} variant="card" height={275}
          description="固定 998 条开发集，77 个类别；第 0 轮为原始 Laya。Macro F1 是各类别 F1 的等权平均。"
          spec={{...chartBase,type:"line",x:"epoch",y:"accuracyRate",fields:["accuracyRate","macroF1Rate"],legend:{labels:{accuracyRate:"准确率",macroF1Rate:"Macro F1"}},valueDecimals:2}}/>
      </Section>
      <Section id="development-section" title="每轮评测与模型选择" spacing="content">
        <DataComponent id="development-table" queryId="development" title="开发集结果" kind="table" variant="card" displayRows={development} sourceRows={development} description="模型选择只比较开发集 NLL，最小者胜出；原始模型也参与比较。">
          {development.length?<><p className="training-callout" data-reviewed-rows>当前最佳：{devBest?.stage} · 准确率 {pct(devBest?.accuracyRate)} · NLL {dec(devBest?.["开发 NLL"])}</p>
          <DataTable rows={development} searchable={false} columns={[{key:"stage",label:"模型"},{key:"accuracyRate",label:"准确率",renderCell:pct},{key:"macroF1Rate",label:"Macro F1",renderCell:pct},{key:"开发 NLL",label:"开发 NLL ↓",renderCell:dec},{key:"samples",label:"样本数"}]} caption="每轮开发集评测"/></>:<Empty>正在评测原始模型，暂时没有完整评测结果。</Empty>}
        </DataComponent>
      </Section>
    </>}
    {activeTabId==="evaluation" && <>
      {!test.length ? <DataComponent id="test-pending" queryId="test" title="最终测试尚未解锁" kind="table" variant="card" displayRows={[]} sourceRows={[]}><Empty>完成 3 轮训练，按开发集选定模型并校准概率后，再评测独立的 3,080 条测试数据。</Empty></DataComponent> : <>
        <Section id="test-top" title="独立测试 · 3,080 条" spacing="none" columns={2}>
          <EvidenceChart id="test-accuracy" queryId="test" title="分类效果对比" variant="card" height={260} rows={filteredTest} sourceRows={filteredTest}
            spec={{...chartBase,type:"bar",x:"model",y:"accuracyRate",fields:["accuracyRate","macroF1Rate"],legend:{labels:{accuracyRate:"准确率",macroF1Rate:"Macro F1"}},valueDecimals:2}}
            description="原始与微调模型使用相同的全部候选输入和同一官方测试集。温度缩放不会改变分类结果。"/>
          <DataComponent id="test-probabilities" queryId="test" title="分类与概率指标" kind="table" variant="card" displayRows={filteredTest} sourceRows={filteredTest}
            headerControls={<SegmentedControl label="校准方式" value={calibration} onChange={setCalibration} options={[{value:"温度校准",label:"温度校准"},{value:"未校准",label:"未校准"}]}/>}
            description="NLL 衡量真实类别的概率，ECE 衡量信心与正确率的偏差；两者越低越好。温度由独立的 997 条校准集拟合。">
            <DataTable rows={filteredTest} searchable={false} columns={[{key:"model",label:"模型"},{key:"accuracyRate",label:"准确率",renderCell:pct},{key:"macroF1Rate",label:"Macro F1",renderCell:pct},{key:"nll",label:"NLL ↓",renderCell:dec},{key:"ece",label:"ECE ↓",renderCell:dec}]} caption="分类与概率校准效果"/>
            <p className="training-muted">NLL 与 ECE 越低越好。校准只改变概率，不改变预测类别。</p>
          </DataComponent>
        </Section>
        <p className="training-callout" data-reviewed-rows>已选择第 {run.selectedEpoch} 轮模型。{delta && <>准确率提升 {(delta.accuracy_difference*100).toFixed(2)} 个百分点；95% 区间 {delta.group_bootstrap_95_percent?.map(v=>(v*100).toFixed(2)).join(" 至 ")} 个百分点（按重复组 bootstrap）。</>}</p>
        <Section id="routing-section" title="置信度分流" spacing="content">
          <DataComponent id="routing-table" queryId="selective" title="高置信度样本的覆盖与准确率" kind="table" variant="card" displayRows={selected} sourceRows={selected}
            description="最高类别概率达到阈值才保留，其余可转人工。这只是固定阈值的描述性测试，尚不是上线承诺。"
            headerControls={<SegmentedControl label="置信度阈值" value={threshold} onChange={setThreshold} options={["0.5","0.8","0.9","0.95","0.99"].map(v=>({value:v,label:`≥ ${Number(v)*100}%`}))}/> }>
            <DataTable rows={selected} searchable={false} columns={[{key:"model",label:"模型"},{key:"count",label:"保留样本"},{key:"coverageRate",label:"覆盖率",renderCell:pct},{key:"accuracyRate",label:"保留样本准确率",renderCell:pct}]} caption="按阈值分流效果"/>
          </DataComponent>
        </Section>
        <Section id="class-section" title="薄弱类别" spacing="content">
          <DataComponent id="class-table" queryId="classes" title="77 类召回与 F1" kind="table" variant="card" displayRows={[...classes].sort((a,b)=>a.tunedRecallRate-b.tunedRecallRate)} sourceRows={classes} description="默认按微调模型召回率从低到高排列。可搜索意图名。">
            <DataTable rows={[...classes].sort((a,b)=>a.tunedRecallRate-b.tunedRecallRate)} columns={[{key:"intent",label:"意图"},{key:"samples",label:"样本"},{key:"baselineRecallRate",label:"原始召回",renderCell:pct},{key:"tunedRecallRate",label:"微调召回",renderCell:pct},{key:"f1Rate",label:"微调 F1",renderCell:pct}]} caption="测试集每类效果"/>
          </DataComponent>
        </Section>
      </>}
      <p className="training-muted training-limit">本轮验证英文银行短句的 77 类闭集分类。尚未验证中文、长上下文、未知意图或线上流量，也尚未与通用 LLM 做同条件比较。</p>
    </>}
    {activeTabId==="data" && <>
      <Section id="data-split-section" title="数据划分与用途" spacing="none">
        <DataComponent id="data-splits" queryId="splits" title="固定划分" kind="table" variant="card" displayRows={splits} sourceRows={splits}>
          <DataTable rows={splits} searchable={false} columns={[{key:"split",label:"划分"},{key:"count",label:"样本数"},{key:"groups",label:"去重组数"},{key:"purpose",label:"用途"},{key:"medianTokens",label:"中位 tokens"},{key:"maxTokens",label:"最大 tokens"}]} caption="数据划分"/>
          <p className="training-muted">排除 40 条与官方测试重叠的训练数据，以及 4 条标签冲突数据。各划分重复组交集为 0。</p>
        </DataComponent>
      </Section>
      <Section id="config-section" title="本次训练配置" spacing="content" columns={2}>
        <DataComponent id="run-config" queryId="run" title="模型与训练" kind="metric" variant="card" displayRows={runRows} sourceRows={runRows}>
          <dl className="training-facts" data-reviewed-rows><dt>方法</dt><dd>Laya · LoRA rank 16 + 决策头</dd><dt>上下文上限</dt><dd>{num(run.contextLimit)} tokens</dd><dt>批量 / 梯度累积</dt><dd>{run.batchSize} / {run.accumulation}</dd><dt>训练轮数</dt><dd>{run.epochs}</dd><dt>精度 / 硬件</dt><dd>BF16 / RTX 5090</dd><dt>运行 ID</dt><dd>{run.run}</dd></dl>
        </DataComponent>
        <DataComponent id="context-fit" queryId="splits" title="真实输入长度" kind="metric" variant="card" displayRows={splits} sourceRows={splits}>
          <div className="training-range" data-reviewed-rows><strong>575–667</strong><span>tokens · 本批训练输入</span></div>
          <p className="training-muted">包含完整的 77 个候选类别。按实际长度组批，没有填充到 8k，也没有截断候选。</p>
          <p className="training-muted">这次训练不能证明 8k 长文本能力；还需要真实长上下文数据的独立实验。</p>
        </DataComponent>
      </Section>
    </>}
  </article>;
}
