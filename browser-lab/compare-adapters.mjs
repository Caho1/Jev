/** Offline control-flow comparison. No real browser/model latency is simulated. */
import * as before from './official-browser-before-20260920.mjs';
import * as after from '/Users/bystanders/.agents/skills/jev-mac-use/scripts/official_browser.mjs';
import {writeFile} from 'node:fs/promises';
const page=body=>`0 window Browser\n 1 AXWebArea Lab, URL: example.org/\n  ${body}`;
const task={surface:'app',scope:'URL: example\\.org/',goal:'Open result',success:['text Complete'],actions:[{id:'go',op:'click',pattern:'^button Go$'}]};
const cases={
 renumbered_no_effect:()=>({task,read:n=>page(`${n+2} button Go`),decision:()=>({id:'go'})}),
 no_eligible_actions:()=>({task,read:()=>page('2 text No results'),decision:()=>({id:'BLOCKED'})}),
 candidate_payload:()=>({task,read:()=>page('2 button Go'),decision:()=>({id:'BLOCKED'})}),
 verified_dependency_chain:()=>({
  task:{...task,actions:[{...task.actions[0],requires:'text Ready',after:'text First done',once:true,deterministic:true},{id:'next',op:'click',pattern:'^button Next$',requires:'text Ready',after:'text Complete',dependsOn:['go'],deterministic:true}]},
  read:(n,mutations)=>page('2 button Go\n  3 button Next\n  4 text Ready'+(mutations?'\n  5 text First done':'')+(mutations===2?'\n  6 text Complete':'')),
  decision:n=>({id:n===1?'go':'next'})
 })
};
const results=[];
for(const [name,make] of Object.entries(cases))for(const [version,adapter] of Object.entries({before,after})) {
 const fixture=make();let observations=0,mutations=0,decisions=0,payload_bytes=0;
 const target={getAXState:async()=>fixture.read(++observations,mutations),click:async()=>{mutations++}};
 const decide=async input=>{decisions++;const {remainingMs,...payload}=input;payload_bytes+=Buffer.byteLength(JSON.stringify(payload));return fixture.decision(decisions)};
 const result=await adapter.runOfficial({target,task:fixture.task,decide,maxSteps:6});
 results.push({case:name,version,status:result.status,observations,mutations,decisions,payload_bytes});
}
const report={kind:'offline synthetic control-flow comparison',warning:'Counts only. Not a live task speed benchmark or comparison with competitors. Host-supplied dependencies are unsupported by the old version.',results};
await writeFile(new URL('./comparison-20260920.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.table(results);
