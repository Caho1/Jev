// Reproduce the synthetic API ablation. Supply TYPESAFE_API_KEY in the environment.
// This sends only this directory's synthetic public fixtures to TypeSafe.
import {readFile,writeFile} from 'node:fs/promises';
import {createJevDecider} from '/Users/bystanders/.agents/skills/jev-mac-use/scripts/official_browser.mjs';
const decide=createJevDecider(process.env.TYPESAFE_API_KEY);
const corpus=JSON.parse(await readFile(new URL('./decision-corpus.json',import.meta.url),'utf8'));
const results=[];
for(const c of corpus)for(const mode of ['duplicated','compact']) {
 const state='AXWebArea Public test fixture\n'+(c.context||'')+(mode==='duplicated'?'\n'+c.candidates.map(x=>x.description).join('\n'):'');
 const started=performance.now();
 try {const answer=await decide({goal:c.goal,state,history:[],candidates:c.candidates});results.push({name:c.name,mode,expected:c.expected,actual:answer.id,passed:answer.id===c.expected,ms:performance.now()-started});}
 catch(e){results.push({name:c.name,mode,passed:false,error:e.message,ms:performance.now()-started});}
}
const file=new URL('./decision-checks-'+Date.now()+'.json',import.meta.url);
await writeFile(file,JSON.stringify({kind:'live API on synthetic decisions, not browser tasks',results},null,2)+'\n');
console.table(results);console.log(file.pathname);
