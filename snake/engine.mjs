export const W=24,H=16,DIRS={UP:[0,-1],DOWN:[0,1],LEFT:[-1,0],RIGHT:[1,0]};
export const cycle=[];
for(let x=0;x<W;x++) { if(x===0) for(let y=0;y<H;y++)cycle.push([x,y]); else if(x%2)for(let y=H-1;y>=1;y--)cycle.push([x,y]); else for(let y=1;y<H;y++)cycle.push([x,y]); }
for(let x=W-1;x>0;x--)cycle.push([x,0]);
const index=new Map(cycle.map((p,i)=>[p.join(','),i]));
const dist=(a,b)=>(b-a+cycle.length)%cycle.length;
export function fresh(){return {snake:cycle.slice(0,6).reverse().map(p=>[...p]),food:[17,5],score:0,over:false,won:false};}
export function candidates(s){const head=s.snake[0],hi=index.get(head.join(',')),ti=index.get(s.snake.at(-1).join(',')),fi=index.get(s.food?.join(','));return Object.entries(DIRS).map(([direction,[dx,dy]])=>{const p=[head[0]+dx,head[1]+dy],eat=p.join(',')===s.food?.join(','),legal=p[0]>=0&&p[0]<W&&p[1]>=0&&p[1]<H&&!s.snake.slice(0,eat?undefined:-1).some(b=>b.join(',')===p.join(','));const advance=dist(hi,index.get(p.join(','))),tail=dist(hi,ti);return {direction,p,eat,legal,advance,safe:legal&&advance>0&&(advance<tail||(advance===tail&&!eat))&&advance<=dist(hi,fi),distance:Math.abs(p[0]-(s.food?.[0]??0))+Math.abs(p[1]-(s.food?.[1]??0))};});}
export function localDecision(s){const c=candidates(s),safe=c.filter(a=>a.safe).sort((a,b)=>b.advance-a.advance);const chosen=safe[0]??c.find(a=>a.legal);const weights=c.map(a=>a.safe?Math.exp(-a.distance/2):0),sum=weights.reduce((a,b)=>a+b,0);return {choice:chosen?.direction,probabilities:Object.fromEntries(c.map((a,i)=>[a.direction,sum?weights[i]/sum:0])),candidates:c};}
export function step(s,d,rng=Math.random){if(s.over)return;const a=candidates(s).find(a=>a.direction===d);if(!a?.legal){s.over=true;return;}s.snake.unshift(a.p);if(a.eat){s.score++;const empty=cycle.filter(p=>!s.snake.some(b=>b[0]===p[0]&&b[1]===p[1]));if(!empty.length){s.over=true;s.won=true;s.food=null;}else s.food=[...empty[Math.floor(rng()*empty.length)]];}else s.snake.pop();}
