import assert from 'node:assert/strict';
import {W,H,cycle,fresh,localDecision,step,candidates} from './engine.mjs';
assert.equal(cycle.length,W*H);assert.equal(new Set(cycle.map(p=>p.join(','))).size,W*H);
for(let i=0;i<cycle.length;i++){let a=cycle[i],b=cycle[(i+1)%cycle.length];assert.equal(Math.abs(a[0]-b[0])+Math.abs(a[1]-b[1]),1);}
for(let seed=1;seed<=5;seed++){let n=seed;const rng=()=>((n=(n*1664525+1013904223)>>>0)/4294967296);const s=fresh();for(let i=0;i<150000&&!s.over;i++){const d=localDecision(s);assert.ok(candidates(s).find(c=>c.direction===d.choice)?.safe);step(s,d.choice,rng);}assert.ok(s.won,'autoplay should fill board');}
const collision=fresh();step(collision,'LEFT');assert.ok(collision.over);
const eating=fresh();eating.food=[0,6];step(eating,'DOWN');assert.equal(eating.score,1);assert.equal(eating.snake.length,7);assert.ok(!eating.snake.some(p=>p.join(',')===eating.food.join(',')));
console.log('PASS: cycle integrity, 5 complete seeded games, collision, growth, food placement.');
