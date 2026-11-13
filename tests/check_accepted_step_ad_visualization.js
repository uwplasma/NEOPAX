/* Tiny numeric and mock-DOM checks. No JAX, NEOPAX or physical simulation. */
function checkAcceptedStepAD(html) {
  const core=html.match(/<script id="ad-core">([\s\S]*?)<\/script>/)[1];
  const ui=html.match(/<script id="ad-ui">([\s\S]*?)<\/script>/)[1];
  new Function(core)();
  const api=globalThis.CoolingAD;
  const assert=(condition,message)=>{if(!condition)throw Error(message);};
  const close=(x,y,tolerance=1e-12)=>Math.abs(x-y)<=tolerance*Math.max(1,Math.abs(y));
  let count=0;
  for(const k of [.05,.2,.5]) for(const tolerance of [.1,.6,2]) for(const firstStep of [1,6,10]){
    const run=api.simulate({k,tolerance,firstStep}),fd=api.finiteDifferences(run);
    assert(close(run.frozenGradient,run.attemptGradient),'AD graphs disagree');
    assert(close(run.finalTemperature,run.frozenTemperature),'Endpoint mismatch');
    assert(close(run.frozenGradient,fd.frozen,2e-7),'Frozen-schedule FD mismatch');
    const accepted=run.attempts.filter(a=>a.accepted);
    // Independent derivative of the product of implicit-Euler stability factors.
    const reference=-(run.finalTemperature-20)*accepted.reduce((s,a)=>s+a.h/(1+k*a.h),0);
    assert(close(run.frozenGradient,reference),'Independent implicit-Euler derivative mismatch');
    assert(close(accepted.reduce((s,a)=>s+a.h,0),10),'Accepted timesteps do not reach final time');
    assert(close(run.frozenContributions.reduce((s,x)=>s+x,0),run.frozenGradient),'Accepted contribution sum mismatch');
    assert(close(run.attemptContributions.reduce((s,x)=>s+x,0),run.attemptGradient),'All-attempts contribution sum mismatch');
    run.attempts.forEach((a,i)=>{
      if(!a.accepted){
        assert(a.nextY===a.y,'Rejection advanced physical state');
        assert(run.attemptContributions[i]===0&&run.frozenContributions[i]===0,'Rejection contributed to gradient');
      }
    });
    count++;
  }
  const elements=Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],{
    value:'',textContent:'',innerHTML:'',max:0,classList:{toggle(){}},events:{},
    addEventListener(name,fn){this.events[name]=fn;}
  }]));
  elements.rate.value='.2';elements.tolerance.value='.6';elements['first-step'].value='6';
  const radios=['frozen','attempts'].map(value=>({value,events:{},addEventListener(name,fn){this.events[name]=fn;}}));
  const documentMock={getElementById(id){assert(elements[id],'Missing element '+id);return elements[id];},querySelectorAll(){return radios;}};
  let tick;
  new Function('document','setInterval','clearInterval',ui)(documentMock,fn=>{tick=fn;return 1;},()=>{});
  let renders=0;
  for(const radio of radios){
    radio.events.change();
    for(let i=0;i<=Number(elements.frame.max);i++){
      elements.frame.value=String(i);elements.frame.events.input();
      assert(!/NaN|undefined/.test(elements.trajectory.innerHTML+elements.flow.innerHTML+elements.numbers.textContent),'Invalid rendered value');
      renders++;
    }
  }
  elements.play.events.click();tick();elements.play.events.click();
  const run=api.simulate();
  assert(run.attempts.some(a=>!a.accepted),'Default animation needs at least one rejected trial');
  return {parameterCases:count,renderedFrames:renders,acceptedGradient:run.frozenGradient,
    allAttemptsGradient:run.attemptGradient,acceptedNodes:run.acceptedNodes,allAttemptNodes:run.attemptNodes};
}
if(typeof module!=='undefined'&&module.exports){
  module.exports={checkAcceptedStepAD};
  if(require.main===module){
    const fs=require('node:fs'),path=require('node:path');
    const html=fs.readFileSync(path.join(__dirname,'../docs/visualizations/accepted_step_ad.html'),'utf8');
    console.log(JSON.stringify(checkAcceptedStepAD(html),null,2));
  }
}
