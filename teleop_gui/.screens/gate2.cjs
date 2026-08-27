const http=require('node:http'),fs=require('node:fs'),{spawn}=require('node:child_process');
const PORT=9339;
const child=spawn('google-chrome',['--headless=new','--no-sandbox','--enable-unsafe-swiftshader',
 '--hide-scrollbars','--window-size=1560,960',`--remote-debugging-port=${PORT}`,'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const req=(p,m='GET')=>new Promise((res,rej)=>{const r=http.request({host:'127.0.0.1',port:PORT,path:p,method:m},
 x=>{let b='';x.on('data',d=>b+=d);x.on('end',()=>res(JSON.parse(b)))});r.on('error',rej);r.end()});
(async()=>{
 for(let i=0;i<40;i++){try{await req('/json/version');break}catch{await sleep(250)}}
 const t=await req('/json/new?'+encodeURIComponent(process.argv[2]),'PUT');
 const WebSocket=require('ws'),ws=new WebSocket(t.webSocketDebuggerUrl);
 let id=0;const pend=new Map();
 const send=(m,p={})=>new Promise(res=>{const i=++id;pend.set(i,res);ws.send(JSON.stringify({id:i,method:m,params:p}))});
 await new Promise(r=>ws.on('open',r));
 ws.on('message',raw=>{const m=JSON.parse(raw.toString());if(m.id&&pend.has(m.id)){pend.get(m.id)(m.result);pend.delete(m.id)}});
 const title=async()=>(await send('Runtime.evaluate',{expression:"document.querySelector('#guidance-title')?.textContent ?? 'none'",returnByValue:true})).result?.value;
 const enter=async()=>{await send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
   await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});await sleep(800);};
 await sleep(7000);
 console.log('시작 게이트:', await title());
 await enter();
 console.log('Enter 후:', await title());
 // 시뮬레이터가 접촉으로 진입할 때까지 (전환 문턱 초과)
 for (let i=0;i<40;i++){ const t2=await title(); if(t2!=='none'){ console.log('전환 게이트:', t2); break; } await sleep(1000); }
 const shot=await send('Page.captureScreenshot',{format:'png'});
 fs.writeFileSync('.screens/gate2.png',Buffer.from(shot.data,'base64'));
 const behind=await send('Runtime.evaluate',{expression:"document.body.innerText.includes('NORMAL FORCE')",returnByValue:true});
 console.log('뒤 콘솔 렌더 유지 =', behind.result?.value);
 ws.close();child.kill();process.exit(0);
})();
