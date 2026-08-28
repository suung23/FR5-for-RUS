const http=require('node:http'),fs=require('node:fs'),{spawn}=require('node:child_process');
const PORT=9347;
const child=spawn('google-chrome',['--headless=new','--no-sandbox','--enable-unsafe-swiftshader',
 '--hide-scrollbars','--window-size=1560,960',`--remote-debugging-port=${PORT}`,'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const req=(p,m='GET')=>new Promise((res,rej)=>{const r=http.request({host:'127.0.0.1',port:PORT,path:p,method:m},
 x=>{let b='';x.on('data',d=>b+=d);x.on('end',()=>res(JSON.parse(b)))});r.on('error',rej);r.end()});
(async()=>{
 for(let i=0;i<60;i++){try{await req('/json/version');break}catch{await sleep(250)}}
 const t=await req('/json/new?'+encodeURIComponent(process.argv[2]),'PUT');
 const WebSocket=require('ws'),ws=new WebSocket(t.webSocketDebuggerUrl);
 let id=0;const pend=new Map();
 const send=(m,p={})=>new Promise(res=>{const i=++id;pend.set(i,res);ws.send(JSON.stringify({id:i,method:m,params:p}))});
 await new Promise(r=>ws.on('open',r));
 ws.on('message',raw=>{const m=JSON.parse(raw.toString());if(m.id&&pend.has(m.id)){pend.get(m.id)(m.result);pend.delete(m.id)}});
 await sleep(4000);
 await send('Input.dispatchKeyEvent',{type:'rawKeyDown',windowsVirtualKeyCode:13,key:'Enter',code:'Enter'});
 await send('Input.dispatchKeyEvent',{type:'keyUp',windowsVirtualKeyCode:13,key:'Enter',code:'Enter'});
 await sleep(22000);           // 힘 이력이 쌓이도록
 for(const spec of process.argv.slice(3)){
   const [view,out]=spec.split('=');
   await send('Runtime.evaluate',{expression:`window.location.hash='${view}'`});
   await sleep(3500);
   const shot=await send('Page.captureScreenshot',{format:'png'});
   fs.writeFileSync(out,Buffer.from(shot.data,'base64'));
   console.log('wrote',out);
 }
 ws.close();child.kill();process.exit(0);
})();
