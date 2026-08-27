const http=require('node:http'),fs=require('node:fs'),{spawn}=require('node:child_process');
const PORT=9338;
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
 await sleep(9000);
 const dlg=await send('Runtime.evaluate',{expression:"document.querySelectorAll('[role=dialog]').length",returnByValue:true});
 const inert=await send('Runtime.evaluate',{expression:"!!document.querySelector('[inert]')",returnByValue:true});
 console.log('gate open =',dlg.result?.value,'| console inert =',inert.result?.value);
 const shot1=await send('Page.captureScreenshot',{format:'png'});
 fs.writeFileSync('.screens/gate.png',Buffer.from(shot1.data,'base64'));
 // Enter 키
 await send('Input.dispatchKeyEvent',{type:'rawKeyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
 await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13,nativeVirtualKeyCode:13});
 await sleep(2500);
 const dlg2=await send('Runtime.evaluate',{expression:"document.querySelectorAll('[role=dialog]').length",returnByValue:true});
 const inert2=await send('Runtime.evaluate',{expression:"!!document.querySelector('[inert]')",returnByValue:true});
 console.log('after ENTER: gate =',dlg2.result?.value,'| inert =',inert2.result?.value);
 await sleep(3000);
 const shot2=await send('Page.captureScreenshot',{format:'png'});
 fs.writeFileSync('.screens/console.png',Buffer.from(shot2.data,'base64'));
 ws.close();child.kill();process.exit(0);
})();
