"""Real inference check on the target computer; run after Start."""
import json,time,urllib.request,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[1]
boundary=uuid.uuid4().hex
payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nforced\r\n'
 +f'--{boundary}\r\nContent-Disposition: form-data; name="params"\r\n\r\n'+json.dumps({'quality':{'auto_quality':0,'steps':28,'native_resolution':1,'seed':42}})
 +f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n').encode()
payload+=(root/'test-images/synthetic/synthetic-04.png').read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
request=urllib.request.Request('http://127.0.0.1:8765/process?sidebyside=1',data=payload,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
t=time.monotonic()
with urllib.request.urlopen(request,timeout=1800) as response:
 data=response.read();assert response.headers.get('X-Pipeline-Edited')=='1'
out=root/'work';out.mkdir(exist_ok=True);(out/'smoke-before-after.png').write_bytes(data)
print('Real inference passed in',round(time.monotonic()-t,1),'seconds:',out/'smoke-before-after.png')
