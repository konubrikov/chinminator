"""Model choice, missing weights, and managed local services."""
import argparse,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'work'
STATE=LOGS/'managed-services.json'

def read_url(url):
    try:
        with urllib.request.urlopen(url,timeout=10) as response:return json.load(response)
    except (OSError,ValueError):return None

def stop_managed():
    import psutil
    current=read_url('http://127.0.0.1:8765/health')
    if current and current.get('busy'):raise RuntimeError('Retouch is running. Wait before switching/stopping.')
    queue=read_url('http://127.0.0.1:8188/queue')
    if queue and (queue.get('queue_running') or queue.get('queue_pending')):
        raise RuntimeError('ComfyUI has active jobs. Wait before stopping.')
    if not STATE.exists():return
    state=json.loads(STATE.read_text('utf-8'))
    for service in ('api','comfy'):
        item=state.get(service)
        if not item:continue
        try:
            proc=psutil.Process(item['pid'])
            if abs(proc.create_time()-item['created'])>.01:continue
            if Path(proc.cwd()).resolve()!=Path(item['cwd']).resolve():continue
            if proc.cmdline()!=item['command']:continue
            proc.terminate();proc.wait(timeout=20)
        except psutil.NoSuchProcess:pass
    STATE.unlink(missing_ok=True)

def spawn(service,command,cwd):
    import psutil
    with (LOGS/(service+'.log')).open('a',encoding='utf-8') as logfile:
        child=subprocess.Popen(command,cwd=cwd,stdout=logfile,stderr=logfile,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    proc=psutil.Process(child.pid)
    state=json.loads(STATE.read_text('utf-8')) if STATE.exists() else {}
    state[service]={'pid':child.pid,'created':proc.create_time(),'cwd':str(cwd),'command':proc.cmdline()}
    STATE.write_text(json.dumps(state,indent=2),encoding='utf-8')
    return child

def wait_ready(child,url,seconds=180):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if child.poll() is not None:raise RuntimeError('Service stopped during startup; see work/*.log')
        value=read_url(url)
        if value:return value
        time.sleep(2)
    raise RuntimeError('Startup timeout; see work/*.log')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-menu',action='store_true')
    parser.add_argument('--stop',action='store_true')
    args=parser.parse_args()
    if args.stop:
        stop_managed();print('Managed Chinminator services stopped.');return 0
    from configure_models import select_engine,select_variant,configure
    from install_comfy_nodes import install
    LOGS.mkdir(exist_ok=True)
    profile_path=ROOT/'hardware-profile.json'
    profile=json.loads(profile_path.read_text('utf-8')) if profile_path.exists() else None
    engine=profile.get('engine','qwen2') if profile else 'qwen2'
    if not args.no_menu:engine=select_engine(engine)
    current=read_url('http://127.0.0.1:8765/health')
    if current:
        if current.get('installation_root')!=str(ROOT):
            raise RuntimeError('Another/older Chinminator is running on port 8765. Close its server first.')
        if current.get('engine')==engine and current.get('api_version',0)>=5 and current.get('ready'):
            print('Selected model is already running:',current['model']);return 0
        stop_managed()
        if read_url('http://127.0.0.1:8765/health'):raise RuntimeError('API is not managed by this launcher. Close it first.')
    if (not profile or profile.get('engine','qwen2')!=engine or
        not all((ROOT/'vendor/ComfyUI/models'/name).exists() for name in profile['models'])):
        variant=select_variant() if engine=='qwen2' else 'standard'
        profile=configure(variant,engine)
    install(ROOT)
    os.environ['CHINMINATOR_MODEL']=engine
    os.environ['CHINMINATOR_MAX_SIDE']=str(profile['quality_max_side'])
    os.environ['CHINMINATOR_AUTO_MAX_SIDE']=str(profile['auto_max_side'])
    if sys.platform=='darwin':
        import torch
        if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU / MPS unavailable')
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
    if not read_url('http://127.0.0.1:8188/system_stats'):
        child=spawn('comfy',[sys.executable,'-X','utf8','main.py','--listen','127.0.0.1','--port','8188']+profile['flags'],ROOT/'vendor/ComfyUI')
        wait_ready(child,'http://127.0.0.1:8188/system_stats')
    api=spawn('api',[sys.executable,'-X','utf8','-m','uvicorn','app:app','--host','127.0.0.1','--port','8765'],ROOT/'server')
    health=wait_ready(api,'http://127.0.0.1:8765/health')
    if not health.get('ready'):raise RuntimeError(health.get('backend','Model unavailable'))
    from install_plugin import install_plugin
    install_plugin(ROOT)
    print('\nREADY:',health['model'],flush=True)
    print('Photoshop: Plugins > Chinminator. Keep this window open. Stop.cmd / Stop.command stops managed services.',flush=True)
    try:return api.wait()
    except KeyboardInterrupt:stop_managed();return 0
