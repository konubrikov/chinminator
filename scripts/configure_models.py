"""Select/download a model variant, then activate it after successful setup."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from hardware_profile import detect
from model_sources import model_metadata, QWEN_ADAPTERS
from download_http import download

ROOT=Path(__file__).resolve().parents[1]
GGUF_REV='373048b8403a7820620065210a691263d4da0a61'

def select_engine(default='qwen2'):
    choices = {'1': ('qwen2', 'Qwen-Image 2.1'),
               '2': ('flux3', 'FLUX 3 Image Klein'),
               '3': ('qwen-lightning', QWEN_ADAPTERS['qwen-lightning']['label']),
               '4': ('qwen-pruna', QWEN_ADAPTERS['qwen-pruna']['label'])}
    print('\nChinminator: choose model')
    for number, (_, label) in choices.items():
        print(f'{number}. {label}')
    number = next((key for key, (engine, _) in choices.items() if engine == default), '1')
    while True:
        value = input(f'Model [1/2/3/4, Enter={number}]: ').strip() or number
        if value in choices:
            return choices[value][0]

def flux_login():
    from huggingface_hub import get_token, login
    if get_token():return
    import webbrowser,getpass
    url='https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy'
    print('FLUX3 requires your own Hugging Face account. Accept the model conditions in the browser.')
    webbrowser.open(url)
    print('Create a READ token at https://huggingface.co/settings/tokens (allow gated repositories).')
    token=getpass.getpass('Paste HF token here (hidden; saved in local HF cache): ')
    if not token.strip():raise RuntimeError('No token entered. Qwen can be installed without this login.')
    login(token=token.strip(),add_to_git_credential=False)

def select_variant():
    print('1. Standard: automatic INT8/BF16 hardware profile (default)')
    print('2. Lightweight Q4_K_M GGUF: less model memory; speed/quality may differ')
    while True:
        value=input('Model variant [1/2, Enter=1]: ').strip()
        if value in ('','1'):return 'standard'
        if value=='2':return 'q4'
        print('Enter 1 or 2.')

def install_gguf():
    target=ROOT/'vendor/ComfyUI/custom_nodes/ComfyUI-GGUF'
    marker=target/'.chinminator-revision'
    if not marker.exists() or marker.read_text().strip()!=GGUF_REV:
        if target.exists():
            raise RuntimeError(f'Existing unrecognized GGUF extension: {target}. Move it aside before installing the pinned version.')
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'gguf.zip'
            urllib.request.urlretrieve(f'https://codeload.github.com/leejet/ComfyUI-GGUF/zip/{GGUF_REV}',archive)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe archive path')
                z.extractall(tmp)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(Path(tmp)/('ComfyUI-GGUF-'+GGUF_REV)),target)
            marker.write_text(GGUF_REV)
    subprocess.run([sys.executable,'-m','pip','install','-r',str(target/'requirements.txt'),'gguf==0.19.0'],check=True)

def make_workflow(profile):
    graph=json.loads((ROOT/'server/workflows/qwen21_edit_quality.json').read_text())
    graph['3']['inputs']['unet_name']=Path(profile['models'][0]).name
    graph['4']['inputs'].update(clip_name=Path(profile['models'][1]).name,device=profile['text_device'])
    graph['8']['inputs']['device']=profile['cache_device']
    if profile['variant']=='q4':
        graph['3']={'class_type':'UnetLoaderGGUF','inputs':{'unet_name':Path(profile['models'][0]).name}}
        # Explicitly disable the reference KV cache to bound memory in Q4 mode.
        graph['8']['inputs']['device']='off'
    return graph

def configure(variant,engine='qwen2'):
    profile=detect(variant,engine)
    print('Selected profile:',profile['name'],flush=True)
    sizes=[]
    for name in profile['models']:
        meta=model_metadata(name)
        target=ROOT/'vendor/ComfyUI/models'/name
        if not target.exists() or target.stat().st_size!=meta['size']:sizes.append(meta['size'])
    required=sum(sizes)+max(sizes,default=0)+5*2**30
    if shutil.disk_usage(ROOT).free<required:
        raise RuntimeError(f'Need approximately {required/2**30:.1f} GiB free for download and assembly')
    if variant=='q4':install_gguf()
    if engine=='flux3' and sizes:flux_login()
    for name in profile['models']:download(name)
    records=[]
    if engine != 'flux3':
        records.append((ROOT/'server/workflows/qwen21_installed.json', make_workflow(profile)))
    records.append((ROOT/'hardware-profile.json',profile))
    for path,data in records:
        pending=path.with_suffix('.pending')
        pending.write_text(json.dumps(data,indent=2),encoding='utf-8')
        pending.replace(path)
    print('Model configuration saved:',engine)
    return profile

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['standard','q4'])
    parser.add_argument('--engine',choices=['qwen2','flux3', *QWEN_ADAPTERS])
    args=parser.parse_args()
    engine=args.engine or select_engine()
    configure((args.variant or select_variant()) if engine=='qwen2' else 'standard',engine)
