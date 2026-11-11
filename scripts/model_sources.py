"""Pinned third-party GGUF; official Comfy-Org sources for other weights."""
from urllib.request import urlopen
import json
from pathlib import Path

FLUX3=json.loads((Path(__file__).with_name('flux3_models.json')).read_text('utf-8'))
QWEN_ADAPTERS = json.loads((Path(__file__).resolve().parents[1] / 'server/qwen_adapters.json').read_text('utf-8'))
ADAPTER_FILES = {adapter['file']: adapter for adapter in QWEN_ADAPTERS.values()}

GGUF_FILE='diffusion_models/qwen_image_2.1_Q4_K_M.gguf'
GGUF_REV='fc008ac98b20f38b7cd338b2506aab2fdc03dfd1'

def model_source(name):
    if name in ADAPTER_FILES:
        adapter = ADAPTER_FILES[name]
        return adapter['repo'], adapter['revision'], adapter['remote']
    if name in FLUX3['files']:
        return FLUX3['repo'],FLUX3['revision'],name
    if name==GGUF_FILE:
        return ('chfm/Qwen-Image-2.1-GGUF',GGUF_REV,name.split('/')[-1])
    return ('Comfy-Org/Qwen-Image-2.1','main',name)

def model_metadata(name):
    if name in ADAPTER_FILES:
        adapter = ADAPTER_FILES[name]
        return {'size': adapter['size'], 'lfs': {'oid': adapter['sha256']}}
    if name in FLUX3['files']:
        entry=FLUX3['files'][name]
        return {'size':entry['size'],'lfs':{'oid':entry['sha256']}}
    repo,revision,remote=model_source(name)
    folder=remote.rpartition('/')[0]
    with urlopen(f'https://huggingface.co/api/models/{repo}/tree/{revision}/'+folder,timeout=30) as response:
        meta=next(x for x in json.load(response) if x['path']==remote)
    if name==GGUF_FILE and meta['lfs']['oid']!='dc956c958fbfa1d5c64ec316d7e865283d17d97a9eb332a4a74a4d63afaae9a5':
        raise RuntimeError('Unexpected GGUF checksum')
    return meta
