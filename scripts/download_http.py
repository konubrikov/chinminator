"""Resumable range downloader with size and SHA256 verification."""
import hashlib
import sys
import json
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from model_sources import model_source, model_metadata, ADAPTER_FILES

ROOT = Path(__file__).resolve().parents[1] / 'vendor/ComfyUI/models'
REPO = 'Comfy-Org/Qwen-Image-2.1'
FILES = ['diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
         'text_encoders/qwen3vl_8b_w4a8.safetensors',
         'vae/qwen_image_2.1_vae_bf16.safetensors']

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def download(name):
    repo, revision, remote = model_source(name)
    meta = model_metadata(name)
    size, digest = meta['size'], meta['lfs']['oid']
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size and sha(target) == digest:
        print('Verified', name, flush=True)
        return
    if name in ADAPTER_FILES:
        # These public adapters are pinned by commit and verified before activation.
        temporary = target.with_suffix('.partial')
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{remote}'
        print('Downloading', name, flush=True)
        subprocess.run([shutil.which('curl.exe') or shutil.which('curl') or 'curl',
                        '-fL', '--retry', '3', '--connect-timeout', '30', '--max-time', '1800',
                        '--output', str(temporary), url], check=True)
        if temporary.stat().st_size != size or sha(temporary) != digest:
            raise RuntimeError('Adapter checksum mismatch: ' + name)
        temporary.replace(target)
        print('Verified', name, flush=True)
        return
    if repo == 'black-forest-labs/flux-3-image-klein-comfy':
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        try:
            downloaded=Path(hf_hub_download(repo,name,revision=revision,local_dir=ROOT,force_download=target.exists()))
        except GatedRepoError as exc:
            raise RuntimeError('FLUX3 access denied. Accept conditions at https://huggingface.co/'+repo+' and log in with the SAME Hugging Face account. Run Start again.') from exc
        if downloaded.stat().st_size!=size or sha(downloaded)!=digest:
            raise RuntimeError('FLUX3 checksum mismatch: '+name)
        print('Verified',name,flush=True)
        return
    chunk_size = 16 * 1024 * 1024
    partial = target.with_suffix('.parts')
    partial.mkdir(exist_ok=True)
    count = (size + chunk_size - 1) // chunk_size
    def part(i):
        start, end = i * chunk_size, min(size, (i + 1) * chunk_size) - 1
        path = partial / str(i)
        if not path.exists() or path.stat().st_size != end - start + 1:
            subprocess.run([shutil.which('curl.exe') or shutil.which('curl') or 'curl', '-f', '-sS', '-L', '--retry', '5', '--connect-timeout', '30',
                            '--max-time', '600', '--range', f'{start}-{end}', '-o', str(path),
                            f'https://huggingface.co/{repo}/resolve/{revision}/{remote}?chunk={i}'], check=True)
        if path.stat().st_size != end - start + 1:
            raise RuntimeError(f'Invalid chunk size: {path}')
        print(f'{name}: chunk {i+1}/{count}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(part, range(count)))
    temporary = target.with_suffix('.assembling')
    with temporary.open('wb') as out:
        for i in range(count):
            with (partial / str(i)).open('rb') as inp:
                while data := inp.read(1024 * 1024):
                    out.write(data)
    if sha(temporary) != digest:
        raise RuntimeError(f'SHA256 mismatch: {name}')
    temporary.replace(target)
    for i in range(count):
        (partial / str(i)).unlink()
    partial.rmdir()
    print('Verified', name, flush=True)

if __name__ == '__main__':
    profile_path = Path(__file__).resolve().parents[1] / 'hardware-profile.json'
    if profile_path.exists():
        FILES = json.loads(profile_path.read_text(encoding='utf-8'))['models']
    elif '--macos' in sys.argv:
        FILES = ['diffusion_models/qwen_image_2.1_int8_convrot.safetensors', 'text_encoders/qwen3vl_8b_int8_convrot.safetensors', 'vae/qwen_image_2.1_vae_bf16.safetensors']
    for name in FILES:
        download(name)
