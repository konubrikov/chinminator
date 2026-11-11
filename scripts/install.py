"""Shared isolated installer; downloads a pinned ComfyUI snapshot without Git."""
import json,os,subprocess,sys,urllib.request,zipfile,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args,cwd=ROOT):subprocess.run([str(x) for x in args],cwd=cwd,check=True)
def main():
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Python 3.12 required')
    from configure_models import select_variant,select_engine
    engine=select_engine()
    variant=select_variant() if engine=='qwen2' else 'standard'
    py=ROOT/('server/.venv/Scripts/python.exe' if sys.platform=='win32' else 'server/.venv/bin/python')
    if not py.exists():run(sys.executable,'-m','venv',ROOT/'server/.venv')
    comfy=ROOT/'vendor/ComfyUI'
    revision=(ROOT/'scripts/comfy_revision.txt').read_text().strip()
    if not (comfy/'main.py').exists():
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'comfy.zip'
            urllib.request.urlretrieve(f'https://codeload.github.com/Comfy-Org/ComfyUI/zip/{revision}',archive)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    if Path(name).is_absolute() or '..' in Path(name).parts:raise RuntimeError('Unsafe archive path')
                z.extractall(tmp)
            comfy.parent.mkdir(exist_ok=True)
            shutil.move(str(Path(tmp)/('ComfyUI-'+revision)),comfy)
    run(py,'-m','pip','install','--upgrade','pip')
    run(py,'-m','pip','install','-r',ROOT/'server/requirements.txt','-r',comfy/'requirements.txt')
    if sys.platform=='win32':
        run(py,'-m','pip','install','--force-reinstall','--no-deps','torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cu130')
    run(py,ROOT/'scripts/doctor.py')
    run(py,ROOT/'scripts/prepare_analysis.py')
    run(py,ROOT/'scripts/configure_models.py','--variant',variant,'--engine',engine)
    from install_comfy_nodes import install
    install(ROOT)
    print('Installation complete. Starting selected model...')
    (ROOT/'.installed-version').write_text('1.7.0')
    run(py,ROOT/'scripts/launch.py','--no-menu')
if __name__=='__main__':
    try:main()
    except Exception as exc:print('INSTALLATION FAILED:',exc,file=sys.stderr);sys.exit(1)
