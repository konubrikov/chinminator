"""Windows bootstrap installer; no PowerShell policy changes."""
import os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args,cwd=ROOT):
    subprocess.run([str(a) for a in args],cwd=cwd,check=True)
def main():
    if sys.platform!='win32' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Windows and Python 3.12 are required')
    if not shutil.which('git'):
        raise RuntimeError('Install Git for Windows from git-scm.com and retry')
    py=ROOT/'server/.venv/Scripts/python.exe'
    if not py.exists():run(sys.executable,'-m','venv',ROOT/'server/.venv')
    comfy=ROOT/'vendor/ComfyUI'
    if not (comfy/'main.py').exists():
        comfy.parent.mkdir(exist_ok=True)
        run('git','clone','https://github.com/Comfy-Org/ComfyUI.git',comfy)
        run('git','checkout',(ROOT/'scripts/comfy_revision.txt').read_text().strip(),cwd=comfy)
    run(py,'-m','pip','install','--upgrade','pip')
    run(py,'-m','pip','install','-r',ROOT/'server/requirements.txt','-r',comfy/'requirements.txt')
    run(py,'-m','pip','install','--force-reinstall','--no-deps','torch','torchvision','torchaudio','--index-url','https://download.pytorch.org/whl/cu130')
    run(py,ROOT/'scripts/doctor.py')
    run(py,ROOT/'scripts/download_http.py')
    print('Installation complete. Run Start.cmd; load plugin/manifest.json in UXP Developer Tool.')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('INSTALLATION FAILED:',exc,file=sys.stderr);sys.exit(1)
