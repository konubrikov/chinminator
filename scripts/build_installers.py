"""Build source/bootstrap installer archives without models or private data."""
import hashlib,zipfile,stat,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'dist'
OUT.mkdir(parents=True,exist_ok=True)
def public_bytes(path):
    data=path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"): return data
    chunks=[data[:8]]; offset=8
    while offset+12<=len(data):
        length=int.from_bytes(data[offset:offset+4],"big")
        kind=data[offset+4:offset+8]
        if kind not in (b"iTXt",b"tEXt",b"zTXt",b"eXIf"):
            chunks.append(data[offset:offset+length+12])
        offset+=length+12
    return b"".join(chunks)

COMMON=[]
for folder in ('plugin','scripts','server'):
    for p in (ROOT/folder).rglob('*'):
        rel=p.relative_to(ROOT)
        if any(part in ('.venv','__pycache__','models','comfyui_user') for part in rel.parts):continue
        if p.name == 'qwen21_installed.json':continue
        if p.is_file() and p.suffix in ('.py','.json','.txt','.md','.css','.html','.js','.jsx','.png','.ps1','.bat','.sh'):COMMON.append(p)
COMMON += [ROOT/'test-images/synthetic/synthetic-04.png',ROOT/'INSTALL-WINDOWS-v2.txt',ROOT/'INSTALL-MACOS-v2.txt',ROOT/'README-release.md',ROOT/'BATCH-VALIDATION.json',ROOT/'IMAGE-PROCESSOR-VALIDATION.json']
COMMON += [ROOT/'Chinminator.ccx']
COMMON += [ROOT/'INSTALL-1.7.txt']
assert (ROOT/'Chinminator.ccx').exists(), 'Package the updated panel with Adobe UXP Developer Tool first'
with zipfile.ZipFile(ROOT/'Chinminator.ccx') as package:
    assert json.loads(package.read('manifest.json'))==json.loads((ROOT/'plugin/manifest.json').read_text('utf-8'))
    for name in ('main.js','index.html','styles.css','default-prompt.txt'):
        assert package.read(name)==(ROOT/'plugin'/name).read_bytes(), 'CCX is stale: '+name
license=ROOT/'LICENSE'
if license.exists():COMMON.append(license)
for platform in ('Windows','macOS-M4-24GB'):
    target=OUT/f'Chinminator-{platform}.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in COMMON:
            name='Chinminator/'+('README.md' if p.name=='README-release.md' else p.relative_to(ROOT).as_posix())
            z.writestr(name,public_bytes(p))
        entries={}
        if platform=='Windows':
            entries={'Install.cmd':'@echo off\r\ncd /d "%~dp0"\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\\bootstrap_windows.ps1\r\nif errorlevel 1 echo Installation failed. See the error above.\r\npause\r\n',
                     'Start.cmd':'@echo off\r\ncd /d "%~dp0"\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\\bootstrap_windows.ps1\r\nif errorlevel 1 pause\r\n',
                     'Test.cmd':'@echo off\r\ncd /d "%~dp0"\r\nserver\\.venv\\Scripts\\python.exe scripts\\smoke_test.py\r\npause\r\n'}
        else:
            for name,script in [('Install','setup_macos.sh'),('Start','setup_macos.sh')]:
                entries[name+'.command']='#!/bin/bash\ncd "$(dirname "$0")"\nbash scripts/'+script+'\nstatus=$?\nread -r -p "Press Enter to close..."\nexit "$status"\n'
            entries['Test.command']='#!/bin/bash\ncd "$(dirname "$0")"\nserver/.venv/bin/python scripts/smoke_test.py\nstatus=$?\nread -r -p "Press Enter to close..."\nexit "$status"\n'
        if platform=='Windows':
            entries['Stop.cmd']='@echo off\r\ncd /d "%~dp0"\r\nserver\\.venv\\Scripts\\python.exe -X utf8 scripts\\launch.py --stop\r\nif errorlevel 1 pause\r\n'
            entries['Choose-model.cmd']='@echo off\r\ncd /d "%~dp0"\r\nserver\\.venv\\Scripts\\python.exe -X utf8 scripts\\configure_models.py\r\nif errorlevel 1 echo Model setup failed. See above.\r\npause\r\n'
        else:
            entries['Stop.command']='#!/bin/bash\ncd "$(dirname "$0")"\nserver/.venv/bin/python -X utf8 scripts/launch.py --stop\nstatus=$?\nread -r -p "Press Enter to close..."\nexit "$status"\n'
            entries['Choose-model.command']='#!/bin/bash\ncd "$(dirname "$0")"\nserver/.venv/bin/python -X utf8 scripts/configure_models.py\nstatus=$?\nread -r -p "Press Enter to close..."\nexit "$status"\n'
        for name,content in entries.items():
            item=zipfile.ZipInfo('Chinminator/'+name);item.create_system=3;item.external_attr=(stat.S_IFREG|0o755)<<16
            z.writestr(item,content)
        z.write(ROOT/'INSTALL-1.7.txt','Chinminator/START-HERE.txt')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert not any('/.venv/' in n or '/work/' in n or '/models/' in n for n in z.namelist())
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (target.with_suffix('.zip.sha256')).write_text(digest+'  '+target.name+'\n')
    print(target,target.stat().st_size,digest)
