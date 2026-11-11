"""Use Adobe's packaged installer once per plugin version."""
import hashlib,os,subprocess,sys
from pathlib import Path

def install_plugin(root):
    package=root/'Chinminator.ccx'
    if not package.exists():
        print('Development checkout: load plugin/manifest.json using Adobe UXP Developer Tool.')
        return
    marker=root/'work/plugin-installed.sha256'
    digest=hashlib.sha256(package.read_bytes()).hexdigest()
    if marker.exists() and marker.read_text().strip()==digest:return
    if sys.platform=='win32':
        agent=Path(os.environ.get('CommonProgramFiles',r'C:\Program Files\Common Files'))/'Adobe/Adobe Desktop Common/RemoteComponents/UPI/UnifiedPluginInstallerAgent/UnifiedPluginInstallerAgent.exe'
        flag='/install'
    else:
        agent=Path('/Library/Application Support/Adobe/Adobe Desktop Common/RemoteComponents/UPI/UnifiedPluginInstallerAgent/UnifiedPluginInstallerAgent.app/Contents/MacOS/UnifiedPluginInstallerAgent')
        flag='--install'
    if agent.exists():
        result=subprocess.run([str(agent),flag,str(package)],capture_output=True,text=True)
        if result.returncode==0:
            marker.write_text(digest);print('Photoshop plugin installed.');return
        print('Adobe automatic install returned',result.returncode,'— confirm installation in Creative Cloud.')
    print('Opening Chinminator.ccx. Confirm Adobe installation; restart Photoshop if requested.')
    if sys.platform=='win32':os.startfile(package)
    else:subprocess.run(['open',str(package)],check=True)
    print('If installation fails, open Chinminator.ccx manually. Server remains available.')
