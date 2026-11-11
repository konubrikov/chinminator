"""Resumeable native-resolution benchmark for Qwen, Turbo LoRA/Full and FLUX klein."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'server'))

VARIANTS=[('qwen-optimized','qwen-report.json',40),('qwen-turbo-lora','turbo-report.json',8),('qwen-turbo-full','turbo-full-report.json',8),('flux2-klein','flux-report.json',4)]

def atomic_json(path,data):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('run',type=Path)
    parser.add_argument('--wait-for',type=Path);parser.add_argument('--wait-count',type=int,default=5)
    args=parser.parse_args();args.run.mkdir(parents=True,exist_ok=True)
    if args.wait_for:
        while not (args.wait_for.exists() and len(json.loads(args.wait_for.read_text()))==args.wait_count):
            print('Waiting for first comparison to release GPU',flush=True);time.sleep(30)
    from config import QualityParams,EDIT_PROMPT_QUALITY
    from pipeline import process_image
    from geometry import analyze
    from face_detect import detect_faces
    import edit_qwen
    from backends.comfyui_backend import ComfyUIBackend
    from backends.flux_klein_backend import FluxKleinBackend
    files=sorted(p for p in args.source.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.webp'))
    assert files and len({p.stem for p in files})==len(files),'Missing inputs or duplicate stems'
    prompt=EDIT_PROMPT_QUALITY
    (args.run/'prompt.txt').write_text(prompt,encoding='utf-8')
    code_hash=hashlib.sha256()
    for f in sorted((ROOT/'server').rglob('*.py')):
        if not any(p in f.parts for p in ('.venv','__pycache__')):code_hash.update(f.read_bytes())
    fingerprint=code_hash.hexdigest()
    atomic_json(args.run/'run-settings.json',dict(code_sha256=fingerprint,seed=42,native_resolution=True,prompt=prompt,source=str(args.source.resolve()),hardware='RTX 4070 Ti SUPER 16GB'))
    for variant,report_name,steps in VARIANTS:
        for key in ('CHINMINATOR_QWEN_TURBO_LORA','CHINMINATOR_QWEN_TURBO_FULL','CHINMINATOR_QWEN_CACHE'):
            os.environ.pop(key,None)
        os.environ['CHINMINATOR_QWEN_OPTIMIZE']='1'
        if variant=='qwen-turbo-lora':os.environ['CHINMINATOR_QWEN_TURBO_LORA']='1'
        if variant=='qwen-turbo-full':os.environ['CHINMINATOR_QWEN_TURBO_FULL']='1'
        edit_qwen._BACKENDS.clear();edit_qwen._BACKENDS[variant]=FluxKleinBackend() if variant=='flux2-klein' else ComfyUIBackend()
        dest=args.run/variant;dest.mkdir(exist_ok=True);report=args.run/report_name
        existing=json.loads(report.read_text()) if report.exists() else []
        rows=[]
        for f in files:
            source_sha=hashlib.sha256(f.read_bytes()).hexdigest();image=cv2.imread(str(f));assert image is not None,f
            output_name=f.stem+'.png'
            prior=next((r for r in existing if r['input']==f.name and r['source_sha256']==source_sha and r.get('code_sha256')==fingerprint),None)
            if prior and (dest/output_name).exists() and hashlib.sha256((dest/output_name).read_bytes()).hexdigest()==prior.get('output_sha256'):
                rows.append(prior);atomic_json(report,rows);print('Resume',variant,f.name,flush=True);continue
            faces=detect_faces(image)
            if not faces:raise RuntimeError('No face in '+f.name)
            geo=analyze(image,faces[0]);cv2.imwrite(str(args.run/(f.stem+'-mask.png')),geo.mask)
            started=time.perf_counter()
            try:
                result=process_image(image,mode='forced',quality=QualityParams(auto_quality=0,native_resolution=1,steps=steps,seed=42,cfg=1),prompt=prompt)
            except Exception as exc:
                atomic_json(args.run/'error.json',dict(variant=variant,input=f.name,error=str(exc)));raise
            seconds=time.perf_counter()-started
            assert result.edited and result.image.shape==image.shape,(variant,f.name)
            outside=int(np.count_nonzero(np.any(result.image!=image,axis=2)&(geo.mask==0)));assert outside==0
            assert cv2.imwrite(str(dest/output_name),result.image)
            rows.append(dict(variant=variant,input=f.name,output_file=output_name,seconds=seconds,steps=steps,seed=result.seed,
                shape=list(image.shape),outside_mask_changed_pixels=outside,source_sha256=source_sha,
                output_sha256=hashlib.sha256((dest/output_name).read_bytes()).hexdigest(),code_sha256=fingerprint,edited=result.edited,backend=result.backend))
            atomic_json(report,rows)
            print(variant,f.name,round(seconds,2),'seconds',flush=True)
    print('All four variants complete:',len(files)*4,'renders',flush=True)

if __name__=='__main__':main()
