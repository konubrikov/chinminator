"""Build a full-resolution, hash-verified four-variant comparison folder."""
import argparse,csv,hashlib,html,json,shutil,statistics
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODELS=[('Qwen 2.1','qwen-optimized','qwen-report.json',40),('Qwen 2.1 Turbo LoRA','qwen-turbo-lora','turbo-report.json',8),('Qwen 2.1 Turbo Full','qwen-turbo-full','turbo-full-report.json',8),('FLUX.2 klein 4B','flux2-klein','flux-report.json',4)]

def build(run,out,source_dir=None):
    source_dir=source_dir or ROOT/"test-images/synthetic"
    if out.name=='Chinminator-3-models-2026-11-03':
        out=out.with_name('Chinminator-4-variants-2026-11-03')
    out.mkdir(parents=True,exist_ok=True)
    records={};speed=[];audit=[]
    for title,variant,report,steps in MODELS:
        rows=[r for r in json.loads((run/report).read_text()) if r['variant']==variant]
        assert rows,(variant,'missing results')
        records[variant]={r['input']:r for r in rows}
    names=sorted(records[MODELS[0][1]])
    assert all(set(group)==set(names) for group in records.values())
    photo_count=len(names)
    for name in names:
        original=source_dir/name
        source=cv2.imread(str(original));assert source is not None
        sha=hashlib.sha256(original.read_bytes()).hexdigest()
        target=out/'00-originals'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(original,target)
        row=[name];photo={'file':name,'source_sha256':sha,'size':[source.shape[1],source.shape[0]],'models':{}}
        mask=cv2.imread(str(run/(Path(name).stem+'-mask.png')),cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape==source.shape[:2]
        for index,(title,variant,report,steps) in enumerate(MODELS,1):
            record=records[variant][name];assert record['source_sha256']==sha
            file=run/variant/record.get('output_file',name);image=cv2.imread(str(file));assert image is not None and image.shape==source.shape
            outside=int(np.count_nonzero(np.any(image!=source,axis=2)&(mask==0)));assert outside==0
            dest=out/(str(index).zfill(2)+'-'+variant)/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
            seconds=record['seconds'];row.append(seconds)
            photo['models'][variant]={'seconds':seconds,'steps':steps,'seed':record['seed'],'backend':record['backend'],'outside_mask_changed_pixels':outside,'output_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        speed.append(row);audit.append(photo)
    speed.append([f'Медиана всех {photo_count}']+[statistics.median(r[i] for r in speed[:photo_count]) for i in range(1,len(MODELS)+1)])
    speed.append([f'Медиана фото 2–{photo_count}']+[statistics.median(r[i] for r in speed[1:photo_count]) for i in range(1,len(MODELS)+1)])
    speed.append([f'Суммарно {photo_count} фото']+[sum(r[i] for r in speed[:photo_count]) for i in range(1,len(MODELS)+1)])
    headers=['Фото']+[m[0]+' (с)' for m in MODELS]
    with (out/'speed.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,delimiter=';');writer.writerow(headers)
        for row in speed:writer.writerow([row[0]]+[f'{v:.2f}'.replace('.',',') for v in row[1:]])
    table='| '+' | '.join(headers)+' |\n|'+'|'.join(['---']+['---:']*len(MODELS))+'|\n'+''.join('| '+row[0]+' | '+' | '.join(f'{v:.2f}' for v in row[1:])+' |\n' for row in speed)
    note='Время полного process_image: анализ, подготовка, GPU-генерация и финальное наложение; без сохранения в Photoshop. RTX 4070 Ti SUPER 16 ГБ, seed=42. Исходное разрешение сохранено; добавлены только поля до сетки модели. Первое фото каждого варианта может включать загрузку/инициализацию. Qwen запущен после базового A/B-прогона, поэтому первая строка не является честным сравнением холодного старта. Медиана без первого фото лучше отражает пакетную обработку. Вне маски изменено 0 пикселей. Числа скорости не доказывают качество анатомии; результаты требуют отдельного визуального просмотра.'
    (out/'SPEED.md').write_text(table+'\n'+note+'\n',encoding='utf-8')
    (out/'validation.json').write_text(json.dumps({'hardware':'RTX 4070 Ti SUPER 16GB','native_resolution':True,'timing_note':note,'photos':audit},ensure_ascii=False,indent=2),encoding='utf-8')
    for _,_,report,_ in MODELS:shutil.copy2(run/report,out/report)
    from config import EDIT_PROMPT_QUALITY
    (out/'prompt.txt').write_text((run/'prompt.txt').read_text('utf-8') if (run/'prompt.txt').exists() else EDIT_PROMPT_QUALITY,encoding='utf-8')
    tr='<tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'
    tr+=''.join('<tr><td>'+html.escape(row[0])+'</td>'+''.join(f'<td>{v:.2f}</td>' for v in row[1:])+'</tr>' for row in speed)
    body=''
    folders=['00-originals']+[str(i).zfill(2)+'-'+m[1] for i,m in enumerate(MODELS,1)]
    titles=['Исходник']+[m[0] for m in MODELS]
    for name in names:
        body+='<h2>'+name+'</h2><div class="photos">'
        for folder,title in zip(folders,titles):
            path=folder+'/'+name
            body+=f'<figure><figcaption>{html.escape(title)}</figcaption><a href="{path}"><img src="{path}" loading="lazy"></a></figure>'
        body+='</div>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Chinminator — 4 варианта</title><style>body{font:16px system-ui;background:#181a1e;color:#eee;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #555;padding:8px 16px}td{text-align:right}.photos{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:10px;overflow:auto}figure{margin:0}img{width:100%;height:auto}figcaption{padding:10px}p{max-width:1100px;line-height:1.5}</style><h1>Исходники и четыре варианта ретуши</h1><p>'+html.escape(note)+'</p><table>'+tr+'</table>'+body,encoding='utf-8')
    print(table);print('Folder:',out)

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT/'server'))
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('out',type=Path);parser.add_argument('--source',type=Path);args=parser.parse_args();build(args.run,args.out,args.source)
