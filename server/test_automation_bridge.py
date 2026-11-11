import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
import cv2
import numpy as np
from config import QualityParams, Thresholds
from pipeline import PipelineResult
from automation_bridge import process_job
from edit_qwen import _context_crop
from fastapi.testclient import TestClient
from app import app

class AutomationContracts(unittest.TestCase):
    def test_larger_context_preserves_native_region_and_clips_at_frame(self):
        self.assertEqual(_context_crop((1500,1500,2000,2000),5000,5000),(1000,1000,2500,2500))
        x1,y1,x2,y2=_context_crop((0,0,500,500),1000,800)
        self.assertEqual((x1,y1,x2,y2),(0,0,1000,800))

    def make_job(self, folder):
        image=np.full((32,48,3),[10,20,30],np.uint8)
        cv2.imwrite(str(folder/'input.png'),image)
        (folder/'settings.json').write_text(json.dumps(dict(mode='forced',
            thresholds=asdict(Thresholds()), quality=asdict(QualityParams()),prompt='snapshot prompt')))
        return image

    def test_job_returns_exact_pixels_and_uses_settings_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); image=self.make_job(folder)
            with patch('automation_bridge.process_image',return_value=PipelineResult(image,1,edited=True)) as run:
                process_job(folder)
            self.assertEqual((folder/'done.txt').read_text(),'edited')
            self.assertFalse((folder/'error.txt').exists())
            np.testing.assert_array_equal(cv2.imread(str(folder/'output.png')),image)
            self.assertEqual(run.call_args.kwargs['prompt'],'snapshot prompt')
            self.assertEqual(run.call_args.kwargs['quality'].native_resolution,1)

    def test_skip_is_explicit_and_failure_never_marks_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); self.make_job(folder)
            with patch('automation_bridge.process_image',return_value=PipelineResult(None,0,skipped_reason='no face')):
                process_job(folder)
            self.assertEqual((folder/'done.txt').read_text(),'skipped')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); self.make_job(folder)
            with patch('automation_bridge.process_image',side_effect=RuntimeError('GPU failed')):
                process_job(folder)
            self.assertFalse((folder/'done.txt').exists())
            self.assertIn('GPU failed',(folder/'error.txt').read_text())

class AutomationSettingsAPI(unittest.TestCase):
    def test_api_saves_validated_snapshot_and_rejects_bad_settings(self):
        client=TestClient(app)
        with patch('app.save_settings') as save:
            response=client.post('/automation/settings',json={'mode':'forced','params':{'prompt':'new prompt','quality':{'steps':40,'native_resolution':1,'auto_quality':0}}})
            self.assertEqual(response.status_code,200)
            mode,thresholds,quality,prompt=save.call_args.args
            self.assertEqual((mode,quality.steps,quality.native_resolution,prompt),('forced',40,1,'new prompt'))
            save.reset_mock()
            self.assertEqual(client.post('/automation/settings',json={'mode':'invalid'}).status_code,400)
            self.assertEqual(client.post('/automation/settings',json={'params':{'quality':{'native_resolution':2}}}).status_code,400)
            save.assert_not_called()

class AutomationReuseContracts(unittest.TestCase):
    def test_formats_share_render_but_changed_document_pixels_settings_do_not(self):
        import automation_bridge as bridge
        bridge._LAST_RENDER=None
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('automation_bridge.process_image',return_value=PipelineResult(np.full((32,48,3),100,np.uint8),1,edited=True)) as run:
                for i in range(5):
                    folder=root/str(i);folder.mkdir()
                    image=np.full((32,48,3),10+(i==3),np.uint8)
                    cv2.imwrite(str(folder/'input.png'),image)
                    settings=dict(mode='forced',thresholds=asdict(Thresholds()),quality=asdict(QualityParams()),prompt='changed' if i==4 else 'same')
                    (folder/'settings.json').write_text(json.dumps(settings))
                    (folder/'document-id.txt').write_text('2' if i>=2 else '1')
                    process_job(folder)
                    self.assertEqual((folder/'done.txt').read_text(),'edited')
                    self.assertEqual(json.loads((folder/'result.json').read_text())['reused'],i==1)
                self.assertEqual(run.call_count,4)
        bridge._LAST_RENDER=None
