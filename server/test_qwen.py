"""Contract tests; no model inference is simulated as a successful real render."""
import io
import json
import unittest
import urllib.error
from unittest.mock import patch
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app import app
from pipeline import PipelineResult
from config import QualityParams
from edit_qwen import BaseBackend, edit_quality, _resize_for_model
from backends.comfy_common import post_prompt, wait_done

class Dummy(BaseBackend):
    name = 'unit-test'
    def inpaint(self, image, *args):
        return np.full_like(image, 255)

class QwenContracts(unittest.TestCase):
    def test_raw_input_preserves_rgb_and_rejects_invalid_length(self):
        import struct
        from app import _decode
        from fastapi import HTTPException
        data=b"CHINRAW1"+struct.pack("<II",2,1)+bytes([10,20,30,40,50,60])
        np.testing.assert_array_equal(_decode(data),np.array([[[30,20,10],[60,50,40]]],np.uint8))
        with self.assertRaises(HTTPException):_decode(data[:-1])

    def test_native_crop_preserves_pixels_and_only_adds_masked_out_padding(self):
        image=np.random.default_rng(42).integers(0,256,(513,789,3),dtype=np.uint8)
        mask=np.ones(image.shape[:2],np.float32)
        captured={}
        class Identity(BaseBackend):
            def inpaint(self,img,msk,*args):
                captured.update(image=img.copy(),mask=msk.copy())
                return img.copy()
        with patch('edit_qwen._pick_backend',return_value=Identity()), \
             patch.dict('os.environ',{'CHINMINATOR_MAX_SIDE':'32','CHINMINATOR_AUTO_MAX_SIDE':'32'}):
            result=edit_quality(image,mask,QualityParams(),region_bbox=(0,0,789,513))
        self.assertEqual(captured['image'].shape,(544,800,3))
        np.testing.assert_array_equal(captured['image'][:513,:789],image)
        self.assertFalse(captured['mask'][513:,:].any())
        self.assertFalse(captured['mask'][:,789:].any())
        np.testing.assert_array_equal(result.image,image)

    def test_reference_size_matches_qwen_grid(self):
        result = _resize_for_model(np.zeros((513,789,3), np.uint8), 512)
        self.assertLessEqual(max(result.shape[:2]),512)
        self.assertEqual(result.shape[0] % 32,0)
        self.assertEqual(result.shape[1] % 32,0)

    def test_raw_response_uses_rgb_and_dimensions(self):
        bgr = np.full((32, 48, 3), [10, 20, 30], np.uint8)
        encoded = cv2.imencode('.png', bgr)[1].tobytes()
        with patch('app.process_image', return_value=PipelineResult(bgr, 1, edited=True, backend='comfyui-qwen-image-2.1', seed=42)):
            response = TestClient(app).post('/process?format=raw', files={'file':('input.png',encoded,'image/png')})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['x-image-width'],'48')
        self.assertEqual(response.headers['x-image-height'],'32')
        self.assertEqual(response.headers['x-pipeline-edited'],'1')
        self.assertEqual(len(response.content),32*48*3)
        self.assertEqual(response.content[:3],bytes([30,20,10]))

    def test_preserves_unmasked_pixels(self):
        image = np.zeros((128,128,3), np.uint8)
        mask = np.zeros((128,128), np.float32)
        mask[40:80,40:80] = 1
        with patch('edit_qwen._pick_backend', return_value=Dummy()):
            result = edit_quality(image, mask, QualityParams(seed=42))
        np.testing.assert_array_equal(result.image[mask == 0], image[mask == 0])
        self.assertGreater(result.image[mask > 0].mean(), 0)
        self.assertEqual(result.seed,42)

    def test_maximum_correction_retains_feather(self):
        image = np.zeros((128,128,3), np.uint8)
        mask = np.zeros((128,128), np.float32)
        mask[35:95,35:95] = 1
        mask[30:35,35:95] = 0.5
        with patch('edit_qwen._pick_backend', return_value=Dummy()):
            result = edit_quality(image, mask, QualityParams(denoise_strength=0.3), strength_scale=0.1)
        self.assertEqual(int(result.image[60,60,0]), 255)
        self.assertEqual(int(result.image[32,60,0]), 128)
        np.testing.assert_array_equal(result.image[mask == 0], image[mask == 0])

    def test_custom_prompt_reaches_pipeline(self):
        encoded = cv2.imencode('.png', np.zeros((32,32,3),np.uint8))[1].tobytes()
        prompt = 'Smooth the neck naturally. Сохранить текстуру кожи.'
        with patch('app.process_image', return_value=PipelineResult(None,0)) as run:
            r = TestClient(app).post('/process', files={'file':('x.png',encoded,'image/png')}, data={'params':json.dumps({'prompt':prompt})})
        self.assertEqual(r.status_code,200)
        self.assertEqual(run.call_args.kwargs['prompt'],prompt)

    def test_neck_mask_margin_feather_and_rotated_bounds(self):
        from face_detect import Face, SUBMENTAL_ARC
        from geometry import build_mask, _rotator
        points = np.zeros((468,2),np.float32)
        x = np.linspace(110,190,len(SUBMENTAL_ARC))
        y = 200 - 30*((x-150)/40)**2
        points[SUBMENTAL_ARC] = np.column_stack([x,y])
        for angle in (0, 25, -25):
            pts = _rotator(angle,np.array([150,200]))(points)
            mask, bbox = build_mask((400,300),Face(pts,(80,80,220,300),angle,100))
            neck = _rotator(angle,np.array([150,200]))([150,275])[0].round().astype(int)
            self.assertEqual(mask[neck[1],neck[0]],255)
            self.assertGreater(np.count_nonzero((mask > 0) & (mask < 255)),0)
            self.assertEqual(mask[80,150],0)
            yy,xx = np.where(mask > 0)
            self.assertEqual(bbox,(xx.min(),yy.min(),xx.max(),yy.max()))
        mask,_ = build_mask((400,300),Face(points,(80,80,220,300),0,100))
        self.assertGreater(mask[240,97],0)  # outside the neck core, in added margin

    def test_semantic_mask_excludes_clothing_and_disconnected_skin(self):
        from face_detect import Face, SUBMENTAL_ARC
        from geometry import build_mask
        points=np.zeros((468,2),np.float32)
        x=np.linspace(110,190,len(SUBMENTAL_ARC))
        points[SUBMENTAL_ARC]=np.column_stack([x,200-30*((x-150)/40)**2])
        face=Face(points,(80,80,220,300),0,100)
        labels=np.full((400,300),4,np.uint8)
        labels[175:245,120:180]=2
        labels[175:201,120:180]=3
        labels[270:280,195:200]=2  # unrelated isolated skin patch
        mask,_=build_mask(labels.shape,face,skin_labels=labels)
        self.assertEqual(mask[220,150],255)
        self.assertEqual(mask[244,150],255)  # old skin edge must be fully replaced
        self.assertEqual(mask[275,150],0)  # clothing below neck, beyond margin
        self.assertEqual(mask[275,197],0)
        with self.assertRaisesRegex(RuntimeError,'Кожа'):
            build_mask(labels.shape,face,skin_labels=np.full_like(labels,4))

    def test_crop_perimeter_is_original_even_with_full_mask(self):
        image = np.full((128,128,3),40,np.uint8)
        with patch('edit_qwen._pick_backend', return_value=Dummy()):
            result=edit_quality(image,np.ones((128,128),np.float32),QualityParams())
        np.testing.assert_array_equal(result.image[0],image[0])
        np.testing.assert_array_equal(result.image[-1],image[-1])
        np.testing.assert_array_equal(result.image[:,0],image[:,0])
        np.testing.assert_array_equal(result.image[:,-1],image[:,-1])
        self.assertEqual(result.image[64,64,0],255)

    def test_inpainting_sends_original_latent_and_real_mask(self):
        from backends.comfyui_backend import ComfyUIBackend
        image=np.zeros((128,128,3),np.uint8)
        mask=np.zeros((128,128),np.float32);mask[40:80,40:80]=1
        mask[:4,:]=1  # the frame must remain protected even for edge-touching skin
        with patch.object(ComfyUIBackend,'_upload',side_effect=['photo.png','mask.png']) as upload, \
             patch('backends.comfyui_backend.post_prompt',return_value='p') as post, \
             patch('backends.comfyui_backend.wait_done',return_value={}), \
             patch('backends.comfyui_backend.fetch_image',return_value=image), \
             patch('urllib.request.urlopen',return_value=io.BytesIO(b'{}')):
            ComfyUIBackend().inpaint(image,mask,'Keep the frame',QualityParams(),.75,42)
        sent=upload.call_args_list[1].args[0]
        self.assertEqual(sent[60,60],255)
        self.assertEqual(sent[20,20],0)
        self.assertFalse(sent[:16].any())
        self.assertFalse(sent[:,:16].any())
        graph=post.call_args.args[0]
        latent_node=graph[graph['9']['inputs']['latent_image'][0]]
        self.assertEqual(latent_node['class_type'],'SetLatentNoiseMask')
        encode=graph[latent_node['inputs']['samples'][0]]
        self.assertEqual(encode['class_type'],'VAEEncode')
        self.assertEqual(encode['inputs']['pixels'],['1',0])
        mask_node=graph[latent_node['inputs']['mask'][0]]
        self.assertEqual(mask_node['inputs'],{'image':'mask.png','channel':'red'})

    def test_auto_quality_uses_crop_size_and_preserves_manual_settings(self):
        from edit_qwen import select_crop_quality
        with patch.dict('os.environ', {'CHINMINATOR_MAX_SIDE':'4096'}):
            small=select_crop_quality(QualityParams(),700,500)
            medium=select_crop_quality(QualityParams(),1600,1400)
            large=select_crop_quality(QualityParams(),3000,2200)
            manual=select_crop_quality(QualityParams(auto_quality=0,steps=35,max_side=1280),3000,2200)
        self.assertEqual((small.steps,small.max_side),(28,1024))
        self.assertEqual((medium.steps,medium.max_side),(28,1024))
        self.assertEqual((large.steps,large.max_side),(40,1536))
        self.assertEqual((manual.steps,manual.max_side),(35,1280))
        with patch.dict('os.environ', {'CHINMINATOR_MAX_SIDE':'768'}):
            self.assertEqual(select_crop_quality(QualityParams(native_resolution=0),3000,2200).max_side,768)

    def test_explicit_quality_still_overrides_auto(self):
        from app import _parse_params
        _,q,_=_parse_params('{"quality":{"steps":20,"max_side":512}}')
        self.assertEqual(q.auto_quality,0)
        with self.assertRaises(Exception):
            _parse_params('{"quality":{"auto_quality":2}}')

    def test_rejected_workflow_keeps_error_details(self):
        error = urllib.error.HTTPError('http://localhost',400,'bad',{},io.BytesIO(b'{"node_errors":{"6":"bad reference"}}'))
        with patch('urllib.request.urlopen',side_effect=error):
            with self.assertRaisesRegex(RuntimeError,'bad reference'):
                post_prompt({})

    def test_execution_failure_is_not_success(self):
        payload = {'p': {'status': {'status_str':'error','messages':[['execution_error',{'exception_message':'CUDA out of memory'}]]},'outputs':{}}}
        with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(RuntimeError,'CUDA out of memory'):
                wait_done('p')

if __name__ == '__main__':
    unittest.main()
