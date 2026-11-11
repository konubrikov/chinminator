import unittest
from unittest.mock import patch
import numpy as np
from config import QualityParams
from backends.flux3_backend import Flux3Backend,make_graph,DEFAULT_MODELS

class Flux3Contracts(unittest.TestCase):
    def test_graph_is_masked_full_replacement_not_turbo(self):
        g=make_graph('source','mask','remove chin',1280,1280,42,dict(models=DEFAULT_MODELS,text_device='default'))
        self.assertEqual(g['16']['inputs']['steps'],16)
        self.assertEqual(g['16']['inputs']['denoise'],1)
        self.assertEqual(g['16']['inputs']['cfg'],1)
        self.assertEqual(g['16']['inputs']['latent_image'],['15',0])
        self.assertEqual(g['15']['inputs']['samples'],['6',0])
        self.assertEqual(g['15']['inputs']['mask'],['14',0])
        self.assertEqual(g['7']['inputs']['latent'],['6',0])
        self.assertEqual(g['5']['inputs']['text'],['expand',0])
        self.assertIn('remove chin',g['expand']['inputs']['prompt'])

    def test_native_and_detail_preserve_output_dimensions(self):
        original=np.full((96,160,3),80,np.uint8);mask=np.ones((96,160),np.float32)
        for params,size in [(QualityParams(auto_quality=1), (96,160)),(QualityParams(auto_quality=0,steps=40),(160,256))]:
            uploads=[]
            def upload(a,name):uploads.append(a.copy());return name
            with patch('backends.flux3_backend.ComfyUIBackend._upload',side_effect=upload),patch('backends.flux3_backend.post_prompt',return_value='id'),patch('backends.flux3_backend.wait_done',return_value={}),patch('backends.flux3_backend.fetch_image',side_effect=lambda _:np.full((*size,3),120,np.uint8)):
                result=Flux3Backend().inpaint(original,mask,'chin',params,1,42)
            self.assertEqual(result.shape,original.shape)
            self.assertEqual(uploads[0].shape[:2],size)
            self.assertFalse(uploads[1][:16].any())
            self.assertFalse(uploads[1][:,:16].any())
            self.assertTrue(np.all(result==120))

    def test_no_silent_resize_of_unexpected_result(self):
        with patch('backends.flux3_backend.ComfyUIBackend._upload',return_value='file'),patch('backends.flux3_backend.post_prompt',return_value='id'),patch('backends.flux3_backend.wait_done',return_value={}),patch('backends.flux3_backend.fetch_image',return_value=np.zeros((64,64,3),np.uint8)):
            with self.assertRaisesRegex(RuntimeError,'dimensions'):
                Flux3Backend().inpaint(np.zeros((96,96,3),np.uint8),np.ones((96,96),np.float32),'chin',QualityParams(),1,42)
