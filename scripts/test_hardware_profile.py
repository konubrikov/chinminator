import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from hardware_profile import choose_profile, DIT4, DIT16, TE4, TE8, TE16
from configure_models import make_workflow, configure

class Profiles(unittest.TestCase):
    def test_failed_download_does_not_replace_active_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            profile=root/'hardware-profile.json'
            workflow=root/'server/workflows/qwen21_installed.json'
            workflow.parent.mkdir(parents=True)
            for path in (profile,workflow):path.write_text('old configuration')
            with patch('configure_models.ROOT',root), \
                 patch('configure_models.detect',return_value=choose_profile('win32',32,8)), \
                 patch('configure_models.model_metadata',return_value={'size':1}), \
                 patch('configure_models.shutil.disk_usage') as disk, \
                 patch('configure_models.download',side_effect=RuntimeError('download failed')):
                disk.return_value.free=100*2**30
                with self.assertRaisesRegex(RuntimeError,'download failed'):configure('standard')
            for path in (profile,workflow):self.assertEqual(path.read_text(),'old configuration')

    def test_q4_keeps_masked_original_latents_and_native_encoder(self):
        for platform,ram,vram in [('win32',32,8),('win32',64,24),('darwin',24,0)]:
            p=choose_profile(platform,ram,vram,'q4')
            self.assertEqual(p['models'][0],DIT4)
            self.assertEqual(p['models'][1],TE8 if platform=='darwin' else TE4)
            graph=make_workflow(p)
            self.assertEqual(graph['3']['class_type'],'UnetLoaderGGUF')
            self.assertNotIn('weight_dtype',graph['3']['inputs'])
            self.assertEqual(graph['9']['inputs']['model'],['8',0])
            self.assertEqual(graph['8']['inputs']['device'],'off')
            self.assertEqual(graph['9']['inputs']['latent_image'],['14',0])
            self.assertEqual(graph['14']['inputs']['samples'],['13',0])
            self.assertEqual(graph['14']['inputs']['mask'],['12',0])
            self.assertEqual(graph['13']['inputs']['pixels'],['1',0])

    def test_standard_workflow_keeps_native_cache(self):
        graph=make_workflow(choose_profile('win32',64,24))
        self.assertEqual(graph['3']['class_type'],'UNETLoader')
        self.assertEqual(graph['9']['inputs']['model'],['8',0])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):choose_profile('win32',64,24,'unknown')

    def test_small_cuda(self):
        p=choose_profile('win32',32,8)
        self.assertEqual(p['models'][1],TE4)
        self.assertEqual((p['auto_max_side'],p['quality_max_side']),(768,1024))
        self.assertEqual(p['text_device'],'cpu')

    def test_larger_cuda_gets_more_precise_models(self):
        self.assertEqual(choose_profile('win32',32,16)['models'][1],TE8)
        self.assertEqual(choose_profile('win32',64,24)['models'][:2],[DIT16,TE16])
        self.assertNotEqual(choose_profile('win32',32,24)['models'][0],DIT16)

    def test_mac_quality_mode_is_available(self):
        p=choose_profile('darwin',24)
        self.assertEqual((p['auto_max_side'],p['quality_max_side']),(768,1024))
        self.assertEqual(p['models'][1],TE8)
        self.assertEqual(choose_profile('darwin',64)['models'][:2],[DIT16,TE16])

    def test_unsupported_resources_fail_before_download(self):
        for platform,ram,vram in [('win32',32,4),('win32',8,16),('darwin',16,0)]:
            with self.assertRaises(RuntimeError):choose_profile(platform,ram,vram)

if __name__=='__main__':unittest.main()
