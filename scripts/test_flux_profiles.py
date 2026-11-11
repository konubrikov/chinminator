import unittest
from hardware_profile import choose_profile
from model_sources import model_metadata,FLUX3
from unittest.mock import patch
import one_click

class FluxProfiles(unittest.TestCase):
    def test_memory_profiles_have_pinned_files(self):
        for platform,ram,vram in [('win32',16,8),('win32',64,16),('win32',64,24),('darwin',24,0),('darwin',64,0)]:
            profile=choose_profile(platform,ram,vram,engine='flux3')
            self.assertEqual(profile['engine'],'flux3')
            self.assertEqual(len(profile['models']),4)
            for name in profile['models']:
                self.assertIn(name,FLUX3['files']);self.assertEqual(len(model_metadata(name)['lfs']['oid']),64)
            if ram>=60 and (platform=='darwin' or vram>=24):self.assertIn('bf16',profile['models'][0])
            else:self.assertIn('int8',profile['models'][0])
    def test_mac_avoids_cuda_w4a8_prompt_encoder(self):
        profile=choose_profile('darwin',24,engine='flux3')
        self.assertIn('int8_convrot',profile['models'][3])
        self.assertEqual(profile['text_device'],'cpu')
    def test_flux_q4_rejected(self):
        with self.assertRaises(ValueError):choose_profile('win32',64,16,'q4','flux3')
    def test_cannot_stop_active_retouch(self):
        with patch('one_click.read_url',return_value={'busy':True}),patch('psutil.Process') as process:
            with self.assertRaisesRegex(RuntimeError,'Retouch'):one_click.stop_managed()
            process.assert_not_called()
    def test_cannot_stop_active_comfy_queue(self):
        with patch('one_click.read_url',side_effect=[{}, {'queue_running':[1]}]),patch('psutil.Process') as process:
            with self.assertRaisesRegex(RuntimeError,'active jobs'):one_click.stop_managed()
            process.assert_not_called()
