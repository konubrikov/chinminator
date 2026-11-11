"""Contracts for sharing native Qwen reference conditioning, no GPU required."""
import unittest,json
from pathlib import Path
from backends.comfyui_backend import optimize_qwen_graph
from comfy_nodes.chinminator_qwen import _PositiveOnlyClip,_CaptureVAE

class EncodeContracts(unittest.TestCase):
    def test_graph_shares_exact_native_reference_latent(self):
        path=Path(__file__).parent/'workflows/qwen21_edit_quality.json'
        original=json.loads(path.read_text());graph=optimize_qwen_graph(original,1.0)
        self.assertEqual(graph['6']['class_type'],'ChinminatorQwenCondition')
        self.assertEqual(graph['14']['inputs']['samples'],['6',2])
        self.assertNotIn('13',graph)
        self.assertTrue(graph['6']['inputs']['skip_negative'])
        self.assertFalse(optimize_qwen_graph(original,2.0)['6']['inputs']['skip_negative'])
        self.assertIn('13',original)
        self.assertEqual(graph['9'],original['9'])
    def test_positive_shortcut_skips_only_second_encode(self):
        class Clip:
            def __init__(self):self.tokenizations=0;self.encodes=0
            def tokenize(self,prompt,**kwargs):self.tokenizations+=1;return prompt
            def encode_from_tokens_scheduled(self,tokens):self.encodes+=1;return [tokens]
        clip=Clip();proxy=_PositiveOnlyClip(clip)
        self.assertEqual(proxy.encode_from_tokens_scheduled(proxy.tokenize('positive')),['positive'])
        self.assertEqual(proxy.encode_from_tokens_scheduled(proxy.tokenize('negative')),['positive'])
        self.assertEqual((clip.tokenizations,clip.encodes),(1,1))

class TurboScheduleContracts(unittest.TestCase):
    def test_turbo_uses_explicit_official_schedule_and_preserves_mask(self):
        from backends.comfyui_backend import turbo_qwen_graph,TURBO_SIGMAS
        original=json.loads((Path(__file__).parent/'workflows/qwen21_edit_quality.json').read_text())
        g=turbo_qwen_graph(optimize_qwen_graph(original,1),42)
        self.assertEqual(g['9']['class_type'],'SamplerCustomAdvanced')
        self.assertEqual(g['9']['inputs']['latent_image'],['14',0])
        self.assertEqual(g['8']['inputs']['model'],['turbo_lora',0])
        self.assertEqual(g['turbo_lora']['inputs']['model'],['3',0])
        self.assertEqual(g['turbo_guider']['inputs']['model'],['8',0])
        sigmas=[float(x) for x in TURBO_SIGMAS.split(',')]
        self.assertEqual(len(sigmas),9)
        self.assertEqual((sigmas[0],sigmas[-1]),(1,0))
        self.assertTrue(all(a>b for a,b in zip(sigmas,sigmas[1:])))

class TurboFullContracts(unittest.TestCase):
    def test_full_checkpoint_uses_no_lora_and_same_native_mask(self):
        from backends.comfyui_backend import turbo_qwen_graph
        original=json.loads((Path(__file__).parent/'workflows/qwen21_edit_quality.json').read_text())
        g=turbo_qwen_graph(optimize_qwen_graph(original,1),42,full_model=True)
        self.assertNotIn('turbo_lora',g)
        self.assertEqual(g['3']['inputs']['unet_name'],'qwen_image_2.1_turbo_int8_convrot.safetensors')
        self.assertEqual(g['8']['inputs']['model'],['3',0])
        self.assertEqual(g['9']['inputs']['latent_image'],['14',0])
