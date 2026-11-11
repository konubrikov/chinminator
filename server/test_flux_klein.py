import unittest
import numpy as np
from backends.flux_klein_backend import flux_graph,skin_reference

class FluxContracts(unittest.TestCase):
    def test_native_masked_sampling_keeps_reference_and_four_step_schedule(self):
        g=flux_graph('photo.png','mask.png','Edit chin',1280,1312,42)
        self.assertEqual(g['13']['inputs'],{'steps':4,'width':1280,'height':1312})
        self.assertEqual(g['15']['inputs'],{'samples':['6',0],'mask':['14',0]})
        self.assertEqual(g['7']['inputs']['latent'],['6',0])
        self.assertEqual(g['16']['inputs']['latent_image'],['15',0])
        self.assertFalse(any('Scale' in n['class_type'] or 'Resize' in n['class_type'] for n in g.values()))
        self.assertEqual(g['12']['inputs']['noise_seed'],42)

    def test_reference_removes_only_masked_skin_and_keeps_sampler_source(self):
        image=np.full((2,3,3),55,dtype=np.uint8)
        labels=np.array([[2,3,4],[0,1,2]],dtype=np.uint8)
        mask=np.array([[255,255,255],[255,255,0]],dtype=np.uint8)
        ref=skin_reference(image,mask,labels)
        self.assertTrue(np.all(ref[0,:2]==127))
        self.assertTrue(np.all(ref[0,2]==55))
        self.assertTrue(np.all(ref[1]==55))
        self.assertTrue(np.all(image==55))
        g=flux_graph('source.png','mask.png','Edit',1280,1280,42,reference='ref.png')
        self.assertEqual(g['7']['inputs']['latent'],['reference_latent',0])
        self.assertEqual(g['15']['inputs']['samples'],['6',0])
