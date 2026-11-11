import unittest
import numpy as np
from lighting import match_context_light, match_boundary_light

class LightingContracts(unittest.TestCase):
    def test_boundary_tint_correction_cannot_copy_old_core(self):
        original=np.full((256,256,3),130,np.uint8)
        generated=np.full_like(original,(140,132,125))
        mask=np.zeros((256,256),np.float32)
        mask[20:236,20:236]=0.3
        mask[40:216,40:216]=1
        corrected=match_boundary_light(original,generated,mask)
        original[mask==1]=0
        np.testing.assert_array_equal(corrected,match_boundary_light(original,generated,mask))
        np.testing.assert_array_equal(corrected[mask==0],generated[mask==0])
        ring=mask==0.3
        self.assertLess(np.abs(corrected[ring].astype(float)-130).mean(),1)

    def test_old_skin_cannot_contribute_to_new_skin(self):
        original=np.full((128,128,3),140,np.uint8)
        generated=np.full_like(original,130)
        mask=np.zeros((128,128),np.float32);mask[32:96,32:96]=1
        first=match_context_light(original,generated,mask)
        original[mask>0]=[0,0,0]
        second=match_context_light(original,generated,mask)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[mask==0],generated[mask==0])
        self.assertGreater(float(first[mask>0].mean()),float(generated[mask>0].mean()))

    def test_no_anchor_is_an_explicit_noop(self):
        image=np.full((32,32,3),90,np.uint8)
        np.testing.assert_array_equal(match_context_light(image,image+10,np.ones((32,32))),image+10)
