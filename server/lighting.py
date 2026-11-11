"""Match generated skin exposure/chroma to stable context, without copying texture."""
import cv2
import numpy as np


def match_boundary_light(original, generated, mask):
    """Fit only a smooth colour field from the outer feather, never old core skin.

    A low-order field cannot transfer pores, folds or other source texture.
    Correction decays inward to preserve the newly generated shading.
    """
    source=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    target=cv2.cvtColor(generated.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    delta=source-target
    valid=(mask>0.05)&(mask<0.5)
    valid &= (np.abs(delta[...,0])<18)&(np.linalg.norm(delta[...,1:],axis=2)<18)
    y,x=np.nonzero(valid)
    if len(y)<128:return generated.copy()
    h,w=mask.shape
    # Deterministic bounded sample count; no resize of image or generated region.
    stride=max(1,len(y)//12000);y=y[::stride];x=x[::stride]
    design=np.stack((np.ones(len(x)),x/max(1,w-1)*2-1,y/max(1,h-1)*2-1),axis=1)
    values=delta[y,x]
    coeff=np.linalg.lstsq(design,values,rcond=None)[0]
    residual=np.linalg.norm(values-design@coeff,axis=1)
    keep=residual<=np.percentile(residual,75)
    coeff=np.linalg.lstsq(design[keep],values[keep],rcond=None)[0]
    yy,xx=np.mgrid[:h,:w]
    field=coeff[0]+(xx[...,None]/max(1,w-1)*2-1)*coeff[1]+(yy[...,None]/max(1,h-1)*2-1)*coeff[2]
    field=np.clip(field,[-8,-6,-6],[8,6,6])
    distance=cv2.distanceTransform((mask>=0.5).astype(np.uint8),cv2.DIST_L2,5)
    field*=np.exp(-distance/max(8,min(h,w)*0.08))[...,None]
    corrected=(target+field).astype(np.float32);corrected[...,0]=np.clip(corrected[...,0],0,100)
    out=np.clip(cv2.cvtColor(corrected,cv2.COLOR_LAB2BGR)*255+0.5,0,255).astype(np.uint8)
    out[mask<=0]=generated[mask<=0]
    return out


def match_context_light(original, generated, mask):
    """Estimate a bounded Lab offset outside the editable support.

    No source pixels from the replaced skin contribute to the estimate. A global
    offset preserves the generated shading gradients and cannot restore a fold.
    Insufficient context leaves the generation unchanged.
    """
    support=(mask>0).astype(np.uint8)
    radius=max(3,round(min(mask.shape)*0.025))
    ring=cv2.dilate(support,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))-support
    valid=ring.astype(bool)
    if np.count_nonzero(valid)<128:
        return generated.copy()
    source=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    target=cv2.cvtColor(generated.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    # Reject unrelated/reconstructed context rather than infer light from a changed object.
    difference=source-target
    valid &= (np.abs(difference[...,0])<20) & (np.linalg.norm(difference[...,1:],axis=2)<15)
    if np.count_nonzero(valid)<128:
        return generated.copy()
    delta=np.median(difference[valid],axis=0)
    delta=np.clip(delta,[-6,-3,-3],[6,3,3])
    corrected=(target+delta).astype(np.float32)
    corrected[...,0]=np.clip(corrected[...,0],0,100)
    output=np.clip(cv2.cvtColor(corrected,cv2.COLOR_LAB2BGR)*255+0.5,0,255).astype(np.uint8)
    output[~support.astype(bool)]=generated[~support.astype(bool)]
    return output

