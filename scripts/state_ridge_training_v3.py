"""Exact deployed ridge-state forward; explicit V11 surrogate backward.

The backward estimator is intentionally the legacy32/32/64 live-mask STE,
which recomputes legacy retained carries from the same inputs. It ignores
latent/predictor derivatives and does not claim exact gradients of ridge state.
The saved inputs are compact contiguous copies, as required by Triton backward.
Inference uses the unchanged frozen packed group-ridge codec.
"""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from state_ppl_training_scan_v11 import _StateQuantV11STE
from state_ppl_training_v11 import StateQuantTrainingV11
from fp4_zero_predictor_codec_v1 import allocate_state,scan,layout_descriptor
from mamba2_recall.resurface_native import tensor_hash as tensor_sha


class _RidgeV11Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,dt,A,B,C,D,dt_bias,permutation,basis,scale,predictor,layout,chunk_size):
        if any(v.requires_grad for v in (A,D,dt_bias,basis,scale,predictor)):
            raise ValueError('Base and static ridge parameters must be frozen')
        tensors=tuple(v.contiguous() for v in (x,dt,A,B,C,D,dt_bias,permutation))
        x,dt,A,B,C,D,dt_bias,permutation=tensors
        ctx.save_for_backward(*tensors)
        ctx.chunk_size=chunk_size
        batch,_,heads,dim=x.shape
        state=allocate_state(batch,heads,dim,128,x.device,layout=layout)
        output,_=scan(x,dt,A,B,C,D,dt_bias,state,permutation,basis,scale,predictor,layout=layout)
        return output

    @staticmethod
    def backward(ctx,grad_y):
        # This is the declared surrogate, not the ridge-state adjoint.
        legacy=_StateQuantV11STE.backward(ctx,grad_y)
        return (*legacy[:8],None,None,None,None,legacy[8])


def scan_training(x,dt,A,B,C,D,dt_bias,permutation,basis,scale,predictor,*,layout,chunk_size=32):
    if type(chunk_size) is not int or not 1<=chunk_size<=256:
        raise ValueError('chunk_size must be an integer in1..256')
    return _RidgeV11Surrogate.apply(x,dt,A,B,C,D,dt_bias,permutation,basis,scale,predictor,layout,chunk_size)


class StateRidgeTrainingV3(StateQuantTrainingV11):
    """Stateless whole-example exact ridge forward for external Resurface masters."""
    def __init__(self,model,table,layouts,bases,scales,predictors,*,chunk_size=32):
        super().__init__(model,table,chunk_size=chunk_size)
        if not all(len(x)==56 for x in (layouts,bases,scales,predictors)):
            raise ValueError('56 frozen layers required')
        self.layouts=tuple(layouts)
        self.bases=[];self.scales=[];self.predictors=[]
        for i,layout in enumerate(self.layouts):
            spec=layout_descriptor(layout)
            expected=((8,spec['nzero'],2),(8,2),(8,spec['n8']+spec['n4']+2,spec['nzero']))
            for target,source,shape in zip((self.bases,self.scales,self.predictors),
                   (bases[i],scales[i],predictors[i]),expected):
                value=source.detach().to(self.device,dtype=torch.float16).contiguous()
                if tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('Frozen ridge static geometry/values differ')
                target.append(value)
            if not bool((self.scales[-1]>0).all()):raise ValueError('Positive latent scales required')
        self._statics=tuple((*self.bases,*self.scales,*self.predictors))
        self._static_identity=tuple((id(x),x.data_ptr(),x._version) for x in self._statics)
        self._static_hashes=tuple(tensor_sha(x) for x in self._statics)

    def assert_frozen(self):
        super().assert_frozen()
        if hasattr(self,'_statics'):
            if tuple((id(x),x.data_ptr(),x._version) for x in self._statics)!=self._static_identity or any(
                    x.requires_grad or x.grad is not None for x in self._statics):
                raise RuntimeError('Ridge static parameters changed')
        return True

    def assert_static_values(self):
        self.assert_frozen()
        if tuple(tensor_sha(x) for x in self._statics)!=self._static_hashes:
            raise RuntimeError('Ridge static content changed')
        return True

    def _forward(self,index,mx,u,seqlen,seq_idx,cu_seqlens,inference_params):
        if not self._installed:raise RuntimeError('Training controller closed')
        if any(value is not None for value in (seqlen,seq_idx,cu_seqlens,inference_params)):
            raise ValueError('Complete examples with zero initial history required')
        if (not isinstance(u,torch.Tensor) or u.ndim!=3 or u.dtype!=torch.float16 or
            u.device!=self.device or min(u.shape[:2])<1 or u.shape[-1]!=4096):
            raise ValueError('CUDA FP16[batch,tokens,4096] required')
        batch,length=u.shape[:2]
        projected=mx.in_proj(u)
        z,xbc,dt=torch.split(projected,(8192,10240,128),dim=-1)
        xbc=self._zero_history_convolution(mx,xbc)
        x,bm,cm=torch.split(xbc,(8192,1024,1024),dim=-1)
        x=x.reshape(batch,length,128,64)
        bm=bm.reshape(batch,length,8,128);cm=cm.reshape(batch,length,8,128)
        aa=-torch.exp(mx.A_log.float())
        y=scan_training(x,dt,aa,bm,cm,mx.D,mx.dt_bias,self.permutations[index],
            self.bases[index],self.scales[index],self.predictors[index],
            layout=self.layouts[index],chunk_size=self.chunk_size)
        if y.dtype!=torch.float16 or tuple(y.shape)!=(batch,length,128,64):
            raise RuntimeError('Unexpected training scan output')
        y=mx.norm(y.reshape(batch,length,8192),z)
        return mx.out_proj(y)
