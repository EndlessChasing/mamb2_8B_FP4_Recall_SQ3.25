"""Frozen FP4 weight-only references; no packed-resident GEMM or file package.

E2M1 codes 0..7 are [0,.5,1,1.5,2,3,4,6], bit3 is sign. Assignment
uses nearest-even code LSB and canonical positive zero. Decoding accepts -0.
G64 stores one FP16 scale. G16 stores E4M3FN block scales and one FP32
whole-matrix scale. G16 assignment divides by FP32(block*global), whereas
the protocol deliberately defines decode as FP32(FP32(code*block)*global).

All operations below are eager PyTorch operations: FP32 subtraction, square,
then a fixed adjacent-pair FP32 SSE tree within each block. Masked padding
contributes zero. A nonfinite unselected candidate receives infinite SSE;
minimum finite SSE across the fixed eleven candidates wins, exact ties keep
the earlier candidate. Fail if no finite candidate exists. Nonfinite source,
selected scales and selected decoded weights always fail. This is selection
within one fixed recipe, never a fallback to another representation.

Quantized chunks are actually nibble-packed and decoded before assigning
native FP16 parameters. Receipts hash these ephemeral buffers; no on-disk
checkpoint export, NVFP4 kernel performance or low resident-VRAM claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from . import runtime

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SHA='ecb0e811798c4da4c8e2a3d82fc3326d8abba3904bf920db13812384e3a5ec10'
FORMATS=('fp4_g64_f16','fp4_g16_e4m3')
GROUP_SIZES=dict(zip(FORMATS,(64,16)))
MULTIPLIERS=(1.,.99,.98,.97,.96,.95,.94,.92,.90,1.25,1.5)
CLIPPING_FACTORS=MULTIPLIERS
E2M1_MAGNITUDES=(0.,.5,1.,1.5,2.,3.,4.,6.)
DEFAULT_CHUNK_ROWS=512
MAX_CHUNK_ELEMENTS=4_194_304
RECEIPT_FORMAT='MAMBA2_FP4_WEIGHT_REFERENCE_V1'
EVIDENCE_FORMAT='MAMBA2_FP4_WEIGHT_SAMPLES_V1'
PARAMETER_COUNT=8_236_999_680
TENSOR_COUNT=507
FP4_TENSOR_COUNT=114


def _need(condition,message):
    if not condition:raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(tensor):
    _need(sys.byteorder=='little','Little-endian host required')
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _format(format_name):
    _need(format_name in FORMATS,'Unknown frozen FP4 format')
    return GROUP_SIZES[format_name]


def recipe(format_name):
    group=_format(format_name)
    return dict(format_name=format_name,code_bits=4,code_order='low_nibble_first',
        magnitudes=list(E2M1_MAGNITUDES),rounding='nearest_ties_even_code_lsb',encoded_zero='positive',
        group_size=group,group_axis='last_input_axis',multipliers=list(MULTIPLIERS),
        scale_dtype='float16' if group==64 else 'float8_e4m3fn',
        global_scale_dtype=None if group==64 else 'float32',
        global_scale_rule=None if group==64 else 'FP32(source_FP16_tensor_absmax/(448*6)); zero_tensor=1',
        comparison_dtype='float32',source_reference_dtype='float16',decoded_dtype='float16',
        scale_selection='minimum_finite_actual_FP16_decode_SSE_first_multiplier_tie',
        group_sse='FP32_difference_then_FP32_square_then_fixed_adjacent_pair_FP32_tree',
        assignment='FP32(source/stored_scale)' if group==64 else 'FP32(source/FP32(stored_block*stored_global))',
        decode='FP16(FP32(code*stored_scale))' if group==64 else 'FP16(FP32(FP32(code*stored_block)*stored_global))',
        training_data_used=False,language_data_used=False,offset=False,
        representation='E2M1_G64_FP16scale' if group==64 else 'NVFP4_scaling_hierarchy_with_weight_only_MSE_range_search')


def _encode_e2m1_finite(values):
    magnitude=values.abs()
    boundaries=torch.tensor((.25,.75,1.25,1.75,2.5,3.5,5.),device=values.device,dtype=torch.float32)
    code=torch.bucketize(magnitude.contiguous(),boundaries,right=False)
    tie=(code<7)&(magnitude==boundaries[code.clamp(max=6)])&((code&1)!=0)
    code=(code+tie.to(torch.int64)).to(torch.uint8)
    return code | (((values<0)&(code!=0)).to(torch.uint8)<<3)


def encode_e2m1(values):
    _need(isinstance(values,torch.Tensor) and values.dtype==torch.float32,'E2M1 assignment requires FP32 values')
    _need(bool(torch.isfinite(values).all()),'Nonfinite E2M1 input')
    return _encode_e2m1_finite(values)


def _decode_e2m1_codes(codes):
    levels=torch.tensor(E2M1_MAGNITUDES,device=codes.device,dtype=torch.float32)
    sign=(1-2*((codes&8)!=0).to(torch.int8)).float()
    return levels[(codes&7).long()]*sign


def decode_e2m1(codes):
    _need(isinstance(codes,torch.Tensor) and codes.dtype==torch.uint8,'E2M1 codes must be uint8')
    _need(bool((codes<=15).all()),'E2M1 nibble outside [0,15]')
    return _decode_e2m1_codes(codes)


def encode_e4m3_scales(values):
    _need(isinstance(values,torch.Tensor) and values.dtype==torch.float32,'Scale conversion requires FP32')
    _need(bool(torch.isfinite(values).all()) and bool((values>=0).all()),'Scale must be finite and nonnegative')
    return values.clamp(0,448).to(torch.float8_e4m3fn)


def pack_nibbles(codes):
    _need(codes.dtype==torch.uint8 and codes.ndim==2,'Codes must be uint8 matrix')
    _need(bool((codes<=15).all()),'Code outside nibble range')
    if codes.shape[1]%2:codes=torch.nn.functional.pad(codes,(0,1))
    return (codes[:,0::2] | (codes[:,1::2]<<4)).contiguous()


def unpack_nibbles(packed,columns):
    _need(packed.dtype==torch.uint8 and packed.ndim==2,'Packed values must be uint8 matrix')
    _need(type(columns) is int and columns>0 and packed.shape[1]==(columns+1)//2,'Packed width differs')
    if columns%2:_need(bool(((packed[:,-1]&0xf0)==0).all()),'Nonzero high-nibble padding')
    result=torch.stack((packed&15,packed>>4),dim=-1).flatten(-2)
    return result[:,:columns].contiguous()


def _pair_sum(values):
    width=values.shape[-1]
    _need(width in (16,64),'Fixed SSE tree requires 16 or 64 lanes')
    while width>1:
        values=values.reshape(*values.shape[:-1],width//2,2)
        values=values[...,0]+values[...,1]
        width//=2
    return values.squeeze(-1)


def _global(value,device):
    _need(isinstance(value,torch.Tensor) and value.dtype==torch.float32 and value.numel()==1,
          'A stored whole-tensor FP32 scale is required')
    value=value.to(device=device).reshape(())
    _need(bool(torch.isfinite(value)) and bool(value>0),'Global scale must be finite and positive')
    return value


def _decode_grouped(codes,scales,format_name,global_scale):
    values=_decode_e2m1_codes(codes)*scales.float().unsqueeze(-1)
    if format_name==FORMATS[1]:values=values*global_scale
    return values.half()


@torch.no_grad()
def quantize_chunk(weights,format_name,tensor_global_scale=None):
    """Return actual packed buffers plus ephemeral selection evidence.

    `packed`, `scales`, and (G16 only) `global_scale` define the representation.
    `candidate_index` and `group_sse` are audit workspace, never weight payload.
    Padding is stored through the full final group and must contain zero codes.
    """
    group=_format(format_name)
    _need(weights.ndim==2 and weights.is_floating_point() and weights.numel()>0,'Nonempty floating matrix required')
    _need(weights.numel()<=MAX_CHUNK_ELEMENTS,'Chunk exceeds bounded workspace')
    reference=weights.half()
    _need(bool(torch.isfinite(reference).all()),'Source contains nonfinite or FP16-overflowing weight')
    rows,columns=reference.shape;groups=(columns+group-1)//group;padded=groups*group
    x=torch.nn.functional.pad(reference.float(),(0,padded-columns)).reshape(rows,groups,group)
    valid=(torch.arange(padded,device=x.device)<columns).reshape(1,groups,group)
    maximum=x.abs().amax(-1)
    global_scale=None
    if group==16:global_scale=_global(tensor_global_scale,x.device)
    else:_need(tensor_global_scale is None,'G64 does not carry a tensor scale')
    dtype=torch.float16 if group==64 else torch.float8_e4m3fn
    scale_bytes=2 if group==64 else 1
    best_error=torch.full_like(maximum,math.inf)
    best_bytes=torch.zeros((*maximum.shape,scale_bytes),dtype=torch.uint8,device=x.device)
    best_codes=torch.zeros_like(x,dtype=torch.uint8)
    best_index=torch.zeros_like(maximum,dtype=torch.uint8)
    reference_error=None;invalid_counts=[]
    for index,factor in enumerate(MULTIPLIERS):
        numerator=maximum*factor
        if group==64:scale=(numerator/6.).half()
        else:scale=(numerator/(global_scale*6.)).clamp(0,448).to(torch.float8_e4m3fn)
        effective=scale.float() if group==64 else scale.float()*global_scale
        zero=effective==0
        denominator=torch.where(zero,torch.ones_like(effective),effective)
        codes=_encode_e2m1_finite(x/denominator.unsqueeze(-1))
        codes=torch.where(zero.unsqueeze(-1)|~valid,0,codes)
        decoded=_decode_grouped(codes,scale,format_name,global_scale).float()
        difference=decoded-x;squared=difference*difference
        error=_pair_sum(torch.where(valid,squared,0.))
        finite=torch.isfinite(error)&torch.isfinite(effective)&(effective>=0)
        error=torch.where(finite,error,math.inf)
        if reference_error is None:reference_error=error
        invalid_counts.append(int((~finite).sum().item()))
        better=error<best_error
        best_error=torch.where(better,error,best_error)
        raw=scale.contiguous().view(torch.uint8).reshape(rows,groups,scale_bytes)
        best_bytes=torch.where(better.unsqueeze(-1),raw,best_bytes)
        best_codes=torch.where(better.unsqueeze(-1),codes,best_codes)
        best_index=torch.where(better,index,best_index)
    _need(bool(torch.isfinite(best_error).all()),'No finite FP4 candidate for at least one block')
    scales=best_bytes.contiguous().view(dtype).reshape(rows,groups)
    packed=pack_nibbles(best_codes.reshape(rows,padded))
    restored=unpack_nibbles(packed,padded)
    _need(torch.equal(restored,best_codes.reshape(rows,padded)),'Actual nibble roundtrip changed selected codes')
    stats=dict(squared_error=float(best_error.double().sum().item()),weight_count=rows*columns,
        group_count=rows*groups,candidate_group_counts=torch.bincount(best_index.long().flatten(),minlength=len(MULTIPLIERS)).cpu().tolist(),
        unclipped_squared_error=float(reference_error.double().sum().item()) if bool(torch.isfinite(reference_error).all()) else None,
        invalid_candidate_group_counts=invalid_counts)
    packet=dict(format=format_name,shape=[rows,columns],group_size=group,packed=packed,scales=scales,
        global_scale=global_scale,candidate_index=best_index,group_sse=best_error,statistics=stats)
    decoded=decode_packet(packet)
    _need(bool(torch.isfinite(decoded).all()),'Selected FP4 decode is nonfinite')
    return packet


quantize_groups=quantize_chunk


@torch.no_grad()
def decode_packet(packet):
    format_name=packet['format'];group=_format(format_name)
    shape=packet['shape']
    _need(isinstance(shape,(tuple,list)) and len(shape)==2 and all(type(v) is int and v>0 for v in shape),'Invalid packet shape')
    rows,columns=shape;groups=(columns+group-1)//group;padded=groups*group
    _need(packet['group_size']==group,'Group size differs from fixed format')
    packed,scales=packet['packed'],packet['scales']
    _need(packed.dtype==torch.uint8 and tuple(packed.shape)==(rows,padded//2),'Packed geometry differs')
    dtype=torch.float16 if group==64 else torch.float8_e4m3fn
    _need(scales.dtype==dtype and tuple(scales.shape)==(rows,groups) and scales.device==packed.device,'Stored scale geometry/dtype differs')
    _need(bool(torch.isfinite(scales.float()).all()) and bool((scales.float()>=0).all()),'Invalid stored scale')
    _need(packed.is_contiguous() and scales.is_contiguous(),'Exact contiguous packed buffers required')
    for value in (packed,scales):
        _need(value.storage_offset()==0 and value.untyped_storage().nbytes()==value.numel()*value.element_size(),
              'Padded/aliased backing storage is not a canonical packet')
    global_scale=None
    if group==16:global_scale=_global(packet['global_scale'],packed.device)
    else:_need(packet['global_scale'] is None,'Unexpected global scale for G64')
    codes=unpack_nibbles(packed,padded)
    if padded!=columns:_need(bool((codes[:,columns:]==0).all()),'Nonzero final-group padding')
    grouped=codes.reshape(rows,groups,group)
    _need(bool(((scales.float()!=0).unsqueeze(-1)|(grouped==0)|(grouped==8)).all()),'Nonzero code in a zero-scale group')
    decoded=_decode_grouped(grouped,scales,format_name,global_scale).reshape(rows,padded)[:,:columns].contiguous()
    _need(bool(torch.isfinite(decoded).all()),'Nonfinite decoded FP16 weights')
    return decoded


def _rows(columns,chunk_rows):
    _need(type(chunk_rows) is int and chunk_rows>0,'Positive integer chunk_rows required')
    _need(columns<=MAX_CHUNK_ELEMENTS,'One row exceeds bounded workspace')
    return min(chunk_rows,max(1,MAX_CHUNK_ELEMENTS//columns))


@torch.no_grad()
def tensor_global_scale(source,chunk_rows=DEFAULT_CHUNK_ROWS):
    """Whole source tensor absmax; bounded CPU pass, never per-chunk globals."""
    _need(source.ndim==2 and source.is_floating_point() and source.numel()>0,'Floating source matrix required')
    step=_rows(source.shape[1],chunk_rows);maximum=0.
    for start in range(0,source.shape[0],step):
        part=source[start:start+step].half()
        _need(bool(torch.isfinite(part).all()),'Invalid source FP16 weight')
        maximum=max(maximum,float(part.abs().max().item()))
    value=torch.tensor(maximum,dtype=torch.float32)/2688. if maximum else torch.tensor(1.,dtype=torch.float32)
    return value


def _is_fp4(name):
    import re
    return name in ('backbone.embedding.weight','lm_head.weight') or bool(re.fullmatch(
        r'backbone\.layers\.([0-9]|[1-4][0-9]|5[0-5])\.mixer\.(in_proj|out_proj)\.weight',name))


def _sample_indices(total):
    return sorted({0,min(1,total-1),total//2,max(0,total-2),total-1})


@torch.no_grad()
def load_fp4_model(source_dir,format_name,*,device='cuda',chunk_rows=DEFAULT_CHUNK_ROWS,evidence_path=None,progress=None):
    """Load native FP16 weights through actual ephemeral packed FP4 buffers.

    Five fixed global block indices per quantized matrix are retained when an
    evidence_path is supplied. All packed payload bytes are hashed, but only
    those bounded samples are saved. The source checkpoint is verified by the
    unchanged safe mmap loader. No former INT4 values enter this conversion.
    """
    group=_format(format_name)
    _need(_sha(ROOT/'docs/FP4_WEIGHT_V1_PROTOCOL.md')==PROTOCOL_SHA,'Frozen FP4 protocol changed')
    if evidence_path is not None:
        evidence_path=Path(evidence_path)
        _need(not evidence_path.exists() and not evidence_path.is_symlink(),'Fresh packed-evidence path required')
    source_state=runtime.load_source_state(source_dir)
    model=runtime.make_model(device='meta',dtype=torch.float16)
    expected={name:tuple(value.shape) for name,value in model.state_dict().items()}
    _need(set(source_state)==set(expected) and len(expected)==TENSOR_COUNT,'Exact 507 source tensors required')
    _need(sum(source_state[n].numel() for n in source_state)==PARAMETER_COUNT,'Parameter coverage differs')
    tensors={};samples=[];fp4_count=0;payload_total=0
    for ordinal,name in enumerate(sorted(expected)):
        source=source_state[name]
        _need(tuple(source.shape)==expected[name],'Source tensor shape differs: '+name)
        destination=torch.empty(source.shape,device=device,dtype=torch.float16)
        source_digest=hashlib.sha256();decoded_digest=hashlib.sha256()
        item=dict(kind=format_name if _is_fp4(name) else 'fp16',shape=list(source.shape),numel=source.numel())
        if not _is_fp4(name):
            flat=source.reshape(-1);target=destination.reshape(-1)
            for start in range(0,flat.numel(),MAX_CHUNK_ELEMENTS):
                part=flat[start:start+MAX_CHUNK_ELEMENTS].half()
                _need(bool(torch.isfinite(part).all()),'Nonfinite retained FP16 weight')
                data=_bytes(part);source_digest.update(data);decoded_digest.update(data)
                target[start:start+part.numel()].copy_(part)
            item.update(payload_bytes=source.numel()*2,source_fp16_sha256=source_digest.hexdigest(),decoded_sha256=decoded_digest.hexdigest())
        else:
            fp4_count+=1
            rows,columns=source.shape
            _need(columns%group==0,'Production source width must be divisible by group size')
            step=_rows(columns,chunk_rows);per_row=columns//group
            global_cpu=tensor_global_scale(source,step) if group==16 else None
            global_device=global_cpu.to(device=device) if global_cpu is not None else None
            code_digest=hashlib.sha256();scale_digest=hashlib.sha256()
            stats=dict(squared_error=0.,unclipped_squared_error=0.,group_count=0,
                candidate_group_counts=[0]*len(MULTIPLIERS),invalid_candidate_group_counts=[0]*len(MULTIPLIERS))
            code_bytes=0;scale_bytes=0;wanted=_sample_indices(rows*per_row);selected=[]
            for start in range(0,rows,step):
                reference=source[start:start+step].half()
                source_digest.update(_bytes(reference))
                packet=quantize_chunk(reference.to(device=device),format_name,global_device)
                decoded=decode_packet(packet)
                destination[start:start+decoded.shape[0]].copy_(decoded)
                decoded_digest.update(_bytes(decoded))
                code_digest.update(_bytes(packet['packed']));scale_digest.update(_bytes(packet['scales']))
                code_bytes+=packet['packed'].untyped_storage().nbytes()
                scale_bytes+=packet['scales'].untyped_storage().nbytes()
                for key in ('squared_error','group_count'):stats[key]+=packet['statistics'][key]
                if packet['statistics']['unclipped_squared_error'] is None:stats['unclipped_squared_error']=None
                elif stats['unclipped_squared_error'] is not None:stats['unclipped_squared_error']+=packet['statistics']['unclipped_squared_error']
                for key in ('candidate_group_counts','invalid_candidate_group_counts'):
                    stats[key]=[a+b for a,b in zip(stats[key],packet['statistics'][key])]
                for index in wanted:
                    local=index-start*per_row
                    if 0<=local<reference.shape[0]*per_row:
                        selected.append(dict(index=index,source=reference.reshape(-1,group)[local].cpu().clone(),
                            packed=packet['packed'].reshape(-1,group//2)[local].cpu().clone(),
                            scale_bytes=packet['scales'].contiguous().view(torch.uint8).reshape(reference.shape[0]*per_row,-1)[local].cpu().clone(),
                            candidate_index=packet['candidate_index'].reshape(-1)[local].cpu().clone(),
                            group_sse=packet['group_sse'].reshape(-1)[local].cpu().clone(),
                            decoded=decoded.reshape(-1,group)[local].cpu().clone()))
                del reference,packet,decoded
            global_bytes=0 if global_cpu is None else len(_bytes(global_cpu))
            item.update(source_fp16_sha256=source_digest.hexdigest(),decoded_sha256=decoded_digest.hexdigest(),
                codes_sha256=code_digest.hexdigest(),scales_sha256=scale_digest.hexdigest(),
                global_scale_sha256=None if global_cpu is None else hashlib.sha256(_bytes(global_cpu)).hexdigest(),
                global_scale=None if global_cpu is None else float(global_cpu.item()),
                codes_bytes=code_bytes,scales_bytes=scale_bytes,global_scale_bytes=global_bytes,
                payload_bytes=code_bytes+scale_bytes+global_bytes,packed_roundtrip_bitwise=True,**stats)
            _need([part['index'] for part in selected]==wanted,'Deterministic sample coverage differs')
            sample=dict(name=name,format=format_name,shape=list(source.shape),group_size=group,
                group_indices=torch.tensor(wanted,dtype=torch.int64),global_scale=global_cpu,
                **{key:torch.stack([part[key] for part in selected]) for key in
                   ('source','packed','scale_bytes','candidate_index','group_sse','decoded')})
            samples.append(sample)
        payload_total+=item['payload_bytes'];tensors[name]=item
        module_name,local_name=name.rsplit('.',1);module=model.get_submodule(module_name)
        if local_name in module._parameters:setattr(module,local_name,nn.Parameter(destination,requires_grad=False))
        elif local_name in module._buffers:module._buffers[local_name]=destination
        else:raise ValueError('Unknown native parameter/buffer: '+name)
        if progress:progress(dict(tensor=name,index=ordinal+1,total=TENSOR_COUNT,kind=item['kind'],
                                  payload_bytes=item['payload_bytes'],decoded_sha256=item['decoded_sha256']))
    del source_state
    _need(fp4_count==FP4_TENSOR_COUNT and not any(t.is_meta for t in model.state_dict().values()),'Incomplete FP4 model')
    _need(model.backbone.embedding.weight.data_ptr()!=model.lm_head.weight.data_ptr(),'Expected untied source embedding/head')
    inventory={name:_sha(ROOT/name) for name in ('mamba2_recall/fp4.py','mamba2_recall/runtime.py','docs/FP4_WEIGHT_V1_PROTOCOL.md')}
    evidence=None
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True,exist_ok=True)
        raw=dict(format=EVIDENCE_FORMAT,complete=True,format_name=format_name,protocol_sha256=PROTOCOL_SHA,
            source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,code_sha256=inventory,
            sample_rule='unique sorted global block indices 0,1,total//2,total-2,total-1',samples=samples)
        with evidence_path.open('xb') as stream:torch.save(raw,stream)
        evidence=dict(file=evidence_path.name,sha256=_sha(evidence_path),bytes=evidence_path.stat().st_size,
                      format=EVIDENCE_FORMAT,samples=len(samples))
    expected_payload=(8_233_418_752*17//32+7_161_856 if group==64 else 8_233_418_752*9//16+114*4+7_161_856)
    _need(payload_total==expected_payload,'Measured FP4 payload differs from exact format accounting')
    model._fp4_receipt=dict(format=RECEIPT_FORMAT,complete=True,format_name=format_name,
        protocol_sha256=PROTOCOL_SHA,source_checkpoint_sha256=runtime.SOURCE_CHECKPOINT_SHA256,
        model_config=runtime.MODEL_CONFIG,quantization=recipe(format_name),code_sha256=inventory,
        source_reference_dtype='float16',tensor_count=TENSOR_COUNT,fp4_tensor_count=fp4_count,
        fp16_tensor_count=TENSOR_COUNT-fp4_count,parameter_count=PARAMETER_COUNT,
        logical_encoded_payload_bytes=payload_total,physical_packed_payload_bytes=payload_total,
        resident_weight_bytes=PARAMETER_COUNT*2,resident_weight_dtype='float16',packed_resident_kernel=False,
        serialization='ephemeral_packed_buffers',actual_packed_roundtrip=True,tensors=tensors,evidence=evidence,
        training_data_used=False,language_data_used=False,adapter_used=False,chunk_rows=chunk_rows,
        max_chunk_elements=MAX_CHUNK_ELEMENTS,hash_order='row-major codes and row-major scale bytes separately; one global FP32 scalar',
        payload_scope='Actual ephemeral encoded weights, including retained FP16 tensors; no file headers, manifest or exported checkpoint')
    return model.eval().requires_grad_(False)
