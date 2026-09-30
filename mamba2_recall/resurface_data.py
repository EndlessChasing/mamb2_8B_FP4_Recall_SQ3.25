"""CPU-only, split-isolated numeric recall data for the proposed readout study.

Importing this module generates no examples and opens no dataset. The training
loader opens only the literal train subdirectory. Confirmation data have a
separate, explicit evaluation loader. Generation is not permission to train or
to inspect confirmation scores; the caller must freeze its protocol first.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

FORMAT = 'MAMBA2_RESURFACE_NUMERIC_DATA_V1'
PUBLIC_EVALUATION_SHA = '95cba0b201e37f92e6cbfadb9b3e9d79ba7b0cdc9efda6a58a6216b0d84d35e0'
TOKENIZER_SHA = '5862e2f71caf762bc9845662be5fec2867deb58d874568235a02a36c5111cd09'
SIZES = (16, 64)
TEMPLATES = (0, 1, 2)
CONTRACT = {
    'train': {'seed':2026092801, 'key_range':[300000,380000],
        'value_ranges':[[681000,690000],[781000,790000],[881000,890000],[981000,990000]],
        'samples_per_cell':256,
        'normal_count':1536, 'removed_count':0, 'positions':'independent uniform rng.randrange(N)'},
    'dev': {'seed':834901, 'key_range':[100000,180000],
        'value_ranges':[[600000,680000]], 'samples_per_cell':64,
        'normal_count':384, 'removed_count':384, 'positions':'floor(sample*(N-1)/63)',
        'control_key_base':190000, 'control_value_base':690000,
        'generator':'frozen public-mamba8-mk-v1 validation, samples_per_cell=64'},
    'confirm': {'seed':2026092802, 'key_range':[400000,480000],
        'value_ranges':[[691000,700000],[791000,800000],[891000,900000],[991000,1000000]],
        'samples_per_cell':64,
        'normal_count':384, 'removed_count':384, 'positions':'floor(sample*(N-1)/63)',
        'control_key_base':490000, 'control_value_base':990000},
}


def in_value_pool(value, spec):
    return any(low<=value<high for low,high in spec['value_ranges'])


def value_pool_statistics(rows, split):
    """Actual normal-row counts; controls are excluded from these draw totals."""
    ranges = CONTRACT[split]['value_ranges']
    normal = [row for row in rows if row['condition']=='normal']
    values = [value for row in normal for _,value in row['records']]
    targets = [int(row['answer']) for row in normal]
    return {'pool_cardinality':sum(high-low for low,high in ranges),
        'normal_binding_value_occurrences':len(values),'normal_target_value_occurrences':len(targets),
        'unique_normal_values':len(set(values)),
        'bands':[{'range_half_open':[low,high],'pool_cardinality':high-low,
            'binding_value_occurrences':sum(low<=value<high for value in values),
            'target_value_occurrences':sum(low<=value<high for value in targets)} for low,high in ranges],
        'sampling':'Uniform without replacement within each prompt over the pooled bands; not exact per-prompt band balancing.',
        'paired_removed_controls_included_in_draw_totals':False}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def token_digest(ids):
    if not isinstance(ids,list) or any(type(x) is not int or not 0 <= x < 256000 for x in ids):
        raise ValueError('Expected a plain list of valid native256K integer token ids')
    return sha_bytes(b''.join(x.to_bytes(8,'little',signed=True) for x in ids))


def render(records, key, template):
    """The three public-v1 surface forms; no answer appended to the prompt."""
    if template == 0:
        body = '\n'.join(f'{k}: {v}' for k,v in records)
        return f'Key-value records:\n{body}\n\nLookup the value for key {key}.\nValue:'
    if template == 1:
        body = '\n'.join(f'Key {k} has value {v}.' for k,v in records)
        return f'{body}\n\nQuestion: What is the value of key {key}?\nAnswer:'
    if template == 2:
        body = '\n'.join(f'{k} -> {v}' for k,v in records)
        return f'Dictionary:\n{body}\n\nReturn the stored value.\n{key} ->'
    raise ValueError('Unsupported public template')


def parse_records(prompt, template):
    if template == 0:
        body = prompt.removeprefix('Key-value records:\n').split('\n\nLookup the value for key ',1)[0]
        pattern = r'([0-9]{6}): ([0-9]{6})'
    elif template == 1:
        body = prompt.split('\n\nQuestion: What is the value of key ',1)[0]
        pattern = r'Key ([0-9]{6}) has value ([0-9]{6})\.'
    elif template == 2:
        body = prompt.removeprefix('Dictionary:\n').split('\n\nReturn the stored value.\n',1)[0]
        pattern = r'([0-9]{6}) -> ([0-9]{6})'
    else:
        raise ValueError('Unsupported public template')
    records = []
    for line in body.splitlines():
        match = re.fullmatch(pattern,line)
        if match is None:
            raise ValueError('Malformed record text')
        records.append([int(match[1]),int(match[2])])
    return records


def _finish_row(row, records, split, sample):
    result = {**row,'split':split,'sample':sample,'records':[list(pair) for pair in records]}
    result['prompt_sha256_utf8'] = sha_bytes(result['prompt'].encode('utf-8'))
    result['row_sha256'] = sha_bytes(canonical_bytes(result))
    return result


def generate_cases(split):
    """Explicit lazy generation; train and confirm are independent RNG streams."""
    if split not in ('train','confirm'):
        raise ValueError('Use frozen_development_cases() for the immutable public development suite')
    spec = CONTRACT[split]
    rng = random.Random(spec['seed'])
    value_pool = [value for low,high in spec['value_ranges'] for value in range(low,high)]
    rows = []
    for size in SIZES:
        for template in TEMPLATES:
            for sample in range(spec['samples_per_cell']):
                keys = rng.sample(range(*spec['key_range']),size)
                values = rng.sample(value_pool,size)
                records = list(zip(keys,values))
                position = (rng.randrange(size) if split=='train'
                    else sample*(size-1)//(spec['samples_per_cell']-1))
                key,value = records[position]
                row = {'id':f'resurface-{split}-n{size}-t{template}-s{sample}',
                    'N':size,'template':template,'query_position':position,'key':key,
                    'answer':str(value),'prompt':render(records,key,template),'condition':'normal'}
                rows.append(_finish_row(row,records,split,sample))
                if split == 'confirm':
                    removed = records.copy()
                    removed[position] = (spec['control_key_base']+sample,spec['control_value_base']+sample)
                    control = {**row,'id':row['id']+'-removed','condition':'target_removed',
                        'prompt':render(removed,key,template)}
                    rows.append(_finish_row(control,removed,split,sample))
    validate_cases(rows,split)
    return rows


def frozen_development_cases():
    """Called by preparation/evaluation only, never by training loaders."""
    path = Path(__file__).with_name('evaluation.py')
    if sha_file(path) != PUBLIC_EVALUATION_SHA:
        raise ValueError('Frozen public-v1 generator changed')
    from .evaluation import synthetic_mk_cases
    rows = []
    for row in synthetic_mk_cases('validation',samples_per_cell=64,sizes=SIZES):
        sample = int(re.search(r'-s(\d+)(?:-removed)?$',row['id'])[1])
        rows.append(_finish_row(row,parse_records(row['prompt'],row['template']),'dev',sample))
    validate_cases(rows,'dev')
    return rows


def validate_cases(rows, split):
    if split not in CONTRACT or not isinstance(rows,list):
        raise ValueError('Unknown split or malformed case list')
    spec = CONTRACT[split]
    if len(rows) != spec['normal_count']+spec['removed_count']:
        raise ValueError('Declared split count differs')
    identifiers, prompts, by_id, cells = set(),set(),{},Counter()
    order = []
    for size in SIZES:
        for template in TEMPLATES:
            for sample in range(spec['samples_per_cell']):
                prefix = 'validation' if split=='dev' else f'resurface-{split}'
                normal = f'{prefix}-n{size}-t{template}-s{sample}'
                order.append(normal)
                if split!='train':
                    order.append(normal+'-removed')
    if [row.get('id') for row in rows] != order:
        raise ValueError('Case ids/order/cell balancing differ')
    for row in rows:
        if row.get('split') != split or row['id'] in identifiers or row['prompt'] in prompts:
            raise ValueError('Wrong split or duplicate case/prompt')
        identifiers.add(row['id']);prompts.add(row['prompt']);by_id[row['id']] = row
        for name in ('N','template','sample','query_position','key'):
            if type(row.get(name)) is not int:
                raise ValueError('Case integers must not be floats or booleans')
        size,template,sample,position = [row[k] for k in ('N','template','sample','query_position')]
        if size not in SIZES or template not in TEMPLATES or not 0<=sample<spec['samples_per_cell'] or not 0<=position<size:
            raise ValueError('Case geometry differs')
        if split!='train' and position != sample*(size-1)//(spec['samples_per_cell']-1):
            raise ValueError('Evaluation query positions must spread across the whole sequence')
        if not isinstance(row.get('answer'),str) or re.fullmatch(r'[0-9]{6}',row['answer']) is None:
            raise ValueError('Expected a six-digit answer string')
        answer = int(row['answer'])
        if not spec['key_range'][0]<=row['key']<spec['key_range'][1] or not in_value_pool(answer,spec):
            raise ValueError('Query/answer split range differs')
        records = row.get('records')
        if (not isinstance(records,list) or len(records)!=size
                or any(not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair) for pair in records)
                or len({k for k,_ in records})!=size or len({v for _,v in records})!=size):
            raise ValueError('Unique integral bindings required')
        condition = row['condition']
        if condition not in ('normal','target_removed') or (split=='train' and condition!='normal'):
            raise ValueError('Training permits normal binding CE only')
        for index,(key,value) in enumerate(records):
            control = condition=='target_removed' and index==position
            if control:
                if [key,value]!=[spec['control_key_base']+sample,spec['control_value_base']+sample]:
                    raise ValueError('Control must replace only the target binding')
            elif not spec['key_range'][0]<=key<spec['key_range'][1] or not in_value_pool(value,spec):
                raise ValueError('Record key/value split range differs')
        if row['prompt'] != render(records,row['key'],template) or parse_records(row['prompt'],template)!=records:
            raise ValueError('Prompt is not the exact reconstruction of declared bindings')
        if row['prompt_sha256_utf8']!=sha_bytes(row['prompt'].encode('utf-8')):
            raise ValueError('Prompt text hash differs')
        content = {k:v for k,v in row.items() if k!='row_sha256'}
        if row['row_sha256']!=sha_bytes(canonical_bytes(content)):
            raise ValueError('Raw row hash differs')
        if condition=='normal':
            if records[position]!=[row['key'],answer]:
                raise ValueError('Queried key is not bound to the exact supervised answer')
        else:
            if row['key'] in {k for k,_ in records} or answer in {v for _,v in records}:
                raise ValueError('Removed binding/answer remains in records')
            if re.search(r'(?<!\d)'+re.escape(row['answer'])+r'(?!\d)',row['prompt']):
                raise ValueError('Removed answer still appears in rendered prompt')
            normal = by_id[row['id'].removesuffix('-removed')]
            if (any(row[k]!=normal[k] for k in ('N','template','sample','query_position','key','answer'))
                    or [i for i,(a,b) in enumerate(zip(normal['records'],records)) if a!=b]!=[position]):
                raise ValueError('Control is not paired to its normal case')
        cells[size,template,condition] += 1
    expected = {(size,template,condition):spec['samples_per_cell'] for size in SIZES for template in TEMPLATES
        for condition in (('normal',) if split=='train' else ('normal','target_removed'))}
    if dict(cells)!=expected:
        raise ValueError('Unbalanced split cells')
    return {'split':split,'normal_count':spec['normal_count'],'removed_count':spec['removed_count'],
        'unique_prompts':len(prompts),'exact_binding_and_control_absence':True,
        'raw_rows_sha256':sha_bytes(b''.join(canonical_bytes(row) for row in rows)),
        'template_scope':'Same three template families; only instances/key-value ranges are separated.'}


def _encode(tokenizer, text):
    ids = list(tokenizer.encode(text))
    token_digest(ids)
    if not ids:
        raise ValueError('Empty encoded sequence')
    return ids


def tokenize_case(row, tokenizer):
    """Use answer suffix labels only for TRAIN, never for control/evaluation rows."""
    prompt_ids = _encode(tokenizer,row['prompt'])
    if len(prompt_ids)+12>4096:
        raise ValueError('Prompt plus frozen12-token generation exceeds native4K context')
    result = {'id':row['id'],'split':row['split'],'raw_row_sha256':row['row_sha256'],
        'prompt_ids':prompt_ids,'prompt_tokens':len(prompt_ids),
        'prompt_token_sha256_int64le':token_digest(prompt_ids)}
    if row['split']=='train':
        if row['condition']!='normal':
            raise ValueError('Removed controls are never supervised training answers')
        full_text = row['prompt']+' '+row['answer']
        full = _encode(tokenizer,full_text)
        length = len(prompt_ids)
        if full[:length]!=prompt_ids or not length<len(full)<=4096:
            raise ValueError('Prompt+answer tokenization is not prefix-stable within native4K context')
        suffix = full[length:]
        if tokenizer.decode(suffix).strip()!=row['answer']:
            raise ValueError('Token suffix does not decode to the whole six-digit answer')
        result.update(full_ids=full,full_text_sha256_utf8=sha_bytes(full_text.encode('utf-8')),
            full_token_sha256_int64le=token_digest(full),answer_ids=suffix,answer_target_count=len(suffix),
            answer_token_positions=list(range(length,len(full))),
            ce_hidden_positions=list(range(length-1,len(full)-1)),
            prefix_stability_verified=True,
            loss_scope='Full256K CE on every answer-suffix token only; no EOS or prompt-token CE.')
    return result


def prepare_split(dataset_root, split, tokenizer, *, protocol_path, protocol_sha256, tokenizer_sha256):
    """Write one explicit new split, only after a caller-pinned protocol exists.

    This function is never called at import. TRAIN preparation does not generate
    or read DEV/CONFIRM; the caller invokes their preparations separately.
    """
    if (split not in CONTRACT or tokenizer_sha256!=TOKENIZER_SHA
            or getattr(tokenizer,'sha256',None)!=TOKENIZER_SHA):
        raise ValueError('Declared split and pinned native tokenizer required')
    protocol_path = Path(protocol_path)
    if (not re.fullmatch(r'[0-9a-f]{64}',protocol_sha256) or not protocol_path.is_file()
            or sha_file(protocol_path)!=protocol_sha256):
        raise ValueError('A frozen existing protocol with exactSHA is required before generation')
    output = Path(dataset_root)/split
    if output.exists() or output.is_symlink():
        raise FileExistsError('Fresh split directory only; preserve existing preparations')
    rows = frozen_development_cases() if split=='dev' else generate_cases(split)
    proof = validate_cases(rows,split)
    tokens = [tokenize_case(row,tokenizer) for row in rows]
    prompt_hashes = [row['prompt_token_sha256_int64le'] for row in tokens]
    if len(set(prompt_hashes))!=len(prompt_hashes):
        raise ValueError('Duplicate prompt tokens within a split')
    raw = b''.join(canonical_bytes(row) for row in rows)
    encoded = b''.join(canonical_bytes(row) for row in tokens)
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw.jsonl').write_bytes(raw)
    (output/'tokens.jsonl').write_bytes(encoded)
    manifest = {'format':FORMAT,'complete':True,'split':split,'contract':CONTRACT[split],
        'sizes':list(SIZES),'templates':list(TEMPLATES),'proof':proof,
        'protocol_sha256':protocol_sha256,'tokenizer_sha256':tokenizer_sha256,
        'helper_source_sha256':sha_file(__file__),'python_version':sys.version,
        'value_pool_statistics':value_pool_statistics(rows,split),
        'public_generator_sha256':PUBLIC_EVALUATION_SHA if split=='dev' else None,
        'files':{'raw.jsonl':{'bytes':len(raw),'sha256':sha_bytes(raw)},
                 'tokens.jsonl':{'bytes':len(encoded),'sha256':sha_bytes(encoded)}},
        'row_count':len(rows),'prompt_tokens_total':sum(t['prompt_tokens'] for t in tokens),
        'prompt_tokens_max':max(t['prompt_tokens'] for t in tokens),
        'answer_target_tokens':sum(t.get('answer_target_count',0) for t in tokens),
        'answer_token_count_histogram':{str(k):v for k,v in sorted(Counter(t['answer_target_count'] for t in tokens if split=='train').items())},
        'automatic_bos_eos':False,'full_vocabulary':256000,
        'evaluation_supervised_targets_stored':False,
        'confirmation_loaded_for_training':False}
    (output/'manifest.json').write_bytes(canonical_bytes(manifest))
    return manifest


def _load_split(dataset_root, split, expected_manifest_sha256, tokenizer):
    if getattr(tokenizer,'sha256',None)!=TOKENIZER_SHA:
        raise ValueError('Actual native tokenizer identity differs')
    directory = Path(dataset_root)/split
    if directory.is_symlink() or {p.name for p in directory.iterdir()}!={'manifest.json','raw.jsonl','tokens.jsonl'}:
        raise ValueError('Expected one complete isolated split directory')
    mp = directory/'manifest.json'
    if mp.is_symlink() or sha_file(mp)!=expected_manifest_sha256:
        raise ValueError('Pinned split manifest differs')
    manifest = json.loads(mp.read_text())
    if (manifest.get('format')!=FORMAT or manifest.get('complete') is not True or manifest.get('split')!=split
            or manifest.get('contract')!=CONTRACT[split] or manifest.get('tokenizer_sha256')!=TOKENIZER_SHA
            or manifest.get('helper_source_sha256')!=sha_file(__file__)
            or set(manifest.get('files',{}))!={'raw.jsonl','tokens.jsonl'}):
        raise ValueError('Prepared split provenance/format differs')
    parsed = {}
    for name,entry in manifest['files'].items():
        path = directory/name
        if (path.is_symlink() or not path.is_file() or type(entry.get('bytes')) is not int
                or path.stat().st_size!=entry['bytes'] or sha_file(path)!=entry['sha256']):
            raise ValueError('Prepared split file identity differs')
        parsed[name] = [json.loads(line) for line in path.read_text().splitlines()]
    rows,tokens = parsed['raw.jsonl'],parsed['tokens.jsonl']
    if validate_cases(rows,split)!=manifest['proof'] or len(tokens)!=len(rows):
        raise ValueError('Prepared raw case proof differs')
    expected_rows = frozen_development_cases() if split=='dev' else generate_cases(split)
    if rows!=expected_rows:
        raise ValueError('Prepared cases differ from the exact frozen seed/generator')
    if manifest.get('value_pool_statistics')!=value_pool_statistics(rows,split):
        raise ValueError('Declared/observed pooled value distribution differs')
    if [tokenize_case(row,tokenizer) for row in rows]!=tokens:
        raise ValueError('Actual tokenizer/prefix/answer-label reconstruction differs')
    return manifest,rows,tokens


def load_training(dataset_root, expected_manifest_sha256, tokenizer):
    """Literal TRAIN-only path; never opens any development or confirmation file."""
    return _load_split(dataset_root,'train',expected_manifest_sha256,tokenizer)


def load_evaluation(dataset_root, split, expected_manifest_sha256, tokenizer):
    if split not in ('dev','confirm'):
        raise ValueError('Explicit dev or confirm evaluation split required')
    return _load_split(dataset_root,split,expected_manifest_sha256,tokenizer)


def audit_split_overlap(*split_rows):
    """Preparation/evaluation audit only; never called by load_training."""
    seen_ids,seen_text,seen_tokens = set(),set(),set()
    for rows,tokens in split_rows:
        if len(rows)!=len(tokens):
            raise ValueError('Raw/token row counts differ')
        current_ids = {row['id'] for row in rows}
        current_text = {row['prompt_sha256_utf8'] for row in rows}
        current_tokens = {row['prompt_token_sha256_int64le'] for row in tokens}
        if seen_ids&current_ids or seen_text&current_text or seen_tokens&current_tokens:
            raise ValueError('Cross-split id/text/token input overlap')
        seen_ids.update(current_ids);seen_text.update(current_text);seen_tokens.update(current_tokens)
    return {'splits':len(split_rows),'unique_cases':len(seen_ids),'id_overlap':0,'text_overlap':0,'token_overlap':0,
        'scope':'Exact prompt identity plus declared disjoint key/value ranges; shared template families are intentional.'}
