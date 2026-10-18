"""Bounded preparation actions in the dedicated persistent WebShop runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

from experiments.omr.assets import digest_file
from experiments.omr.preparation import source_receipt


def index_documents(payload):
    native = Path(payload['repo']) / 'agent_system/environments/env_package/webshop/webshop'
    sys.path.insert(0, str(native))
    from web_agent_site.engine.engine import load_products
    random.seed(0)
    paths = payload['binding']['paths']
    products, *_ = load_products(paths['file_path'], paths['attr_path'],
                                human_attr_path=paths['human_attr_path'])
    target = Path(payload['documents'])
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same text construction and normalization as BEACON's converter; only write
    # the selected catalog, instead of four duplicate resources directories.
    with target.open('w', encoding='utf-8') as stream:
        for product in products:
            options = ', and '.join(f"{name}: {', '.join(values)}"
                                    for name, values in product.get('options', {}).items())
            contents = ' '.join([product['Title'], product['Description'],
                                 product['BulletPoints'][0], options]).lower()
            stream.write(json.dumps({'id': product['asin'], 'contents': contents,
                                     'product': product}) + '\n')
    return {'documents': len(products), 'sha256': digest_file(target),
            'pyserini': importlib.metadata.version('pyserini')}


def index_check(payload):
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(payload['binding']['paths']['index_path'])
    try:
        expected = payload['document_count']
        if searcher.num_docs != expected:
            raise RuntimeError(f'Lucene document count mismatch: {searcher.num_docs} != {expected}')
        with Path(payload['documents']).open(encoding='utf-8') as stream:
            ids = [json.loads(line)['id'] for line in stream]
        if any(searcher.doc(identity) is None for identity in ids):
            raise RuntimeError('Lucene index is missing a catalog product')
        return {'documents': searcher.num_docs, 'all_catalog_ids_present': True}
    finally:
        searcher.close()


def ray_imports(payload):
    import ray
    from experiments.omr.preparation import source_receipt
    if ray.is_initialized():
        raise RuntimeError('Preparation must own its private Ray instance')
    ray.init(address='local', num_cpus=2, include_dashboard=False,
             _temp_dir=str(Path(os.environ['TMPDIR']) / 'ray'))
    try:
        remote = ray.remote(num_cpus=1)(source_receipt)
        worker = ray.get(remote.remote(payload['repo'], payload['prefix']), timeout=90)
        return {'driver': source_receipt(payload['repo'], payload['prefix']), 'ray_worker': worker}
    finally:
        ray.shutdown()  # only this private instance; never ray stop or pkill


def model_load(payload):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('WebShop preparation requires its single allocated H200')
    path = payload['model']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').to('cuda').eval()
    inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Reply with one word: ready'}],
        add_generation_prompt=True, tokenize=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        output = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=8,
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    generated = output.shape[-1] - inputs.shape[-1]
    if generated < 1:
        raise RuntimeError('Weight-load fixture generated no tokens')
    return {'weights_loaded': True, 'bf16_flash_attention_generation': True,
            'generated_tokens': generated, 'test_only': True, 'optimizer_update': False}


def monitoring(payload):
    from experiments.omr.monitoring import SwanLabMonitor
    result = []
    for code, name in ((0, 'success'), (1, 'intentional-failure')):
        directory = Path(payload['output_directory']) / ('swanlab-' + name)
        monitor = SwanLabMonitor('beacon-omr-readiness', name, {'test_only': True},
                                 {'logdir': str(directory), 'mode': 'offline', 'resume': 'never'})
        monitor.log({'fixture/value': 1.0}, 0)
        monitor.finish(code, 'test-only intentional failure' if code else None)
        status = json.loads((directory / 'omr-transport.json').read_text())
        if not status.get('run_id') or not status.get('sdk_finish_returned'):
            raise RuntimeError('SwanLab did not return a run identity and explicit finish')
        result.append(status)
    return {'attempts': result, 'online_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['documents', 'index-check', 'dependencies',
                                        'interaction', 'ray-imports', 'model-load', 'monitoring'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    source_receipt(payload['repo'], payload['prefix'])
    if args.action in ('dependencies', 'interaction'):
        from experiments.omr.webshop_probe import dependencies, interaction
        result = (dependencies if args.action == 'dependencies' else interaction)(payload)
        ok = result['all_imports_passed'] if args.action == 'dependencies' else result['interaction_verified']
    else:
        result = {'documents': index_documents, 'index-check': index_check, 'ray-imports': ray_imports,
                  'model-load': model_load, 'monitoring': monitoring}[args.action](payload)
        ok = True
    Path(args.output).write_text(json.dumps({'passed': ok, 'result': result}, indent=2) + '\n')
    if not ok:
        raise RuntimeError(f'{args.action} failed; see its JSON report')


if __name__ == '__main__':
    main()
