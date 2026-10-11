"""Write bounded provenance/integrity records after v17 stopped at M2 parity."""
from pathlib import Path
import hashlib, json, subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
V16=OUT.parent/'range_w4_fullmodel_baseline_v16'

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path): return sha_bytes(Path(path).read_bytes())
def dump(name, obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    # Verify the frozen v16 files against that release's own integrity list.
    expected=json.loads((V16/'ARTIFACT_SHA256.json').read_text(encoding='utf-8'))
    mismatches=[]
    for rel,digest in expected.items():
        path=V16/rel
        actual=sha_file(path) if path.is_file() else None
        if actual!=digest:mismatches.append({'path':rel,'expected':digest,'actual':actual})
    dump('HISTORICAL_INTEGRITY_CHECK.json',{
        'checked_release':'range_w4_fullmodel_baseline_v16',
        'manifest':'ARTIFACT_SHA256.json','files_checked':len(expected),
        'mismatch_count':len(mismatches),'mismatches':mismatches,
        'status':'PASS' if not mismatches else 'FAIL',
        'scope_note':'v8-v15 historical fingerprints were not recomputed in this stopped run; this turn wrote only inside v17.'})

    # Verify official v16 checkout source bytes without touching that checkout.
    upstream_dir=V16/'official_source'
    upstream_hashes=json.loads((V16/'OFFICIAL_SOURCE_SHA256.json').read_text(encoding='utf-8'))
    source_mismatches=[]
    for rel,digest in upstream_hashes.items():
        path=upstream_dir/rel
        actual=sha_file(path) if path.is_file() else None
        if actual!=digest:source_mismatches.append({'path':rel,'expected':digest,'actual':actual})
    clone=OUT/'omniquant_source'
    def git(*args):
        return subprocess.run(['git','-C',str(clone),*args],check=True,capture_output=True,text=True).stdout.strip()
    commit=git('rev-parse','HEAD'); tree=git('rev-parse','HEAD^{tree}')
    branch_source=git('rev-parse','--show-toplevel')
    patch_files=['models/int_qwen2_layer.py','quantize/omniquant.py','utils.py']
    local_patches=[]
    for rel in patch_files:
        path=clone/rel
        local=sha_file(path) if path.is_file() else None
        try: upstream=sha_bytes(subprocess.run(['git','-C',str(clone),'show',f'HEAD:{rel}'],check=True,capture_output=True).stdout)
        except subprocess.CalledProcessError: upstream=None
        local_patches.append({'path':rel,'upstream_commit_blob_sha256':upstream,'local_sha256':local,'differs_from_upstream':upstream!=local})
    status=git('status','--short')
    dump('OFFICIAL_SOURCE_PROVENANCE.json',{
        'repository':'OpenGVLab/OmniQuant','commit':commit,'expected_commit':'feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14',
        'tree_sha1':tree,'v17_clone_path':branch_source,
        'v16_source_files_checked':len(upstream_hashes),'v16_source_hash_mismatch_count':len(source_mismatches),
        'v16_source_hash_mismatches':source_mismatches,'tracked_code_patch_hashes':local_patches,
        'local_git_status':status.splitlines(),
        'windows_case_collision':'The clone reports imgs/OmniQuant.png modified because the upstream repository also has imgs/omniquant.png on a case-insensitive filesystem; no image was intentionally changed.'})

if __name__=='__main__': main()
