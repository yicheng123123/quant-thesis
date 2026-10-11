"""Bounded source/protocol diagnosis only: no repair, no scoring, no PPL."""
import run_v52 as r
from run_v52 import *
import transformers

def main():
 budget.start('bounded actual block0 prefix compatibility diagnosis after correctness stop')
 rt.WALL_START=time.time();rt.elapsed=elapsed;rt.deadline=deadline;rt.MODE='diagnostic';rt.LAYER=0;rt.MODULE='actual_block0_return'
 watcher=threading.Timer(min(90,max(1,CAP-elapsed())),lambda:(budget.finish(),os._exit(124)));watcher.daemon=True;watcher.start()
 try:
  with rt.timer('prefix_diagnostic_locked_model_load'):
   model=AutoModelForCausalLM.from_pretrained(str(OUT/'R1/checkpoint/final'),torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='sdpa').eval().cuda()
  inherited=[]
  for group in GROUP_ORDER:
   a=torch.load(OUT/'formal/groups'/f'layer_00_{group}.pt',map_location='cpu',weights_only=False)
   b=torch.load(BASE/'v51_m2_fast_integer_repair_fullmodel/formal/groups'/f'layer_00_{group}.pt',map_location='cpu',weights_only=False)
   for name,m in a['modules'].items():
    key=f'model.layers.0.{name}.weight';actual=model.get_submodule(key[:-7]).weight.detach().cpu()
    assert torch.equal(m['weight'],b['modules'][name]['weight']) and torch.equal(m['weight'],actual)
    inherited.append(name)
  hidden=rt.embed(model);real=np.empty_like(hidden);wrong=np.empty_like(hidden);shape=None;different_docs=0
  positions=torch.arange(2048,device='cuda')[None]
  with rt.timer('actual_prefix_vs_frozen_indexing_diagnosis'):
   with torch.inference_mode():
    for i in range(128):
     h=torch.from_numpy(hidden[i].copy()).cuda()[None]
     result=model.model.layers[0](h,attention_mask=None,position_ids=positions,cache_position=positions[0],position_embeddings=model.model.rotary_emb(h,positions),use_cache=False)
     assert torch.is_tensor(result);shape=list(result.shape);assert shape==[1,2048,896]
     real[i]=result[0].cpu().numpy()
     wrong[i]=result[0][0].cpu().numpy()
     different_docs+=int(not np.array_equal(real[i],wrong[i]))
  v51prefix=read(BASE/'v51_m2_fast_integer_repair_fullmodel/formal/PREFIX_STATUS_layer_00.json');v52prefix=read('formal/PREFIX_STATUS_layer_00.json')
  rh=xhash(real);wh=xhash(wrong)
  # Both the earlier restored boundary and this complete run recorded the same broadcast boundary.
  assert wh==v51prefix['hidden_sha256']==v52prefix['hidden_sha256'];assert rh!=wh;assert different_docs==128
  with rt.timer('layer1_qkv_source_comparison'):
   realX=ORIGINAL_CAPTURE(model,1,real,'qkv');wrongX=ORIGINAL_CAPTURE(model,1,wrong,'qkv')
  cp=torch.load(OUT/'formal/groups/layer_01_qkv.pt',map_location='cpu',weights_only=False)
  rx=xhash(realX);wx=xhash(wrongX);assert wx==cp['input_sha256'] and rx!=wx
  record={'confirmed_correctness_error':True,'transformers_version':transformers.__version__,'installed_decoder_source_sha256':sha(ROOT/'.venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py'),'actual_layer_output_type':type(result).__name__,'actual_layer_output_shape':shape,'correct_next_block_document_shape':list(real[0].shape),'frozen_indexed_value_shape':list(result[0][0].shape),'broadcast_documents':different_docs,'old_broadcast_prefix_sha256':wh,'true_full_token_prefix_sha256':rh,'matches_v51_and_v52_recorded_boundary':True,'layer1_committed_X_sha256':cp['input_sha256'],'old_broadcast_layer1_X_sha256':wx,'true_layer1_X_sha256':rx,'layer1_and_later_input_source_invalid':True,'inherited_block0_weight_equality_checked':inherited,'no_repair_or_PPL_run_in_diagnosis':True,'no_weights_codes_grid_or_algorithm_changed':True,'probe_raw_timing_file':'STAGE_TIMINGS.csv','shared_charged_seconds_at_completion':elapsed()}
  dump('PREFIX_COMPATIBILITY_DIAGNOSIS.json',record);print('ACTUAL_PREFIX_ERROR_CONFIRMED',json.dumps(record),flush=True)
 finally:watcher.cancel();budget.finish()
if __name__=='__main__':main()
