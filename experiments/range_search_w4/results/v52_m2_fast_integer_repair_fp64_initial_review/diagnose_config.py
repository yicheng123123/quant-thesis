import json,hashlib,time
from pathlib import Path
from transformers import AutoConfig
import transformers
O=Path(__file__).resolve().parent;B=O.parent
paths={'M2':B/'range_w4_fullmodel_v18/M2/checkpoint/final','v52':O/'R1/checkpoint/final','original_R2':B/'v45_m2_single_integer_repair_fullmodel/R/checkpoint/final'}
ignore={'_name_or_path','transformers_version'}
raw={k:json.loads((p/'config.json').read_text()) for k,p in paths.items()}
loaded={k:AutoConfig.from_pretrained(str(p),local_files_only=True).to_dict() for k,p in paths.items()}
base=loaded['M2']
def diff(a,b):return {k:{'a':a.get(k,'<missing>'),'b':b.get(k,'<missing>')} for k in sorted(set(a)|set(b)) if k not in ignore and a.get(k,'<missing>')!=b.get(k,'<missing>')}
r={'transformers_version':transformers.__version__,'raw_config_differences':{k:diff(raw['M2'],x) for k,x in raw.items()},'loaded_config_differences':{k:diff(base,x) for k,x in loaded.items()},'all_loaded_fields_equal_excluding_original_ignore':all(not diff(base,x) for x in loaded.values()),'excluded_fields':sorted(ignore),'loaded_configs':loaded,'sources':{k:{'path':str(p/'config.json'),'sha256':hashlib.sha256((p/'config.json').read_bytes()).hexdigest()} for k,p in paths.items()},'CPU_only':True,'timestamp':time.time()}
(O/'CONFIG_SCHEMA_DIAGNOSTIC.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='loaded_configs'},indent=2,default=str))
