"""AST observation copies of the exact loaded functions; no site-package edits."""
import ast, inspect, json, types
from pathlib import Path
import torch

class Recorder:
    def __init__(self,path,label,meta):self.path=path;self.label=label;self.meta=meta;self.events=[];self.tensors={};self.column=0
    def emit(self,op,tensors,col=-1,line=0):
        if col>=0:self.column=col
        if col>=128:return
        item={'step':len(self.events),'operation':op,'column':col,'source_line':line,'tensors':{}}
        for name,t in tensors.items():
            if not isinstance(t,torch.Tensor):continue
            key=f'{len(self.events):04d}/{name}'
            self.tensors[key]=t.detach().cpu().clone()
            info=self.meta(t);info.update(storage_ptr=t.untyped_storage().data_ptr(),data_ptr=t.data_ptr(),storage_offset=t.storage_offset(),has_base=t._base is not None,tensor_key=key)
            item['tensors'][name]=info
        self.events.append(item)
    def save(self):
        torch.save(self.tensors,self.path/f'operator_tensors_{self.label}.pt')
        (self.path/f'operator_trace_{self.label}.json').write_text(json.dumps({'scope':'group0 from a common captured native Hinv; upstream failure is recorded separately','events':self.events,'tensor_file':f'operator_tensors_{self.label}.pt'},indent=2)+'\n',encoding='utf-8')

def make_instrumented(fn,recorder,mode,out):
    original=inspect.unwrap(fn);file=Path(original.__code__.co_filename);tree=ast.parse(file.read_text(encoding='utf-8'))
    target=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==original.__name__)
    target.decorator_list=[]
    def emit(op,expressions,line,col):
        vals='{'+','.join(repr(k)+':'+v for k,v in expressions.items())+'}'
        return ast.parse(f'_audit_emit({op!r}, {vals}, {col}, {line})').body
    class Instrument(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            body=[]
            for s in node.body:body.extend(self.transform(s))
            node.body=body;return node
        def transform(self,s):
            # Recurse through loop/if/with/try blocks, preserving original AST expressions.
            for field in ('body','orelse','finalbody'):
                if hasattr(s,field):
                    body=[]
                    for child in getattr(s,field):body.extend(self.transform(child))
                    setattr(s,field,body)
            if isinstance(s,ast.Try):
                for h in s.handlers:
                    body=[]
                    for child in h.body:body.extend(self.transform(child))
                    h.body=body
            text=ast.unparse(s).split('\n')[0];line=s.lineno
            col='i1+i' if mode=='native' else 'col'
            if isinstance(s,ast.AugAssign) and ('W1[:, i:]' in text or 'local[:, j:]' in text):
                value=s.value;s.value=ast.Name(id='_observed_product',ctx=ast.Load())
                ass=ast.copy_location(ast.Assign(targets=[ast.Name(id='_observed_product',ctx=ast.Store())],value=value),s)
                matrix='W1[:,i:]' if mode=='native' else 'local[:,j:]'
                return [ass]+emit('compensation_product',{'value':'_observed_product'},line,col)+[s]+emit('updated_remaining_W',{'value':matrix},line,col)
            if isinstance(s,ast.AugAssign) and ('W[:, i2:]' in text or 'work[:, i2:]' in text):
                value=s.value;s.value=ast.Name(id='_observed_transition',ctx=ast.Load())
                ass=ast.copy_location(ast.Assign(targets=[ast.Name(id='_observed_transition',ctx=ast.Store())],value=value),s)
                matrix='W[:,i2:]' if mode=='native' else 'work[:,i2:]'
                return [ass]+emit('transition_product',{'value':'_observed_transition'},line,'i1')+[s]+emit('group_exit_remaining_W',{'value':matrix},line,'i1')
            extra=[]
            if mode=='native':
                if text=='w = W1[:, i]':extra=emit('pre_quant_value',{'value':'w'},line,col)
                elif text=='d = Hinv1[i, i]':extra=emit('divisor',{'value':'d'},line,col)
                elif text.startswith('self.quantizer.find_params(W[:,'):extra=emit('group_entry',{'value':'W[:,i1+i:i1+i+self.qcfg.group_size]','scale':'self.quantizer.scale','zero':'self.quantizer.zero'},line,col)
                elif text.startswith('q = self.quantizer.quantize'):extra=emit('dequantized_q',{'value':'q'},line,col)
                elif text=='err1 = (w - q) / d':extra=emit('error_vector',{'value':'err1'},line,col)
                elif text.startswith('W1 = W[:,'):extra=emit('working_block',{'value':'W1','Hinv_block':'Hinv[i1:i2,i1:i2]'},line,'i1')
            else:
                if text.startswith('current = local[:,'):extra=emit('pre_quant_value',{'value':'current.flatten()'},line,col)+emit('divisor',{'value':'local_inv[j,j]'},line,col)
                elif text.startswith('current_scale, current_zero ='):extra=emit('group_entry',{'value':'entering','scale':'current_scale','zero':'current_zero'},line,col)
                elif text.startswith('q = (torch.round'):extra=emit('rounded_integer_q',{'value':'q.flatten()'},line,col)
                elif text=='q = (q - current_zero) * current_scale':extra=emit('dequantized_q',{'value':'q.flatten()'},line,col)
                elif text.startswith('error = (current.flatten()'):extra=emit('error_vector',{'value':'error'},line,col)
                elif text.startswith('local = work[:,'):extra=emit('working_block',{'value':'local','Hinv_block':'hinv[i1:i2,i1:i2]'},line,'i1')
            return [s]+extra
    target=Instrument().visit(target);ast.fix_missing_locations(target)
    mod=ast.Module(body=[target],type_ignores=[]);env=dict(original.__globals__);env['_audit_emit']=recorder.emit
    exec(compile(mod,str(file),'exec'),env)
    (out/f'instrumented_{mode}.py').write_text(ast.unparse(mod)+'\n',encoding='utf-8')
    return env[target.name]

def native_integer_observer(recorder,original_helper):
    original=inspect.unwrap(original_helper);file=Path(original.__code__.co_filename)
    tree=ast.parse(file.read_text(encoding='utf-8'))
    target=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==original.__name__)
    class AddObservation(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node).startswith('q = torch.clamp'):
                obs=ast.parse(f"_rec.emit('rounded_integer_q', {{'value':q.flatten()}}, _rec.column, {node.lineno})").body
                return [node]+obs
            return node
    target=AddObservation().visit(target);ast.fix_missing_locations(target)
    mod=ast.Module(body=[target],type_ignores=[]);env=dict(original.__globals__);env['_rec']=recorder
    exec(compile(mod,str(file),'exec'),env)
    (recorder.path/'instrumented_quantizer.py').write_text(ast.unparse(mod)+'\n',encoding='utf-8')
    return env[target.name]
