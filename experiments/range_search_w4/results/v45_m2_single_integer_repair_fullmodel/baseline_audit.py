"""Read-only historical baseline verification, before new R PPL exists."""
import audit_helpers as ah
if __name__=='__main__':
 assert not (ah.OUT/'R_wikitext.json').exists() and not (ah.OUT/'R_c4_fixed_budget.json').exists()
 assert ah.ppl_audit()==[]
 print('BASELINE_SOURCE_AND_WINDOWS_PASS',flush=True)
