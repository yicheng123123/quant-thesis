"""CPU snapshot audit of currently committed groups; no model/CUDA use."""
from finalize_v45 import persisted_audit
if __name__=='__main__':
 m,r,j,f=persisted_audit();print('SNAPSHOT_AUDIT_PASS',len(m),len(r),'rechecks',len(f),'rejects',len(j),flush=True)
