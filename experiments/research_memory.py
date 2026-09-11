"""Array-payload accounting; excludes interpreter overhead, dataset and scratch space."""
import numpy as np


def array_inventory(model):
    arrays = {}
    seen = set()

    def walk(value, path):
        if isinstance(value, np.ndarray):
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in seen:
                seen.add(id(base))
                arrays[path] = int(base.nbytes)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                walk(child, f"{path}.{i}")
        elif hasattr(value, "__dict__"):
            walk(vars(value), path)
    walk(model, "model")
    return arrays


def faststore_memory(store):
    active = int(np.count_nonzero(store.s > 0))
    row_bytes = store.M[0].nbytes + store.s.dtype.itemsize + store.tau.dtype.itemsize
    # Mirror the frozen implementation's formula ONLY for comparison, not enforcement.
    implementation_row_bytes = store.d * 2 + 5
    actual = active * row_bytes
    return {"active_rows": active, "configured_budget_bytes": store.budget_bytes,
            "implementation_row_bytes": implementation_row_bytes,
            "actual_row_bytes": row_bytes,
            "implementation_occupancy_bytes": active * implementation_row_bytes,
            "actual_active_payload_bytes": actual,
            "allocated_faststore_bytes": store.M.nbytes + store.s.nbytes + store.tau.nbytes,
            "actual_exceeds_configured_budget": actual > store.budget_bytes}


def memory_report(model, is_mnema=False):
    inventory = array_inventory(model)
    total = sum(inventory.values())
    result = {"array_inventory": inventory, "total_resident_array_bytes": total}
    if is_mnema:
        store = faststore_memory(model.store)
        cortex = (model.cortex.syn_fc.u.nbytes + model.cortex.eligibility.nbytes
                  + model.cortex.v.nbytes + model.cortex.b.nbytes)
        other = model.separator.theta.nbytes + model.separator.running_rate.nbytes
        if model.encoder.x_ref is not None:
            other += model.encoder.x_ref.nbytes
        fixed = model.separator.indices.nbytes + model.cortex.syn_fc.C.nbytes + model.cortex.syn_fc.g.nbytes
        result.update(store, model_adaptive_bytes=cortex + other, fixed_scaffold_bytes=fixed,
                      cortex_synaptic_adaptive_bytes=cortex, other_adaptive_array_bytes=other,
                      auxiliary_allocated_bytes=store["allocated_faststore_bytes"],
                      auxiliary_content_bytes=store["actual_active_payload_bytes"])
    else:
        params = sum(p.nbytes for p in model.parameters().values())
        result.update(model_adaptive_bytes=params, fixed_scaffold_bytes=0,
                      auxiliary_allocated_bytes=total - params, auxiliary_content_bytes=total - params)
        if hasattr(model, "buffer"):
            result.update(model.buffer.memory())
        if hasattr(model, "fishers"):
            result.update(fisher_bytes=sum(v.nbytes for f in model.fishers for v in f.values()),
                          parameter_snapshot_bytes=sum(v.nbytes for f in model.optima for v in f.values()),
                          other_cl_array_bytes=0)
    assert total == (result["model_adaptive_bytes"] + result["fixed_scaffold_bytes"]
                     + result["auxiliary_allocated_bytes"])
    return result
