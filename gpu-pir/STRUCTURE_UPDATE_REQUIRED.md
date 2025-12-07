# Structure Update Required

This document lists all places that need to be updated after changing structures from distinct fields to arrays.

## Structure Changes Summary

1. **Poly**: Changed from `crt_0` and `crt_1` fields → `crt[2]` array
   - `.crt_0` → `.crt[0]`
   - `.crt_1` → `.crt[1]`

2. **Ciphertext**: Changed from `a` and `as_e` fields → `polys[2]` array
   - `.a` → `.polys[0]`
   - `.as_e` → `.polys[1]`

3. **CiphertextHalf**: Changed from `a` and `as_e` fields → `data[2]` array
   - `.a` → `.data[0]`
   - `.as_e` → `.data[1]`

---

## Files Requiring Updates

### 1. `db_mul.cuh`
**Lines 56-61**: Accessing Ciphertext fields and Poly fields
- `first_dim->selectors[row].a.crt_0` → `first_dim->selectors[row].polys[0].crt[0]`
- `first_dim->selectors[row].a.crt_1` → `first_dim->selectors[row].polys[0].crt[1]`
- `first_dim->selectors[row].as_e.crt_0` → `first_dim->selectors[row].polys[1].crt[0]`
- `first_dim->selectors[row].as_e.crt_1` → `first_dim->selectors[row].polys[1].crt[1]`
- `db->col[col].row[row].crt_0` → `db->col[col].row[row].crt[0]`
- `db->col[col].row[row].crt_1` → `db->col[col].row[row].crt[1]`

**Lines 74-78**: Writing to Ciphertext fields
- `folding_inputs->selectors[col].a.crt_0.data[coeff]` → `folding_inputs->selectors[col].polys[0].crt[0].data[coeff]`
- `folding_inputs->selectors[col].as_e.crt_0.data[coeff]` → `folding_inputs->selectors[col].polys[1].crt[0].data[coeff]`
- `folding_inputs->selectors[col].a.crt_1.data[coeff]` → `folding_inputs->selectors[col].polys[0].crt[1].data[coeff]`
- `folding_inputs->selectors[col].as_e.crt_1.data[coeff]` → `folding_inputs->selectors[col].polys[1].crt[1].data[coeff]`

### 2. `db_mul_test.cu`
**Lines 159-176**: Comparing Ciphertext fields in test
- `computed_folding_inputs->selectors[i].a.crt_0.data[j]` → `computed_folding_inputs->selectors[i].polys[0].crt[0].data[j]`
- `computed_folding_inputs->selectors[i].a.crt_1.data[j]` → `computed_folding_inputs->selectors[i].polys[0].crt[1].data[j]`
- `computed_folding_inputs->selectors[i].as_e.crt_0.data[j]` → `computed_folding_inputs->selectors[i].polys[1].crt[0].data[j]`
- `computed_folding_inputs->selectors[i].as_e.crt_1.data[j]` → `computed_folding_inputs->selectors[i].polys[1].crt[1].data[j]`
- Same for `correct_folding_inputs`

### 3. `fold_test.cu`
**Line 48**: Commented out printf (may need update if uncommented)
- `query.accumulators[0][i].a.crt_0.data` → `query.accumulators[0][i].polys[0].crt[0].data`

**Lines 52-60**: Assigning CiphertextHalf fields from Ciphertext
- `query.v_folding[i][0][j].a = test_keys[i][j].a.crt_0` → `query.v_folding[i][0][j].data[0] = test_keys[i][j].polys[0].crt[0]`
- `query.v_folding[i][1][j].a = test_keys[i][j].a.crt_1` → `query.v_folding[i][1][j].data[0] = test_keys[i][j].polys[0].crt[1]`
- `query.v_folding[i][0][j].as_e = test_keys[i][j].as_e.crt_0` → `query.v_folding[i][0][j].data[1] = test_keys[i][j].polys[1].crt[0]`
- `query.v_folding[i][1][j].as_e = test_keys[i][j].as_e.crt_1` → `query.v_folding[i][1][j].data[1] = test_keys[i][j].polys[1].crt[1]`
- Same pattern for `v_folding_neg`

### 4. `host.cu`
**Lines 157-158**: Reading Poly fields in `ntt_to_gpu_form`
- `p.crt_0.data[i]` → `p.crt[0].data[i]`
- `p.crt_1.data[i]` → `p.crt[1].data[i]`

**Lines 161-164**: Accessing Poly fields
- `p.crt_0.data` → `p.crt[0].data`
- `p.crt_1.data` → `p.crt[1].data`
- `rearrange(p.crt_0)` → `rearrange(p.crt[0])`
- `rearrange(p.crt_1)` → `rearrange(p.crt[1])`

**Lines 170-171**: Reading Poly fields in `raw_to_gpu_form`
- `p.crt_0.data[i]` → `p.crt[0].data[i]`
- `p.crt_1.data[i]` → `p.crt[1].data[i]`

**Lines 192-193**: Assigning Ciphertext fields in `to_ciphertexts`
- `c.a = v[i]` → `c.polys[0] = v[i]`
- `c.as_e = v[i + len_half]` → `c.polys[1] = v[i + len_half]`

**Lines 203-204, 209-210**: Assigning CiphertextHalf fields in `to_ciphertext_halfs`
- `c.a = v[i].a.crt_0` → `c.data[0] = v[i].polys[0].crt[0]`
- `c.as_e = v[i].as_e.crt_0` → `c.data[1] = v[i].polys[1].crt[0]`
- `c.a = v[i].a.crt_1` → `c.data[0] = v[i].polys[0].crt[1]`
- `c.as_e = v[i].as_e.crt_1` → `c.data[1] = v[i].polys[1].crt[1]`

### 5. `gadget.cuh`
**Lines 31-32**: Writing Poly fields in `write_regs`
- `dst.crt_0.data[z]` → `dst.crt[0].data[z]`
- `dst.crt_1.data[z]` → `dst.crt[1].data[z]`

**Lines 67-68**: Reading Poly fields in `inv_gadget_ntt_mult1`
- `input.crt_0.data[z]` → `input.crt[0].data[z]`
- `input.crt_1.data[z]` → `input.crt[1].data[z]`

**Line 78**: Accessing CiphertextHalf field
- `mat[row].a` → `mat[row].data[0]`

**Line 80**: Accessing CiphertextHalf field
- `mat[row].as_e` → `mat[row].data[1]`

**Line 90**: Accessing Ciphertext fields
- `input.a` → `input.polys[0]`
- `input.as_e` → `input.polys[1]`

### 6. `classes.cuh`
**Lines 209-210**: Writing Poly fields in `gadget_inv_rdim1`
- `input_in_high_low_crt.crt_0.data[j]` → `input_in_high_low_crt.crt[0].data[j]`
- `input_in_high_low_crt.crt_1.data[j]` → `input_in_high_low_crt.crt[1].data[j]`

**Note**: Lines 125-126, 199-200, 222-229 use `CiphertextNTT` and `CiphertextRaw` which are DIFFERENT structures and don't need updating.

### 7. `expansion.cuh`
**Note**: Most of this file uses `CiphertextNTT` and `CiphertextRaw` structures (defined in `classes.cuh`), which are DIFFERENT from `Ciphertext` in `structures.cuh` and do NOT need updating.

**Line 89**: `Ciphertext* w` points to `PublicParameters->left_expansion[r]` or `right_expansion[r]`, which are `Ciphertext` arrays (from `structures.cuh`). If `w` is dereferenced to access `.a` or `.as_e` fields anywhere, those would need updating. Currently, `w` is just passed as a pointer to functions, so verify if any direct field access occurs.

**Line 114**: `gadget_inv_rdim1(&ct_auto.a, w, gadget_dim)` - `ct_auto` is `CiphertextRaw` (different structure), so no update needed here.

### 8. `ntt_test.cu`
**Line 40**: Reading Poly fields
- `test[0].crt_0.data[i]` → `test[0].crt[0].data[i]`
- `test[0].crt_1.data[i]` → `test[0].crt[1].data[i]`

---

## Notes

- **Python files** in `cpu_version/` directory are separate implementations and may not need immediate updates, but should be updated for consistency.
- **CiphertextNTT** and **CiphertextRaw** in `classes.cuh` are different structures and should NOT be updated.
- Be careful with nested accesses - the pattern is typically:
  - `ciphertext.a.crt_0` → `ciphertext.polys[0].crt[0]`
  - `ciphertext.as_e.crt_1` → `ciphertext.polys[1].crt[1]`
  - `ciphertext_half.a` → `ciphertext_half.data[0]`
  - `ciphertext_half.as_e` → `ciphertext_half.data[1]`
  - `poly.crt_0` → `poly.crt[0]`
  - `poly.crt_1` → `poly.crt[1]`

