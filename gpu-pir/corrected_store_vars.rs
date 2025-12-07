pub fn store_vars(db: &[u32], v_firstdim: &[u32], out: &mut Vec<PolyMatrixNTT>) {
    let folder = "logs";

    // Write u32 buffers using a raw cast to u8
    fn write_u32_binary(filename: &str, buf: &[u32]) {
        if let Ok(mut f) = std::fs::File::create(filename) {
            use std::io::Write;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4)
            };
            let _ = f.write_all(bytes);
        } else {
            eprintln!("Failed to write {}", filename);
        }
    }

    // Write u64 buffers as u32 (converting from u64 to u32)
    // The GPU expects uint32_t values, not u64
    fn write_u64_as_u32_binary(filename: &str, buf: &[u64]) {
        if let Ok(mut f) = std::fs::File::create(filename) {
            use std::io::Write;
            // Convert u64 to u32 by taking the lower 32 bits
            // This matches the GPU structure which uses uint32_t
            let u32_buf: Vec<u32> = buf.iter().map(|&x| x as u32).collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(u32_buf.as_ptr() as *const u8, u32_buf.len() * 4)
            };
            let _ = f.write_all(bytes);
        } else {
            eprintln!("Failed to write {}", filename);
        }
    }

    // Flatten Vec<PolyMatrixNTT> into Vec<u64> (keeping u64 for processing)
    fn flatten_polymatrixntt(vec: &Vec<PolyMatrixNTT>) -> Vec<u64> {
        let mut out = Vec::new();
        for pm in vec {
            out.extend_from_slice(pm.data.as_slice());
        }
        out
    }

    // Write db
    write_u32_binary(&format!("{}/eric_db.dat", folder), db);

    // Write v_firstdim
    write_u32_binary(&format!("{}/eric_first_dim.dat", folder), v_firstdim);

    // Write out - convert from u64 to u32 to match GPU structure
    // The GPU FoldingInputs structure uses uint32_t (unsigned) for all coefficients
    // Layout: For each column: [a.crt_0][a.crt_1][as_e.crt_0][as_e.crt_1]
    // Each section is POLY_LEN uint32_t values
    let flat_out = flatten_polymatrixntt(out);
    write_u64_as_u32_binary(&format!("{}/eric_folding_inputs.dat", folder), &flat_out);
}

