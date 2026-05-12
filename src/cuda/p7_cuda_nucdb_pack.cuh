/* Device-side helpers for reading 2-bit packed .nucdb v2 residues.
 *
 * The packed buffer stores 4 residues per byte: residue at position pos
 * occupies bits [2*(pos%4) .. 2*(pos%4)+1] of byte pos/4.
 * Codes: A=0, C=1, G=2, T=3. No N on device (N positions stored as 0,
 * scored as A; mask is host-only).
 *
 * RC strand: caller passes the forward-strand absolute position
 * transformed as (chunk_len - 1 - local_pos), and rc=1 triggers XOR
 * with 3 (A<->T, C<->G).
 */
#ifndef P7_CUDA_NUCDB_PACK_CUH_INCLUDED
#define P7_CUDA_NUCDB_PACK_CUH_INCLUDED

__device__ __forceinline__ int
p7_nucdb_fetch1(const uint8_t *packed, int pos, int rc)
{
  int b = pos >> 2;
  int s = (pos & 3) << 1;
  int code = (packed[b] >> s) & 0x3;
  return rc ? (code ^ 0x3) : code;
}

#endif /* P7_CUDA_NUCDB_PACK_CUH_INCLUDED */
