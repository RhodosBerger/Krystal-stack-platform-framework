/* ============================================================================
 * KRYSTAL STACK — UNIVERSAL DEVICE MEMORY-MAPPED I/O (MMIO) & BAR APERTURE ABI
 * File: include/krystal_universal_mmio_aperture.h
 * Targets: Linux VFIO/UIO/DMA-BUF, Windows WDDM 3.2/DXCore/KMDF, POSIX shm
 * Constitutional Invariant: KRYSTAL_VITAL_MAX_HP == 6 (\neg Valid => \Delta S = 0)
 * ============================================================================ */

#ifndef KRYSTAL_UNIVERSAL_MMIO_APERTURE_H
#define KRYSTAL_UNIVERSAL_MMIO_APERTURE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "kr_authority_bits.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KRYSTAL_MMIO_MAGIC_U32       0x4B4D4D49U /* "KMMI" */
#define KRYSTAL_MMIO_ABI_VERSION     0x00010000U
#define KRYSTAL_MMIO_VITAL_MAX_HP    6U
#define KRYSTAL_MMIO_BAR0_BYTES      4096U
#define KRYSTAL_MMIO_CACHELINE_BYTES 64U

/* Canonical BAR0 Control/Status Register Byte Offsets */
#define KRYSTAL_REG_DEVICE_MAGIC        0x0000U /* RO u32 */
#define KRYSTAL_REG_ABI_VERSION         0x0004U /* RO u32 */
#define KRYSTAL_REG_VITAL_MAX_HP        0x0008U /* RO u32 == 6 */
#define KRYSTAL_REG_DEVICE_STATUS       0x000CU /* RO u32 */
#define KRYSTAL_REG_AUTHORITY_MASK      0x0010U /* RW u64 */
#define KRYSTAL_REG_UMA_UTIL_MILLI      0x0018U /* RW u32 (0..1000, veto >= 780) */
#define KRYSTAL_REG_THERMAL_DECI_C      0x001CU /* RW u32 (0.1 C, hard trip >= 1000) */
#define KRYSTAL_REG_DMA_SRC_OFFSET      0x0020U /* RW u64 (64B-aligned) */
#define KRYSTAL_REG_DMA_DST_OFFSET      0x0028U /* RW u64 (64B-aligned) */
#define KRYSTAL_REG_DMA_BYTE_LEN        0x0030U /* RW u32 */
#define KRYSTAL_REG_OP_COMMAND          0x0034U /* RW u32 */
#define KRYSTAL_REG_DOORBELL_TAIL       0x0040U /* WO/RW u64 atomic doorbell */
#define KRYSTAL_REG_COMPLETED_HEAD      0x0048U /* RO u64 completion fence */
#define KRYSTAL_REG_REJECTED_NOOP_COUNT 0x0050U /* RO u64 (\Delta S = 0 counter) */
#define KRYSTAL_REG_ATTESTATION_SHA256  0x0060U /* RO 32B digest (0x0060..0x007F) */

typedef enum KrystalMmioCommand {
    KRYSTAL_MMIO_CMD_NOP               = 0,
    KRYSTAL_MMIO_CMD_ZERO_COPY_XFER    = 1,
    KRYSTAL_MMIO_CMD_SCALE_RELU_F32    = 2,
    KRYSTAL_MMIO_CMD_SUBNORMAL_FLUSH   = 3,
    KRYSTAL_MMIO_CMD_FENCE_SYNC        = 4
} KrystalMmioCommand;

typedef enum KrystalMmioError {
    KRYSTAL_MMIO_OK                    = 0,
    KRYSTAL_MMIO_ERR_NULL_APERTURE     = -1,
    KRYSTAL_MMIO_ERR_OUT_OF_BOUNDS     = -2,
    KRYSTAL_MMIO_ERR_UNALIGNED_ADDR    = -3,
    KRYSTAL_MMIO_ERR_READ_ONLY_REG     = -4,
    KRYSTAL_MMIO_ERR_HP_INVARIANT      = -5,
    KRYSTAL_MMIO_ERR_UMA_SATURATED     = -6,
    KRYSTAL_MMIO_ERR_THERMAL_TRIP      = -7,
    KRYSTAL_MMIO_ERR_AUTHORITY_ZERO    = -8
} KrystalMmioError;

static inline bool krystal_mmio_is_read_only_offset(uint32_t offset) {
    return (offset == KRYSTAL_REG_DEVICE_MAGIC) ||
           (offset == KRYSTAL_REG_ABI_VERSION) ||
           (offset == KRYSTAL_REG_VITAL_MAX_HP) ||
           (offset == KRYSTAL_REG_DEVICE_STATUS) ||
           (offset == KRYSTAL_REG_COMPLETED_HEAD) ||
           (offset == KRYSTAL_REG_REJECTED_NOOP_COUNT) ||
           (offset >= KRYSTAL_REG_ATTESTATION_SHA256 && offset < 0x0080U);
}

static inline KrystalMmioError krystal_mmio_read_u32(
    const uint8_t *bar0_base,
    size_t bar0_size,
    uint32_t offset,
    uint32_t *out_val
) {
    if (!bar0_base || !out_val) return KRYSTAL_MMIO_ERR_NULL_APERTURE;
    if ((offset & 0x3U) != 0U) return KRYSTAL_MMIO_ERR_UNALIGNED_ADDR;
    if ((size_t)offset + sizeof(uint32_t) > bar0_size) return KRYSTAL_MMIO_ERR_OUT_OF_BOUNDS;
    *out_val = *(const volatile uint32_t *)(bar0_base + offset);
    return KRYSTAL_MMIO_OK;
}

static inline KrystalMmioError krystal_mmio_write_u32(
    uint8_t *bar0_base,
    size_t bar0_size,
    uint32_t offset,
    uint32_t val
) {
    if (!bar0_base) return KRYSTAL_MMIO_ERR_NULL_APERTURE;
    if ((offset & 0x3U) != 0U) return KRYSTAL_MMIO_ERR_UNALIGNED_ADDR;
    if ((size_t)offset + sizeof(uint32_t) > bar0_size) return KRYSTAL_MMIO_ERR_OUT_OF_BOUNDS;
    if (krystal_mmio_is_read_only_offset(offset)) return KRYSTAL_MMIO_ERR_READ_ONLY_REG;
    *(volatile uint32_t *)(bar0_base + offset) = val;
    return KRYSTAL_MMIO_OK;
}

#ifdef __cplusplus
}
#endif

#endif /* KRYSTAL_UNIVERSAL_MMIO_APERTURE_H */
