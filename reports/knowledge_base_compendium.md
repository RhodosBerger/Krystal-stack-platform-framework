# Knowledge base compendium

## Induced rules (symbolic)

- **SUPPORTED_WITH_SCOPE** — GPU conv kernel computes sequential FMA in natural k order `(R_GPU_RECIPE)`
- **SUPPORTED_WITH_SCOPE** — CPU conv kernel matches no sequential summation recipe (blocked accumulation) `(R_CPU_RECIPE)`
- **SUPPORTED_WITH_SCOPE** — BLAS == sequential FMA iff K < 449 (matmul bisection) agrees with conv profile sweep `(R_BRIDGE_BLAS_BOUNDARY)`
- **DERIVED** — Allocate the conv pipeline to GPU only when > 530 inferences run per compiled model; otherwise CPU `(R_ALLOC_BREAK_EVEN)`

## LLM answer to: Which device should run the conv pipeline and why?

_LLM unavailable: URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>. No generated text included._
