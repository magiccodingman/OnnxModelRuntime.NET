# Native AOT and C ABI v1

## Goal

`OnnxModelRuntime.Native` publishes the generic scheduling/runtime layer as a Native AOT shared library without pretending that arbitrary ONNX model contracts can be represented by one useful C inference function.

The public header is:

```text
native/include/onnx_model_runtime.h
```

ABI version 1 is discovered with:

```c
uint32_t omr_abi_version(void);
```

## ABI principles

- explicit numeric ABI version;
- `struct_size` and `abi_version` on versioned input/output structs;
- opaque `intptr_t` runtime handles;
- stable numeric status values;
- managed exceptions are caught and mapped before crossing the boundary;
- byte/string data use pointer + explicit byte length;
- no returned string relies on NUL termination;
- last-error state is thread-local;
- library-owned output buffers are released only with `omr_buffer_free`;
- callback-owned buffers can be released with the caller's own `release_response` callback;
- foreign callers are never required to allocate or free memory with a .NET allocator.

## Executor callback model

`omr_runtime_create` receives `omr_executor_v1`. It contains an opaque `user_data` pointer plus callbacks:

```text
create_instance   optional per-model-copy initialization
execute           required request execution
release_response  optional release for callback-owned response bytes
destroy_instance  optional per-instance cleanup
```

`create_instance` receives instance index, generation, and resolved threads/model. It can return an opaque per-instance context.

`execute` receives only opaque request bytes and returns an `omr_buffer_view`. Those bytes are intentionally not given universal ONNX semantics. A higher-level package can define exactly how its request is encoded.

When `execute` returns a non-success `omr_executor_status`, response bytes are interpreted as a UTF-8 error message and the status maps to the runtime's application/recoverable/memory/fatal failure policy.

The runtime copies the callback response before invoking `release_response`, so callback-owned memory does not escape its ownership domain.

## Returned buffers

`omr_runtime_execute` returns an `omr_buffer` allocated by the library. The caller owns that buffer until:

```c
omr_buffer_free(&buffer);
```

The free call clears both pointer and length.

`omr_runtime_get_instance_last_failure` uses the same buffer ownership rule.

`omr_get_last_error` is different: it uses a caller-provided buffer. Call once with a null/short buffer to discover `required_length`, allocate with the caller's normal allocator, then call again.

## Diagnostics

Foreign callers can retrieve:

- runtime instance count;
- instance index;
- health;
- active requests;
- maximum concurrent requests;
- generation;
- total successful recoveries;
- current recovery attempt;
- last failure text.

`omr_instance_info_v1` is versioned with `struct_size` and `abi_version`.

## What ABI v1 does not mirror

The following managed concepts are intentionally not projected into the generic ABI:

- `TRequest` / `TResponse` generic types;
- `InferenceSession` object access;
- `SessionOptions` configuration callbacks;
- ONNX tensor dictionaries or a universal tensor JSON schema;
- model input/output names;
- tokenization;
- embedding/vector formats;
- pooling and normalization;
- reranking contracts;
- classifier contracts;
- prompt formats;
- generative streaming protocols.

Those semantics should be defined by a higher-level native package that actually understands the model family.

## C smoke test

`native/smoke/c/main.c` is a standalone consumer compiled directly against the checked-in public header. CI dynamically loads the published Native AOT shared library, resolves the ABI functions, creates a callback-backed runtime, injects a recoverable generation-1 failure, verifies the runtime rebuilds to generation 2 and retries successfully, validates buffer ownership, reads diagnostics, destroys the handle, and validates last-error behavior.

The same CI matrix also runs a real ONNX Runtime path in a managed Native AOT smoke executable, ensuring the produced AOT bundle includes and can load the RID-specific ONNX Runtime native sidecar.
