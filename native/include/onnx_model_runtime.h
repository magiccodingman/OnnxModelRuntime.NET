#ifndef ONNX_MODEL_RUNTIME_H
#define ONNX_MODEL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMR_ABI_VERSION 1u

#if defined(_WIN32)
#define OMR_CALL __cdecl
#else
#define OMR_CALL
#endif

typedef enum omr_status {
    OMR_OK = 0,
    OMR_INVALID_ARGUMENT = 1,
    OMR_BUFFER_TOO_SMALL = 2,
    OMR_INVALID_HANDLE = 3,
    OMR_APPLICATION_ERROR = 4,
    OMR_RECOVERABLE_RUNTIME_ERROR = 5,
    OMR_MEMORY_PRESSURE = 6,
    OMR_FATAL_RUNTIME_ERROR = 7,
    OMR_DISPOSED = 8,
    OMR_INTERNAL_ERROR = 255
} omr_status;

typedef enum omr_executor_status {
    OMR_EXECUTOR_OK = 0,
    OMR_EXECUTOR_APPLICATION_ERROR = 1,
    OMR_EXECUTOR_RECOVERABLE_RUNTIME_ERROR = 2,
    OMR_EXECUTOR_MEMORY_PRESSURE = 3,
    OMR_EXECUTOR_FATAL_RUNTIME_ERROR = 4
} omr_executor_status;

typedef enum omr_instance_health {
    OMR_INSTANCE_STARTING = 0,
    OMR_INSTANCE_HEALTHY = 1,
    OMR_INSTANCE_DRAINING = 2,
    OMR_INSTANCE_RECOVERING = 3,
    OMR_INSTANCE_FAULTED = 4,
    OMR_INSTANCE_DISPOSED = 5
} omr_instance_health;

typedef struct omr_runtime_options_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t model_instance_count;
    int32_t threads_per_model;
    int32_t maximum_auto_threads_per_model;
    int32_t concurrent_requests_per_model;
    int32_t queue_capacity;
} omr_runtime_options_v1;

typedef struct omr_buffer_view {
    const uint8_t* data;
    size_t length;
} omr_buffer_view;

typedef struct omr_buffer {
    uint8_t* data;
    size_t length;
} omr_buffer;

typedef int32_t (OMR_CALL *omr_create_instance_fn)(
    void* user_data,
    int32_t instance_index,
    int32_t generation,
    int32_t threads_per_model,
    void** output_instance_context);

typedef int32_t (OMR_CALL *omr_execute_fn)(
    void* instance_context,
    const uint8_t* request,
    size_t request_length,
    omr_buffer_view* output);

typedef void (OMR_CALL *omr_destroy_instance_fn)(void* instance_context);
typedef void (OMR_CALL *omr_release_response_fn)(void* instance_context, const uint8_t* data, size_t length);

typedef struct omr_executor_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    void* user_data;
    omr_create_instance_fn create_instance;
    omr_execute_fn execute;
    omr_destroy_instance_fn destroy_instance;
    omr_release_response_fn release_response;
} omr_executor_v1;

typedef struct omr_instance_info_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t index;
    int32_t health;
    int32_t active_requests;
    int32_t max_concurrent_requests;
    int32_t generation;
    int32_t total_recoveries;
    int32_t recovery_attempts;
} omr_instance_info_v1;

uint32_t omr_abi_version(void);
int32_t omr_get_last_error(uint8_t* buffer, size_t buffer_length, size_t* required_length);
int32_t omr_runtime_create(const omr_runtime_options_v1* options, const omr_executor_v1* executor, intptr_t* output_handle);
int32_t omr_runtime_destroy(intptr_t handle);
int32_t omr_runtime_execute(intptr_t handle, const uint8_t* request, size_t request_length, omr_buffer* output);
int32_t omr_runtime_get_instance_count(intptr_t handle, int32_t* output_count);
int32_t omr_runtime_get_instance_info(intptr_t handle, int32_t instance_index, omr_instance_info_v1* output);
int32_t omr_runtime_get_instance_last_failure(intptr_t handle, int32_t instance_index, omr_buffer* output);
void omr_buffer_free(omr_buffer* buffer);

#ifdef __cplusplus
}
#endif

#endif
