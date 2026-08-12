#include "onnx_model_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE library_handle;
static library_handle open_library(const char* path) { return LoadLibraryA(path); }
static void* load_symbol(library_handle lib, const char* name) { return (void*)GetProcAddress(lib, name); }
static void close_library(library_handle lib) { if (lib) FreeLibrary(lib); }
#else
#include <dlfcn.h>
typedef void* library_handle;
static library_handle open_library(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
static void* load_symbol(library_handle lib, const char* name) { return dlsym(lib, name); }
static void close_library(library_handle lib) { if (lib) dlclose(lib); }
#endif

typedef uint32_t (OMR_CALL *abi_version_fn)(void);
typedef int32_t (OMR_CALL *get_last_error_fn)(uint8_t*, size_t, size_t*);
typedef int32_t (OMR_CALL *runtime_create_fn)(const omr_runtime_options_v1*, const omr_executor_v1*, intptr_t*);
typedef int32_t (OMR_CALL *runtime_destroy_fn)(intptr_t);
typedef int32_t (OMR_CALL *runtime_execute_fn)(intptr_t, const uint8_t*, size_t, omr_buffer*);
typedef int32_t (OMR_CALL *runtime_get_instance_count_fn)(intptr_t, int32_t*);
typedef int32_t (OMR_CALL *runtime_get_instance_info_fn)(intptr_t, int32_t, omr_instance_info_v1*);
typedef void (OMR_CALL *buffer_free_fn)(omr_buffer*);

typedef struct smoke_instance {
    int32_t index;
    int32_t generation;
} smoke_instance;

static int32_t OMR_CALL create_instance(
    void* user_data,
    int32_t instance_index,
    int32_t generation,
    int32_t threads_per_model,
    void** output_instance_context) {
    (void)user_data;
    (void)threads_per_model;
    smoke_instance* instance = (smoke_instance*)calloc(1, sizeof(smoke_instance));
    if (!instance) return OMR_EXECUTOR_MEMORY_PRESSURE;
    instance->index = instance_index;
    instance->generation = generation;
    *output_instance_context = instance;
    return OMR_EXECUTOR_OK;
}

static int32_t OMR_CALL execute_echo(
    void* instance_context,
    const uint8_t* request,
    size_t request_length,
    omr_buffer_view* output) {
    smoke_instance* instance = (smoke_instance*)instance_context;
    if (!instance || !output) return OMR_EXECUTOR_FATAL_RUNTIME_ERROR;
    if (instance->generation == 1) {
        const char* failure = "recover generation one";
        const size_t failure_length = strlen(failure);
        uint8_t* error = (uint8_t*)malloc(failure_length);
        if (!error) return OMR_EXECUTOR_MEMORY_PRESSURE;
        memcpy(error, failure, failure_length);
        output->data = error;
        output->length = failure_length;
        return OMR_EXECUTOR_RECOVERABLE_RUNTIME_ERROR;
    }

    uint8_t* response = NULL;
    if (request_length > 0) {
        response = (uint8_t*)malloc(request_length);
        if (!response) return OMR_EXECUTOR_MEMORY_PRESSURE;
        memcpy(response, request, request_length);
    }
    output->data = response;
    output->length = request_length;
    return OMR_EXECUTOR_OK;
}

static void OMR_CALL destroy_instance(void* instance_context) { free(instance_context); }
static void OMR_CALL release_response(void* instance_context, const uint8_t* data, size_t length) {
    (void)instance_context;
    (void)length;
    free((void*)data);
}

#define LOAD_REQUIRED(target, type, name) do { \
    target = (type)load_symbol(lib, name); \
    if (!(target)) { fprintf(stderr, "missing symbol: %s\n", name); close_library(lib); return 3; } \
} while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: native-smoke <library>\n");
        return 2;
    }

    library_handle lib = open_library(argv[1]);
    if (!lib) {
        fprintf(stderr, "could not load native library\n");
        return 3;
    }

    abi_version_fn abi_version;
    get_last_error_fn get_last_error;
    runtime_create_fn runtime_create;
    runtime_destroy_fn runtime_destroy;
    runtime_execute_fn runtime_execute;
    runtime_get_instance_count_fn get_instance_count;
    runtime_get_instance_info_fn get_instance_info;
    buffer_free_fn buffer_free;

    LOAD_REQUIRED(abi_version, abi_version_fn, "omr_abi_version");
    LOAD_REQUIRED(get_last_error, get_last_error_fn, "omr_get_last_error");
    LOAD_REQUIRED(runtime_create, runtime_create_fn, "omr_runtime_create");
    LOAD_REQUIRED(runtime_destroy, runtime_destroy_fn, "omr_runtime_destroy");
    LOAD_REQUIRED(runtime_execute, runtime_execute_fn, "omr_runtime_execute");
    LOAD_REQUIRED(get_instance_count, runtime_get_instance_count_fn, "omr_runtime_get_instance_count");
    LOAD_REQUIRED(get_instance_info, runtime_get_instance_info_fn, "omr_runtime_get_instance_info");
    LOAD_REQUIRED(buffer_free, buffer_free_fn, "omr_buffer_free");

    if (abi_version() != OMR_ABI_VERSION) {
        fprintf(stderr, "ABI version mismatch\n");
        close_library(lib);
        return 4;
    }

    omr_runtime_options_v1 options;
    memset(&options, 0, sizeof(options));
    options.struct_size = (uint32_t)sizeof(options);
    options.abi_version = OMR_ABI_VERSION;
    options.model_instance_count = 1;
    options.threads_per_model = 2;
    options.maximum_auto_threads_per_model = 16;
    options.concurrent_requests_per_model = 2;
    options.queue_capacity = 8;

    omr_executor_v1 executor;
    memset(&executor, 0, sizeof(executor));
    executor.struct_size = (uint32_t)sizeof(executor);
    executor.abi_version = OMR_ABI_VERSION;
    executor.create_instance = create_instance;
    executor.execute = execute_echo;
    executor.destroy_instance = destroy_instance;
    executor.release_response = release_response;

    intptr_t handle = 0;
    int32_t status = runtime_create(&options, &executor, &handle);
    if (status != OMR_OK || handle == 0) {
        fprintf(stderr, "runtime_create failed: %d\n", status);
        close_library(lib);
        return 5;
    }

    int32_t count = 0;
    if (get_instance_count(handle, &count) != OMR_OK || count != 1) {
        fprintf(stderr, "unexpected instance count\n");
        runtime_destroy(handle);
        close_library(lib);
        return 6;
    }

    omr_instance_info_v1 info;
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    info.abi_version = OMR_ABI_VERSION;
    if (get_instance_info(handle, 0, &info) != OMR_OK || info.health != OMR_INSTANCE_HEALTHY || info.generation != 1) {
        fprintf(stderr, "unexpected instance diagnostics\n");
        runtime_destroy(handle);
        close_library(lib);
        return 7;
    }

    const char* message = "generic-runtime-smoke";
    omr_buffer output = {0};
    status = runtime_execute(handle, (const uint8_t*)message, strlen(message), &output);
    if (status != OMR_OK || output.length != strlen(message) || memcmp(output.data, message, output.length) != 0) {
        fprintf(stderr, "execute/echo failed: %d\n", status);
        buffer_free(&output);
        runtime_destroy(handle);
        close_library(lib);
        return 8;
    }
    buffer_free(&output);
    if (output.data != NULL || output.length != 0) {
        fprintf(stderr, "buffer_free did not clear ownership structure\n");
        runtime_destroy(handle);
        close_library(lib);
        return 9;
    }

    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    info.abi_version = OMR_ABI_VERSION;
    if (get_instance_info(handle, 0, &info) != OMR_OK ||
        info.health != OMR_INSTANCE_HEALTHY ||
        info.generation != 2 ||
        info.total_recoveries != 1) {
        fprintf(stderr, "native recovery diagnostics were unexpected\n");
        runtime_destroy(handle);
        close_library(lib);
        return 10;
    }

    if (runtime_destroy(handle) != OMR_OK) {
        fprintf(stderr, "runtime_destroy failed\n");
        close_library(lib);
        return 11;
    }

    status = get_instance_count(handle, &count);
    if (status != OMR_INVALID_HANDLE) {
        fprintf(stderr, "invalid-handle error mapping failed: %d\n", status);
        close_library(lib);
        return 12;
    }

    size_t required = 0;
    status = get_last_error(NULL, 0, &required);
    if (status != OMR_BUFFER_TOO_SMALL || required == 0) {
        fprintf(stderr, "last-error size discovery failed: %d\n", status);
        close_library(lib);
        return 13;
    }
    uint8_t* error = (uint8_t*)calloc(required + 1, 1);
    if (!error || get_last_error(error, required, &required) != OMR_OK) {
        fprintf(stderr, "last-error retrieval failed\n");
        free(error);
        close_library(lib);
        return 14;
    }
    free(error);

    close_library(lib);
    printf("native ABI smoke passed\n");
    return 0;
}
