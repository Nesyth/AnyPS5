if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_pipelined_predication_read LATE)
endif()

if(BUILD_TESTING)
    agc_add_gpu_heap_test(agc_driver_pipelined_predication_read)
endif()
