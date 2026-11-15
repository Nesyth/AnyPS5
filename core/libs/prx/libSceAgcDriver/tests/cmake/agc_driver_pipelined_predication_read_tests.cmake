if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_pipelined_predication_read LATE)
    agc_add_deep_flow_test(agc_driver_pipelined_predication_read)
endif()

if(BUILD_TESTING)
    agc_add_gpu_heap_test(agc_driver_pipelined_predication_read)
endif()
