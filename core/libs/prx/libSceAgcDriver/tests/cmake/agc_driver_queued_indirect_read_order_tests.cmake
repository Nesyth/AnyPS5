if(BUILD_TESTING)
    set_tests_properties(agc_driver_queued_indirect_read_order PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    agc_add_deep_flow_test(agc_driver_queued_indirect_read_order)
endif()
