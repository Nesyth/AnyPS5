if(BUILD_TESTING)
    set(agcLeanListsDifference "\\[lean-lists-verify\\] [-a-z ]+ differs:")
    add_test(NAME agc_driver_lean_list_batch COMMAND agc_driver_pm4_tests list-batch)
    set_tests_properties(agc_driver_lean_list_batch PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1"
            SKIP_RETURN_CODE 77 TIMEOUT 120 FAIL_REGULAR_EXPRESSION "${agcLeanListsDifference}")
    add_test(NAME agc_driver_lean_list_batch_verify COMMAND agc_driver_pm4_tests list-batch)
    set_tests_properties(agc_driver_lean_list_batch_verify PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_VERIFY_LEAN_LISTS=1"
            SKIP_RETURN_CODE 77 TIMEOUT 120 FAIL_REGULAR_EXPRESSION "${agcLeanListsDifference}")
    add_test(NAME agc_driver_lean_list_batch_timing COMMAND agc_driver_pm4_tests list-batch-timing)
    set_tests_properties(agc_driver_lean_list_batch_timing PROPERTIES ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1"
            SKIP_RETURN_CODE 77 TIMEOUT 300)
    set(agcLeanListBatchTests agc_driver_lean_list_batch agc_driver_lean_list_batch_verify agc_driver_lean_list_batch_timing)
    foreach(agcListMode IN ITEMS rewind-list-cache rewind-lists rewind-kept-tails rewind-chain rewind-labels rewind-presents)
        string(REPLACE "-" "_" agcListTest "agc_driver_lean_list_batch_${agcListMode}")
        add_test(NAME ${agcListTest} COMMAND agc_driver_pm4_tests ${agcListMode})
        set_tests_properties(${agcListTest} PROPERTIES
                ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_LEAN_FRONT=1;APS5_VERIFY_LEAN_FRONT=2;APS5_LEAN_LIST_BATCH=1;APS5_VERIFY_LEAN_LISTS=1"
                SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcLeanListsDifference}")
        list(APPEND agcLeanListBatchTests ${agcListTest})
        if(agcListMode STREQUAL "rewind-list-cache")
            continue()
        endif()
        string(REPLACE "-" "_" agcDirectTest "agc_driver_lean_list_direct_${agcListMode}")
        add_test(NAME ${agcDirectTest} COMMAND agc_driver_pm4_tests ${agcListMode})
        set_tests_properties(${agcDirectTest} PROPERTIES
                ENVIRONMENT "ANYPS5_NO_SHADER_CACHE=1;APS5_LEAN_FRONT=1;APS5_VERIFY_LEAN_FRONT=2;APS5_LEAN_LIST_DIRECT=1"
                SKIP_RETURN_CODE 77 TIMEOUT 60 FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference}")
        list(APPEND agcLeanListBatchTests ${agcDirectTest})
    endforeach()
    if(WIN32)
        set_tests_properties(${agcLeanListBatchTests} PROPERTIES ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:libc>")
    endif()
endif()
