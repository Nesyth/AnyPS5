if(BUILD_TESTING)
    set(agcCleanViewsEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment})
    set(agcCleanViewsDifference "${agcLeanFrontDifference};clean-currency\\] verify")
    agc_add_env_test(agc_driver_clean_resident_views_plans agc_driver_clean_resident_views_tests
            ENV ${agcCleanViewsEnvironment})
    agc_add_env_test(agc_driver_clean_resident_views_verify agc_driver_clean_resident_views_tests
            ENV ${agcCleanViewsEnvironment} APS5_VERIFY_CLEAN_CURRENCY=1)
    agc_add_env_test(agc_driver_clean_resident_views_precise agc_driver_clean_resident_views_tests
            ENV ${agcCleanViewsEnvironment} APS5_CLEAN_PRECISE_SERIALS=1 APS5_VERIFY_CLEAN_CURRENCY=2)
    agc_add_env_test(agc_driver_clean_resident_views_direct agc_driver_clean_resident_views_tests ARGS --direct
            ENV ${agcCleanViewsEnvironment} APS5_VERIFY_CLEAN_CURRENCY=2)
    agc_add_env_test(agc_driver_clean_resident_views_late agc_driver_clean_resident_views_tests
            ENV ${agcCleanViewsEnvironment} ${agcLateStackEnvironment} APS5_CLEAN_PRECISE_SERIALS=1 APS5_VERIFY_CLEAN_CURRENCY=2)
    set_tests_properties(agc_driver_clean_resident_views agc_driver_clean_resident_views_plans agc_driver_clean_resident_views_verify agc_driver_clean_resident_views_precise agc_driver_clean_resident_views_direct agc_driver_clean_resident_views_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcCleanViewsDifference}")
endif()
