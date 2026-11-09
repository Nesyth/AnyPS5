if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_draw_records_hits LATE)
    agc_add_draw_records_test(agc_driver_draw_records_hits LATE)
    set_tests_properties(agc_driver_draw_records_hits_draw_pipeline agc_driver_draw_records_hits_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDrawRecordsDifference}")
    agc_add_env_test(agc_driver_draw_records_hits_inject agc_driver_draw_records_hits_tests ARGS --inject
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DRAW_RECORDS=1 APS5_VERIFY_DRAW_RECORDS=1)
    agc_add_env_test(agc_driver_draw_records_hits_inject_commit agc_driver_draw_records_hits_tests ARGS --inject-commit
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DRAW_RECORDS=1 APS5_VERIFY_DRAW_RECORDS=2)
    set_tests_properties(agc_driver_draw_records_hits_inject agc_driver_draw_records_hits_inject_commit PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference}")
endif()
