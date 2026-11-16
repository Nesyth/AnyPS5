if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_draw_records LATE)
    agc_add_draw_records_test(agc_driver_draw_records LATE)
    agc_add_stage_plans_test(agc_driver_draw_records LATE)
    agc_add_direct_draws_test(agc_driver_draw_records LATE)
    agc_add_write_validation_test(agc_driver_draw_records)
    agc_add_commit_feed_test(agc_driver_draw_records)
    agc_add_follow_ahead_test(agc_driver_draw_records)
    set_tests_properties(agc_driver_draw_records_draw_pipeline agc_driver_draw_records_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDrawRecordsDifference}")
    agc_add_env_test(agc_driver_draw_records_direct_draws_deposit agc_driver_draw_records_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 ${agcLateStackEnvironment}
            APS5_THIRD_FINISHER=1 APS5_FOURTH_FINISHER=1 APS5_FIFTH_FINISHER=1 APS5_FINISHER_DEPOSIT=1 APS5_VERIFY_FINISHER_DEPOSIT=1 APS5_VERIFY_DIRECT_DRAWS=2)
    set_tests_properties(agc_driver_draw_records_direct_draws_deposit PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};\\[finisher-deposit-verify\\] [-a-z ]+ differs:")
endif()
