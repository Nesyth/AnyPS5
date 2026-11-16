if(BUILD_TESTING)
    set(agcFinisherDepositDifference "\\[finisher-deposit-verify\\] [-a-z ]+ differs:")
    set(agcFinisherDepositEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1)
    agc_add_env_test(agc_driver_draw_finisher_deposit_verify agc_driver_draw_finisher_deposit_tests
            ENV ${agcFinisherDepositEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 300)
    agc_add_env_test(agc_driver_draw_finisher_deposit_late agc_driver_draw_finisher_deposit_tests
            ENV ${agcFinisherDepositEnvironment} ${agcLateStackEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 300)
    agc_add_env_test(agc_driver_draw_finisher_deposit_shutdown agc_driver_draw_finisher_deposit_tests ARGS --shutdown
            ENV ${agcFinisherDepositEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 60)
    agc_add_env_test(agc_driver_draw_finisher_deposit_shutdown_late agc_driver_draw_finisher_deposit_tests ARGS --shutdown
            ENV ${agcFinisherDepositEnvironment} ${agcLateStackEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 60)
    set_tests_properties(agc_driver_draw_finisher_deposit agc_driver_draw_finisher_deposit_verify agc_driver_draw_finisher_deposit_late agc_driver_draw_finisher_deposit_shutdown agc_driver_draw_finisher_deposit_shutdown_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcFinisherDepositDifference}")
    agc_add_env_test(agc_driver_draw_finisher_deposit_deep_flow agc_driver_draw_finisher_deposit_tests
            ENV ${agcFinisherDepositEnvironment} ${agcLateStackEnvironment} ${agcDeepFlowEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 300)
    set_tests_properties(agc_driver_draw_finisher_deposit_deep_flow PROPERTIES FAIL_REGULAR_EXPRESSION "${agcFinisherDepositDifference};${agcDeepFlowDifference}")
    agc_add_env_test(agc_driver_draw_finisher_deposit_commit_ring agc_driver_draw_finisher_deposit_tests
            ENV ${agcFinisherDepositEnvironment} ${agcLateStackEnvironment} ${agcDeepFlowEnvironment} ${agcCommitRingEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 300)
    agc_add_env_test(agc_driver_draw_finisher_deposit_shutdown_commit_ring agc_driver_draw_finisher_deposit_tests ARGS --shutdown
            ENV ${agcFinisherDepositEnvironment} ${agcLateStackEnvironment} ${agcDeepFlowEnvironment} ${agcCommitRingEnvironment} APS5_VERIFY_FINISHER_DEPOSIT=1 TIMEOUT 60)
    set_tests_properties(agc_driver_draw_finisher_deposit_commit_ring agc_driver_draw_finisher_deposit_shutdown_commit_ring PROPERTIES FAIL_REGULAR_EXPRESSION "${agcFinisherDepositDifference};${agcDeepFlowDifference};${agcCommitRingDifference}")
endif()
