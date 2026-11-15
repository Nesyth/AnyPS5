if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_lean_finisher_ring LATE)
    agc_add_draw_records_test(agc_driver_lean_finisher_ring)
    agc_add_env_test(agc_driver_lean_finisher_ring_five_deposit agc_driver_lean_finisher_ring_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment}
            APS5_THIRD_FINISHER=1 APS5_FOURTH_FINISHER=1 APS5_FIFTH_FINISHER=1 APS5_FINISHER_DEPOSIT=1 APS5_VERIFY_FINISHER_DEPOSIT=1)
    set_tests_properties(agc_driver_lean_finisher_ring_five_deposit PROPERTIES FAIL_REGULAR_EXPRESSION "\\[finisher-deposit-verify\\] [-a-z ]+ differs:")
    agc_add_env_test(agc_driver_lean_finisher_ring_deep_flow agc_driver_lean_finisher_ring_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment} ${agcDeepFlowEnvironment})
    set_tests_properties(agc_driver_lean_finisher_ring_deep_flow PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDeepFlowDifference}")
    agc_add_env_test(agc_driver_lean_finisher_ring_five_deposit_deep_flow agc_driver_lean_finisher_ring_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment}
            APS5_THIRD_FINISHER=1 APS5_FOURTH_FINISHER=1 APS5_FIFTH_FINISHER=1 APS5_FINISHER_DEPOSIT=1 APS5_VERIFY_FINISHER_DEPOSIT=1 ${agcDeepFlowEnvironment})
    set_tests_properties(agc_driver_lean_finisher_ring_five_deposit_deep_flow PROPERTIES FAIL_REGULAR_EXPRESSION "\\[finisher-deposit-verify\\] [-a-z ]+ differs:;${agcDeepFlowDifference}")
endif()
