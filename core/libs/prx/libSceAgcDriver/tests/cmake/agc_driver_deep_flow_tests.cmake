if(BUILD_TESTING)
    agc_add_env_test(agc_driver_deep_flow_late agc_driver_deep_flow_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment} ${agcDeepFlowEnvironment} TIMEOUT 120)
    agc_add_env_test(agc_driver_deep_flow_verify_sparse agc_driver_deep_flow_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment} APS5_DEEP_FLOW=1 APS5_VERIFY_DEEP_FLOW=7 APS5_PIPELINE_STATS=1 TIMEOUT 120)
    set_tests_properties(agc_driver_deep_flow_late agc_driver_deep_flow_verify_sparse PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDeepFlowDifference}")
    agc_add_env_test(agc_driver_deep_flow_commit_ring agc_driver_deep_flow_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment} ${agcDeepFlowEnvironment} ${agcCommitRingEnvironment} TIMEOUT 120)
    set_tests_properties(agc_driver_deep_flow_commit_ring PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDeepFlowDifference};${agcCommitRingDifference}")
endif()
