if(BUILD_TESTING)
    set(agcCommitRingPipeline ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment})
    agc_add_env_test(agc_driver_commit_ring_off agc_driver_commit_ring_tests
            ENV ${agcCommitRingPipeline} ${agcDeepFlowEnvironment} TIMEOUT 180)
    agc_add_env_test(agc_driver_commit_ring_on agc_driver_commit_ring_tests
            ENV ${agcCommitRingPipeline} ${agcDeepFlowEnvironment} ${agcCommitRingEnvironment} TIMEOUT 180)
    agc_add_env_test(agc_driver_commit_ring_sparse agc_driver_commit_ring_tests
            ENV ${agcCommitRingPipeline} APS5_DEEP_FLOW=1 APS5_COMMIT_RING=1 APS5_VERIFY_COMMIT_RING=7 APS5_PIPELINE_STATS=1 TIMEOUT 180)
    set_tests_properties(agc_driver_commit_ring_off agc_driver_commit_ring_on agc_driver_commit_ring_sparse PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDeepFlowDifference};${agcCommitRingDifference}")
endif()
