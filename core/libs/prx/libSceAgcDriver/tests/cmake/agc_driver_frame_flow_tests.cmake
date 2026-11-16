if(BUILD_TESTING)
    set(agcFrameFlowPipeline ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLateStackEnvironment} ${agcLeanFrontEnvironment})
    agc_add_env_test(agc_driver_frame_flow_on agc_driver_frame_flow_tests
            ENV ${agcFrameFlowPipeline} ${agcDeepFlowEnvironment} ${agcFrameFlowEnvironment} TIMEOUT 120)
    agc_add_env_test(agc_driver_frame_flow_sparse agc_driver_frame_flow_tests
            ENV ${agcFrameFlowPipeline} APS5_DEEP_FLOW=1 APS5_FRAME_FLOW=1 APS5_VERIFY_FRAME_FLOW=3 APS5_PIPELINE_STATS=1 TIMEOUT 120)
    agc_add_env_test(agc_driver_frame_flow_ordered agc_driver_frame_flow_tests
            ENV ${agcFrameFlowPipeline} ${agcDeepFlowEnvironment} ${agcFrameFlowEnvironment} ${agcOrderedSyncsEnvironment} APS5_PIPELINE_DISPATCH_MISSES=1 TIMEOUT 120)
    agc_add_env_test(agc_driver_frame_flow_ordered_sparse agc_driver_frame_flow_tests
            ENV ${agcFrameFlowPipeline} APS5_DEEP_FLOW=1 APS5_FRAME_FLOW=1 ${agcOrderedSyncsEnvironment} APS5_PIPELINE_DISPATCH_MISSES=1 APS5_VERIFY_FRAME_FLOW=3 APS5_PIPELINE_STATS=1 TIMEOUT 120)
    set_tests_properties(agc_driver_frame_flow_on agc_driver_frame_flow_sparse agc_driver_frame_flow_ordered agc_driver_frame_flow_ordered_sparse PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDeepFlowDifference};${agcFrameFlowDifference}")
endif()
