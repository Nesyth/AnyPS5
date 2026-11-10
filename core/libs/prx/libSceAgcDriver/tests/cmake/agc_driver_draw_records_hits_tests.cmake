if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_draw_records_hits LATE)
    agc_add_draw_records_test(agc_driver_draw_records_hits LATE)
    agc_add_stage_plans_test(agc_driver_draw_records_hits LATE)
    agc_add_direct_draws_test(agc_driver_draw_records_hits LATE)
    set_tests_properties(agc_driver_draw_records_hits_draw_pipeline agc_driver_draw_records_hits_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDrawRecordsDifference}")
    agc_add_env_test(agc_driver_draw_records_hits_direct_buffers agc_driver_draw_records_hits_tests ARGS --direct-buffers
            ENV ANYPS5_NO_SHADER_CACHE=1)
    set(agcDirectBuffersEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 APS5_DIRECT_BUFFERS=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_buffers_draws agc_driver_draw_records_hits_tests
            ENV ${agcDirectBuffersEnvironment})
    agc_add_env_test(agc_driver_draw_records_hits_direct_buffers_verify agc_driver_draw_records_hits_tests
            ENV ${agcDirectBuffersEnvironment} APS5_VERIFY_DIRECT_BUFFERS=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_buffers_late agc_driver_draw_records_hits_tests
            ENV ${agcDirectBuffersEnvironment} ${agcLateStackEnvironment} APS5_VERIFY_DIRECT_BUFFERS=2)
    set_tests_properties(agc_driver_draw_records_hits_direct_buffers_draws agc_driver_draw_records_hits_direct_buffers_verify agc_driver_draw_records_hits_direct_buffers_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference}")
    set(agcDirectInputsDifference "\\[direct-inputs\\] verify:")
    agc_add_env_test(agc_driver_draw_records_hits_direct_inputs agc_driver_draw_records_hits_tests ARGS --direct-inputs
            ENV ANYPS5_NO_SHADER_CACHE=1)
    set(agcDirectInputsEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 APS5_DIRECT_INPUTS=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_inputs_draws agc_driver_draw_records_hits_tests
            ENV ${agcDirectInputsEnvironment})
    agc_add_env_test(agc_driver_draw_records_hits_direct_inputs_verify agc_driver_draw_records_hits_tests
            ENV ${agcDirectInputsEnvironment} APS5_VERIFY_DIRECT_INPUTS=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_inputs_late agc_driver_draw_records_hits_tests
            ENV ${agcDirectInputsEnvironment} APS5_DIRECT_BUFFERS=1 ${agcLateStackEnvironment} APS5_VERIFY_DIRECT_INPUTS=2)
    set_tests_properties(agc_driver_draw_records_hits_direct_inputs agc_driver_draw_records_hits_direct_inputs_draws agc_driver_draw_records_hits_direct_inputs_verify agc_driver_draw_records_hits_direct_inputs_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcDirectInputsDifference}")
    agc_add_env_test(agc_driver_draw_records_hits_inject agc_driver_draw_records_hits_tests ARGS --inject
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DRAW_RECORDS=1 APS5_VERIFY_DRAW_RECORDS=1)
    agc_add_env_test(agc_driver_draw_records_hits_inject_commit agc_driver_draw_records_hits_tests ARGS --inject-commit
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DRAW_RECORDS=1 APS5_VERIFY_DRAW_RECORDS=2)
    agc_add_env_test(agc_driver_draw_records_hits_stage_plans_inject agc_driver_draw_records_hits_tests ARGS --inject-plans
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_STAGE_PLANS=1 APS5_VERIFY_STAGE_PLANS=1)
    agc_add_env_test(agc_driver_draw_records_hits_stage_plans_textures_threads agc_driver_draw_records_hits_tests ARGS --textures
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_STAGE_PLANS=1 APS5_THREAD_REPORT=1 APS5_VERIFY_CLEAN_CURRENCY=2)
    agc_add_env_test(agc_driver_draw_records_hits_stage_plans_textures_full_proofs agc_driver_draw_records_hits_tests ARGS --textures
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_STAGE_PLANS=1 APS5_NO_PLAN_CLEAN_PROOFS=1)
    set_tests_properties(agc_driver_draw_records_hits_inject agc_driver_draw_records_hits_inject_commit agc_driver_draw_records_hits_stage_plans_inject PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference}")
    set_tests_properties(agc_driver_draw_records_hits_stage_plans_textures_threads agc_driver_draw_records_hits_stage_plans_textures_full_proofs PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};clean-currency\\] verify")
endif()
