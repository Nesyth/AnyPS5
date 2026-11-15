if(BUILD_TESTING)
    agc_add_draw_pipeline_test(agc_driver_draw_records_hits LATE)
    agc_add_draw_records_test(agc_driver_draw_records_hits LATE)
    agc_add_stage_plans_test(agc_driver_draw_records_hits LATE)
    agc_add_direct_draws_test(agc_driver_draw_records_hits LATE)
    set_tests_properties(agc_driver_draw_records_hits_draw_pipeline agc_driver_draw_records_hits_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcDrawRecordsDifference}")
    agc_add_env_test(agc_driver_draw_records_hits_direct_draws_deposit agc_driver_draw_records_hits_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 ${agcLateStackEnvironment}
            APS5_THIRD_FINISHER=1 APS5_FOURTH_FINISHER=1 APS5_FIFTH_FINISHER=1 APS5_FINISHER_DEPOSIT=1 APS5_VERIFY_FINISHER_DEPOSIT=1 APS5_VERIFY_DIRECT_DRAWS=2)
    set_tests_properties(agc_driver_draw_records_hits_direct_draws_deposit PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};\\[finisher-deposit-verify\\] [-a-z ]+ differs:")
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
    set(agcDirectRepeatDifference "\\[direct-repeat-verify\\] ")
    set(agcDirectRepeatEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 APS5_DIRECT_BUFFERS=1 APS5_DIRECT_INPUTS=1 APS5_DIRECT_REPEAT=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_repeat agc_driver_draw_records_hits_tests
            ENV ${agcDirectRepeatEnvironment} APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_repeat_verify agc_driver_draw_records_hits_tests
            ENV ${agcDirectRepeatEnvironment} APS5_VERIFY_DIRECT_REPEAT=1)
    agc_add_env_test(agc_driver_draw_records_hits_direct_repeat_late agc_driver_draw_records_hits_tests
            ENV ${agcDirectRepeatEnvironment} ${agcLateStackEnvironment} APS5_VERIFY_DIRECT_REPEAT=3)
    set_tests_properties(agc_driver_draw_records_hits_direct_repeat agc_driver_draw_records_hits_direct_repeat_verify agc_driver_draw_records_hits_direct_repeat_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcDirectInputsDifference};${agcDirectRepeatDifference}")
    set(agcCommitBindingsDifference "\\[commit-bindings-verify\\] ")
    set(agcCommitBindingsEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 APS5_DIRECT_BUFFERS=1 APS5_DIRECT_INPUTS=1 APS5_COMMIT_BINDINGS=1 APS5_COMMIT_BINDINGS_STAMPS=1 APS5_COMMIT_BINDINGS_SETS=1 APS5_COMMIT_BINDINGS_DYNAMIC=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_bindings agc_driver_draw_records_hits_tests
            ENV ${agcCommitBindingsEnvironment} APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_bindings_verify agc_driver_draw_records_hits_tests
            ENV ${agcCommitBindingsEnvironment} APS5_VERIFY_COMMIT_BINDINGS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_bindings_late agc_driver_draw_records_hits_tests
            ENV ${agcCommitBindingsEnvironment} ${agcLateStackEnvironment} APS5_VERIFY_COMMIT_BINDINGS=3)
    set_tests_properties(agc_driver_draw_records_hits_commit_bindings agc_driver_draw_records_hits_commit_bindings_verify agc_driver_draw_records_hits_commit_bindings_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcCommitBindingsDifference}")
    set(agcCommitFragmentsDifference "\\[commit-fragments-verify\\] ")
    set(agcCommitFragmentsEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 ${agcLeanFrontEnvironment} APS5_DIRECT_DRAWS=1 APS5_DIRECT_BUFFERS=1 APS5_DIRECT_INPUTS=1 APS5_COMMIT_BINDINGS=1 APS5_COMMIT_BINDINGS_STAMPS=1 APS5_COMMIT_FRAGMENTS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_fragments agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_fragments_verify agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_VERIFY_COMMIT_FRAGMENTS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_fragments_replayed agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_commit_fragments_late agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} ${agcLateStackEnvironment} APS5_COMMIT_BINDINGS_SETS=1 APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_VERIFY_COMMIT_FRAGMENTS=3)
    set_tests_properties(agc_driver_draw_records_hits_commit_fragments agc_driver_draw_records_hits_commit_fragments_verify agc_driver_draw_records_hits_commit_fragments_replayed agc_driver_draw_records_hits_commit_fragments_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcCommitBindingsDifference};${agcCommitFragmentsDifference}")
    set(agcExpressCommitsDifference "\\[express-commits-verify\\] ")
    agc_add_env_test(agc_driver_draw_records_hits_express_commits agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_DIRECT_CLEAN_PROOFS=1 APS5_CLEAN_PRECISE_SERIALS=1 APS5_EXPRESS_COMMITS=1 APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_express_commits_verify agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_DIRECT_CLEAN_PROOFS=1 APS5_CLEAN_PRECISE_SERIALS=1 APS5_EXPRESS_COMMITS=1 APS5_VERIFY_EXPRESS_COMMITS=1 APS5_VERIFY_COMMIT_FRAGMENTS=7)
    agc_add_env_test(agc_driver_draw_records_hits_express_commits_late agc_driver_draw_records_hits_tests
            ENV ${agcCommitFragmentsEnvironment} ${agcLateStackEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_DIRECT_CLEAN_PROOFS=1 APS5_CLEAN_PRECISE_SERIALS=1 APS5_EXPRESS_COMMITS=1 APS5_VERIFY_EXPRESS_COMMITS=3)
    set_tests_properties(agc_driver_draw_records_hits_express_commits agc_driver_draw_records_hits_express_commits_verify agc_driver_draw_records_hits_express_commits_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcCommitBindingsDifference};${agcCommitFragmentsDifference};${agcExpressCommitsDifference}")
    set(agcDrawTicketsDifference "\\[draw-tickets-verify\\] [-a-z ]+ differs:")
    set(agcDrawTicketsEnvironment ${agcCommitFragmentsEnvironment} APS5_ASYNC_SUBMIT=1 APS5_DEFERRED_ENCODE=1 APS5_DIRECT_CLEAN_PROOFS=1 APS5_CLEAN_PRECISE_SERIALS=1 APS5_EXPRESS_COMMITS=1 APS5_RUN_FOLLOW=1 APS5_DIRECT_HANDOFF=1 APS5_DRAW_TICKETS=1)
    agc_add_env_test(agc_driver_draw_records_hits_draw_tickets agc_driver_draw_records_hits_tests
            ENV ${agcDrawTicketsEnvironment} APS5_VERIFY_DRAW_TICKETS=1 APS5_JOB_ALLOCATIONS=1 APS5_PIPELINE_STATS=1)
    agc_add_env_test(agc_driver_draw_records_hits_draw_tickets_verify agc_driver_draw_records_hits_tests
            ENV ${agcDrawTicketsEnvironment} APS5_TICKET_PREFETCH=1 APS5_VERIFY_DRAW_TICKETS=1 APS5_VERIFY_EXPRESS_COMMITS=3 APS5_VERIFY_COMMIT_FRAGMENTS=3 APS5_VERIFY_DIRECT_BUFFERS=3 APS5_VERIFY_DIRECT_INPUTS=3 APS5_VERIFY_DRAW_RECORDS=5)
    agc_add_env_test(agc_driver_draw_records_hits_draw_tickets_late agc_driver_draw_records_hits_tests
            ENV ${agcDrawTicketsEnvironment} ${agcLateStackEnvironment} APS5_FINISHER_DEPOSIT=1 APS5_IDLE_HIT_RELEASE=1 APS5_TICKET_PREFETCH=1 APS5_VERIFY_DRAW_TICKETS=2 APS5_VERIFY_EXPRESS_COMMITS=3)
    set_tests_properties(agc_driver_draw_records_hits_draw_tickets agc_driver_draw_records_hits_draw_tickets_verify agc_driver_draw_records_hits_draw_tickets_late PROPERTIES FAIL_REGULAR_EXPRESSION "${agcLeanFrontDifference};${agcDrawRecordsDifference};${agcStagePlansDifference};${agcDirectDrawsDifference};${agcDirectDrawsBuildDifference};${agcDirectBuffersDifference};${agcCommitBindingsDifference};${agcCommitFragmentsDifference};${agcExpressCommitsDifference};${agcDrawTicketsDifference}")
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
