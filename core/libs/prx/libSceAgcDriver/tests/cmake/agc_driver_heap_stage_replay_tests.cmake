if(BUILD_TESTING AND TARGET agc_driver_heap_stage_replay_tests)
    agc_add_env_test(agc_driver_heap_stage_replay_checked agc_driver_heap_stage_replay_tests
            ARGS checked ENV ANYPS5_NO_SHADER_CACHE=1)
    agc_add_env_test(agc_driver_heap_stage_replay_shadow agc_driver_heap_stage_replay_tests
            ARGS shadow ENV ANYPS5_NO_SHADER_CACHE=1)
    agc_add_env_test(agc_driver_heap_stage_replay_census agc_driver_heap_stage_replay_tests
            ARGS replay ENV ANYPS5_NO_SHADER_CACHE=1 APS5_REPLAY_CENSUS=1 APS5_THREAD_REPORT=1)
    set_tests_properties(agc_driver_heap_stage_replay_census PROPERTIES
            PASS_REGULAR_EXPRESSION "\\[threads\\] .*committer.*\\[replay-heap-stage\\] .*stages replayed.*\\[replay-census\\] .*stage refreshes.*heap stage replay record tests passed")
endif()

if(BUILD_TESTING AND TARGET agc_driver_draw_cache_reuse_tests)
    set(agcHeapReplayDrawEnvironment ANYPS5_NO_SHADER_CACHE=1 APS5_GPU_HEAP=1 APS5_VERIFY_GPU_HEAP=1 APS5_DRAW_PIPELINE=1 APS5_PIPELINED_DRAWS=32 APS5_PIPELINE_DISPATCH=1 APS5_REPLAY_HEAP_STAGE=1)
    agc_add_env_test(agc_driver_draw_cache_reuse_heap_replay agc_driver_draw_cache_reuse_tests
            ARGS replay ENV ${agcHeapReplayDrawEnvironment} APS5_VERIFY_REPLAY_HEAP_STAGE=0 APS5_VERIFY_REPLAY_STRICT=0)
    agc_add_env_test(agc_driver_draw_cache_reuse_heap_replay_checked agc_driver_draw_cache_reuse_tests
            ARGS replay ENV ${agcHeapReplayDrawEnvironment} APS5_VERIFY_REPLAY_HEAP_STAGE=2 APS5_VERIFY_REPLAY_STRICT=1)
    agc_add_env_test(agc_driver_draw_cache_reuse_heap_replay_shadow agc_driver_draw_cache_reuse_tests
            ARGS replay ENV ${agcHeapReplayDrawEnvironment} APS5_VERIFY_REPLAY_HEAP_STAGE=1 APS5_VERIFY_REPLAY_STRICT=0)
    set_tests_properties(agc_driver_draw_cache_reuse_heap_replay agc_driver_draw_cache_reuse_heap_replay_checked agc_driver_draw_cache_reuse_heap_replay_shadow PROPERTIES FAIL_REGULAR_EXPRESSION "FATAL:")
endif()
