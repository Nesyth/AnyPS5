if(BUILD_TESTING AND TARGET agc_driver_heap_stage_replay_tests)
    agc_add_env_test(agc_driver_heap_stage_replay_checked agc_driver_heap_stage_replay_tests
            ARGS checked ENV ANYPS5_NO_SHADER_CACHE=1)
    agc_add_env_test(agc_driver_heap_stage_replay_shadow agc_driver_heap_stage_replay_tests
            ARGS shadow ENV ANYPS5_NO_SHADER_CACHE=1)
    agc_add_env_test(agc_driver_heap_stage_replay_census agc_driver_heap_stage_replay_tests
            ARGS replay ENV ANYPS5_NO_SHADER_CACHE=1 APS5_REPLAY_CENSUS=1 APS5_THREAD_REPORT=1)
endif()
