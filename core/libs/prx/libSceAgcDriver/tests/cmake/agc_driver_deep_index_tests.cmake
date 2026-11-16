if(BUILD_TESTING)
    agc_add_env_test(agc_driver_deep_index_census agc_driver_deep_index_tests
            ENV ANYPS5_NO_SHADER_CACHE=1 APS5_PIPELINE_STATS=1 TIMEOUT 120)
endif()
