cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(chess_rl_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── LibTorch ──────────────────────────────────────────────────────────────────
set(Torch_DIR "C:/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
# Propagate LibTorch compiler flags (enables /EHsc on MSVC, etc.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Threading (pthreads on Linux, native on MSVC — no-op but keeps CMakeLists portable) ──
find_package(Threads REQUIRED)

# ── chess-library (Disservin, header-only, ~single header) ───────────────────
# Downloads automatically on first cmake configure.
# If behind a firewall, clone manually and point GIT_REPOSITORY to a local path.
include(FetchContent)
FetchContent_Declare(
    chess_lib
    GIT_REPOSITORY https://github.com/Disservin/chess-library.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(chess_lib)
# Header lives at <chess_lib_SOURCE_DIR>/src/chess.hpp
set(CHESS_INCLUDE_DIR ${chess_lib_SOURCE_DIR}/include)

# ── Common include dirs ───────────────────────────────────────────────────────
set(COMMON_INCLUDES
    ${CHESS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # allows #include "ChessEnv.h" from tests/
)

# ── Shared sources (ChessEnv + MCTS + SelfPlay) ───────────────────────────────
set(LIB_SOURCES
    src/ChessEnv.cpp
    src/MCTS.cpp
    src/SelfPlay.cpp
)

# ── Phase 4: parallel / async sources ────────────────────────────────────────
set(PARALLEL_SOURCES
    src/Evaluator.cpp
    src/CentralEvaluator.cpp
    src/AsyncMCTS.cpp
    src/ParallelSelfPlay.cpp
)

# ── Executable: selfplay ──────────────────────────────────────────────────────
add_executable(selfplay src/main.cpp ${LIB_SOURCES} ${PARALLEL_SOURCES})
target_include_directories(selfplay PRIVATE ${COMMON_INCLUDES})
target_link_libraries(selfplay "${TORCH_LIBRARIES}" Threads::Threads)

# ── Executable: test_encoding ─────────────────────────────────────────────────
# Validates that C++ encode_state / get_action_mask / apply_action match
# the golden vectors produced by Python's generate_test_vectors.py (Phase 2).
add_executable(test_encoding tests/test_encoding.cpp src/ChessEnv.cpp)
target_include_directories(test_encoding PRIVATE ${COMMON_INCLUDES})
target_link_libraries(test_encoding "${TORCH_LIBRARIES}" Threads::Threads)

# ── Executable: test_mcts ─────────────────────────────────────────────────────
# Phase 3 validation: Node structure (Tier 1, no model) + MCTS simulations
# (Tier 2, requires a TorchScript model as the first argument).
add_executable(test_mcts tests/test_mcts.cpp ${LIB_SOURCES})
target_include_directories(test_mcts PRIVATE ${COMMON_INCLUDES})
target_link_libraries(test_mcts "${TORCH_LIBRARIES}" Threads::Threads)

# ── Executable: test_parallel ─────────────────────────────────────────────────
# Phase 4 validation: EvalQueue unit tests (Tier 1) + AsyncMCTS with stub
# evaluator (Tier 2) + full ParallelSelfPlay integration (Tier 3).
add_executable(test_parallel tests/test_parallel.cpp ${LIB_SOURCES} ${PARALLEL_SOURCES})
target_include_directories(test_parallel PRIVATE ${COMMON_INCLUDES})
target_link_libraries(test_parallel "${TORCH_LIBRARIES}" Threads::Threads)

# ── Windows: copy LibTorch DLLs next to each executable ──────────────────────
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    foreach(EXE_TARGET selfplay test_encoding test_mcts test_parallel)
        add_custom_command(TARGET ${EXE_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:${EXE_TARGET}>
        )
    endforeach()
endif()
