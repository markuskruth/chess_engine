// test_encoding.cpp — golden-file validation for ChessEnv functions
//
// Phase 2.2 workflow:
//   1. python generate_test_vectors.py          → test_vectors.json
//   2. Build this exe, then: test_encoding.exe [path/to/test_vectors.json]
//
// Each JSON entry:
//   { "fen": "...", "state": [1280 floats], "mask": [4672 ints] }

#include "ChessEnv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>   // strtof

// ── Minimal JSON parser ───────────────────────────────────────────────────────
// Parses exactly the format written by Python's json.dump() — no library needed.

struct TestVector {
    std::string        fen;
    std::vector<float> state;   // 1280 floats
    std::vector<bool>  mask;    // 4672 bools (stored as 0/1 ints in JSON)
};

static std::vector<TestVector> load_test_vectors(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[error] Cannot open: " << path << "\n";
        return {};
    }

    // Read entire file into a mutable buffer so we can use strtof() in-place.
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    std::vector<TestVector> result;

    // Working pointer into the buffer.
    char* p   = json.data();
    char* end = p + json.size();

    auto skip_ws = [&]() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
    };

    // Read a JSON string starting with '"'; leaves p after closing '"'.
    auto read_string = [&]() -> std::string {
        std::string s;
        if (*p == '"') ++p;  // skip opening "
        while (p < end && *p != '"') {
            if (*p == '\\') { ++p; }  // skip escape char (not needed for FEN)
            if (p < end) s += *p++;
        }
        if (p < end) ++p;  // skip closing "
        return s;
    };

    // Read a JSON array of floats: [...]; leaves p after ']'.
    auto read_float_array = [&](std::vector<float>& arr) {
        arr.reserve(STATE_CHANNELS * BOARD_SZ * BOARD_SZ);
        if (*p == '[') ++p;
        while (p < end && *p != ']') {
            skip_ws();
            if (*p == ']') break;
            char* next;
            arr.push_back(std::strtof(p, &next));
            p = next;
            skip_ws();
            if (*p == ',') ++p;
        }
        if (p < end) ++p;  // skip ']'
    };

    // Read a JSON array of 0/1 ints: [...]; leaves p after ']'.
    auto read_bool_array = [&](std::vector<bool>& arr) {
        arr.reserve(ACTION_DIM);
        if (*p == '[') ++p;
        while (p < end && *p != ']') {
            skip_ws();
            if (*p == ']') break;
            arr.push_back(*p == '1');
            // advance past the digit
            while (p < end && *p != ',' && *p != ']') ++p;
            if (*p == ',') ++p;
        }
        if (p < end) ++p;  // skip ']'
    };

    // Find outer '['.
    while (p < end && *p != '[') ++p;
    if (p >= end) return result;
    ++p;

    // Iterate over object entries.
    while (p < end) {
        skip_ws();
        if (*p == ']') break;   // end of outer array
        if (*p == ',') { ++p; continue; }
        if (*p != '{') { ++p; continue; }
        ++p;  // skip '{'

        TestVector vec;
        vec.state.reserve(STATE_CHANNELS * BOARD_SZ * BOARD_SZ);
        vec.mask.reserve(ACTION_DIM);

        // Parse key-value pairs inside the object.
        while (p < end && *p != '}') {
            skip_ws();
            if (*p == ',') { ++p; continue; }
            if (*p != '"') { ++p; continue; }

            std::string key = read_string();

            skip_ws();
            if (*p == ':') ++p;
            skip_ws();

            if (key == "fen") {
                vec.fen = read_string();
            } else if (key == "state") {
                read_float_array(vec.state);
            } else if (key == "mask") {
                read_bool_array(vec.mask);
            } else {
                // Unknown key — skip value (string, number, or nested array)
                if (*p == '"') { read_string(); }
                else { while (p < end && *p != ',' && *p != '}') ++p; }
            }
        }

        if (!vec.fen.empty()) result.push_back(std::move(vec));
        if (p < end && *p == '}') ++p;
    }

    return result;
}

// ── Test helpers ──────────────────────────────────────────────────────────────

static bool test_state(int idx, const TestVector& vec) {
    chess::Board board(vec.fen);
    auto t = ChessEnv::encode_state(board).to(torch::kCPU).contiguous();
    const float* data = t.data_ptr<float>();

    const int n = static_cast<int>(vec.state.size());
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(data[i] - vec.state[i]) > 1e-5f) {
            int ch  = i / 64;
            int row = (i % 64) / 8;
            int col = i % 8;
            std::cerr << "[state #" << idx << "] ch=" << ch
                      << " row=" << row << " col=" << col
                      << "  got=" << data[i]
                      << " expected=" << vec.state[i] << "\n";
            ok = false;
            if (!ok && i > n / 10) {
                std::cerr << "  ... (further mismatches suppressed)\n";
                break;
            }
        }
    }
    return ok;
}

static bool test_mask(int idx, const TestVector& vec) {
    chess::Board board(vec.fen);
    auto mask = ChessEnv::get_action_mask(board);

    bool ok = true;
    for (int i = 0; i < ACTION_DIM; ++i) {
        if (mask[i] != vec.mask[i]) {
            int row   = i / (COLS * ACTION_PLANES);
            int col   = (i / ACTION_PLANES) % COLS;
            int plane = i % ACTION_PLANES;
            std::cerr << "[mask  #" << idx << "] idx=" << i
                      << " row=" << row << " col=" << col << " plane=" << plane
                      << "  got=" << mask[i]
                      << " expected=" << vec.mask[i] << "\n";
            ok = false;
            if (!ok && i > ACTION_DIM / 10) {
                std::cerr << "  ... (further mismatches suppressed)\n";
                break;
            }
        }
    }
    return ok;
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "=== ChessEnv encoding golden-file test ===\n\n";

    std::string json_path = (argc > 1) ? argv[1] : "../test_vectors.json";
    std::cout << "Loading: " << json_path << "\n";

    auto vectors = load_test_vectors(json_path);

    if (vectors.empty()) {
        std::cout << "[info] No test vectors loaded.\n"
                  << "[info] Generate them with:\n"
                  << "[info]   python generate_test_vectors.py\n"
                  << "[info] then re-run this executable.\n";
        return 0;
    }

    std::cout << "Loaded " << vectors.size() << " test vectors.\n\n";

    // Validate array sizes
    for (int i = 0; i < static_cast<int>(vectors.size()); ++i) {
        if (static_cast<int>(vectors[i].state.size()) != STATE_CHANNELS * BOARD_SZ * BOARD_SZ) {
            std::cerr << "[error] vector #" << i << " state size "
                      << vectors[i].state.size() << " != 1280\n";
            return 1;
        }
        if (static_cast<int>(vectors[i].mask.size()) != ACTION_DIM) {
            std::cerr << "[error] vector #" << i << " mask size "
                      << vectors[i].mask.size() << " != 4672\n";
            return 1;
        }
    }

    int pass = 0, fail = 0;
    for (int i = 0; i < static_cast<int>(vectors.size()); ++i) {
        bool state_ok = test_state(i, vectors[i]);
        bool mask_ok  = test_mask (i, vectors[i]);
        if (state_ok && mask_ok) {
            ++pass;
        } else {
            ++fail;
            std::cerr << "[FAIL] vector #" << i
                      << "  FEN: " << vectors[i].fen << "\n";
        }
    }

    std::cout << "\nResults: " << pass << " passed, " << fail << " failed"
              << " (out of " << vectors.size() << ")\n";
    return (fail == 0) ? 0 : 1;
}
