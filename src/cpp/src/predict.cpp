#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>

#include <cxxopts.hpp>
#include <xgboost/c_api.h>

#define check_xgboost(call)                                                                                            \
    {                                                                                                                  \
        auto err = (call);                                                                                             \
        if (err != 0) {                                                                                                \
            std::fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError());            \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

#define check_stream_open(stream, fname)                                                                               \
    {                                                                                                                  \
        if (!stream) {                                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " failed to open file '" << fname << "'" << std::endl;         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

struct Args {
    std::string model_file;
    std::string data_file;
    std::string result_file;
};

Args
parse_args(int argc, char** argv)
{
    cxxopts::Options args("predict", "Apply pretrained XGBoost regression model");

    // clang-format off
    args.add_options()
        ("h,help", "show this help and exit")
        ("m,model", "file with model in binary XGBoost format", cxxopts::value<std::string>())
        ("d,data", "file with data in CSV format", cxxopts::value<std::string>())
        ("r,result", "file where to write prediction result", cxxopts::value<std::string>())
        ;
    // clang-format on

    Args opts;

    try {
        auto parsed_opts = args.parse(argc, argv);
        if (parsed_opts.count("help")) {
            std::cout << args.help() << std::endl;
            exit(EXIT_SUCCESS);
        }

        if (parsed_opts.count("model") == 0 || parsed_opts.count("data") == 0 || parsed_opts.count("result") == 0) {
            std::cerr << "Not all required option specified! Please run with -h for available options." << std::endl;
            exit(EXIT_FAILURE);
        }

        opts.model_file = parsed_opts["model"].as<std::string>();
        opts.data_file = parsed_opts["data"].as<std::string>();
        opts.result_file = parsed_opts["result"].as<std::string>();
    } catch (std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    return opts;
}

std::string
trim(const std::string& s, const std::string& whitespace = " \t")
{
    const auto start = s.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return std::string();
    }

    const auto end = s.find_last_not_of(whitespace);
    const auto range = end - start + 1;

    return s.substr(start, range);
}

std::vector<float>
tokenize(const std::string& line, char separator)
{
    std::vector<float> tokens;
    std::vector<int> marks;
    std::string s;

    marks.push_back(-1);

    for (size_t i = 0; i < line.size(); i++) {
        if (line[i] == separator) {
            marks.push_back(i);
        }
    }

    size_t count = marks.size();
    char* end = nullptr;

    marks.push_back(line.size());

    for (size_t idx = 0; idx < count; idx++) {
        s.clear();
        s.append(line, marks[idx] + 1, marks[idx + 1] - marks[idx] - 1);
        auto trimmed = trim(s);
        auto str = trimmed.c_str();
        auto v = std::strtof(str, &end);
        if (v == 0 && end == str) {
            std::cerr << "string to float convertion error on '" << s << "'" << std::endl;
            exit(EXIT_FAILURE);
        }
        tokens.push_back(v);
    }

    return tokens;
}

int
main(int argc, char** argv)
{
    auto args = parse_args(argc, argv);

    BoosterHandle booster;
    DMatrixHandle dtest = nullptr;
    bst_ulong out_len = 0;
    const float* out_result = nullptr;

    std::fstream input;

    input.open(args.data_file, std::ios::in);
    check_stream_open(input, args.data_file);

    std::vector<float> data;
    std::string line;
    bst_ulong nrow = 0, ncol = 0;

    if (std::getline(input, line)) { // skip CSV file header
        while (std::getline(input, line)) {
            auto values = tokenize(line, ',');
            if (ncol == 0) {
                ncol = values.size();
            } else {
                if (values.size() != ncol) {
                    std::cerr << "invariant failed: " << values.size() << "!=" << ncol << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            std::copy(std::begin(values), std::end(values), std::back_inserter(data));
            nrow++;
        }
    }

    check_xgboost(XGBoosterCreate(nullptr, 0, &booster));
    check_xgboost(XGBoosterLoadModel(booster, args.model_file.c_str()));
    check_xgboost(XGDMatrixCreateFromMat(data.data(), nrow, ncol, NAN, &dtest));
    check_xgboost(XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result));

    if (out_len != nrow) {
        std::cerr << "unexpected number of results" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ofstream result(args.result_file, std::ios::trunc);
    check_stream_open(result, args.result_file);

    for (bst_ulong i = 0; i < out_len; i++) {
        result << std::fixed << std::setprecision(3) << out_result[i] << std::endl;
    }

    result.close();

    check_xgboost(XGDMatrixFree(dtest));
    check_xgboost(XGBoosterFree(booster));

    return EXIT_SUCCESS;
}
