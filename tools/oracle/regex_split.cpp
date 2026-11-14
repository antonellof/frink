// The oracle's pre-tokenizer split: libllama's own `unicode_regex_split`
// (src/unicode.cpp at the pin), called with one arm's `regex_exprs`.
//
//   regex_split <patterns.hex> < corpus.hex > splits.hex
//
// <patterns.hex> holds one regex per line, hex-encoded UTF-8, in the
// order `llm_tokenizer_bpe` lists them. Each stdin line is one text,
// hex-encoded; each stdout line is that text's pieces, hex-encoded and
// separated by spaces. Empty pieces are dropped: they tokenize to
// nothing. `byte_encode` is false so the pieces are the text's own
// bytes, not the GPT-2 byte map of them.
//
// unicode_regex_split is exported from libllama but declared only in
// the private src/unicode.h, so it is declared here.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> unicode_regex_split(const std::string & text,
                                             const std::vector<std::string> & regex_exprs,
                                             bool byte_encode);

static std::string unhex(const std::string & h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        out.push_back((char) std::stoi(h.substr(i, 2), nullptr, 16));
    }
    return out;
}

static std::string hex(const std::string & s) {
    static const char * d = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out.push_back(d[c >> 4]);
        out.push_back(d[c & 15]);
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <patterns.hex> < corpus.hex\n", argv[0]);
        return 2;
    }
    std::vector<std::string> patterns;
    std::ifstream pf(argv[1]);
    for (std::string line; std::getline(pf, line);) {
        patterns.push_back(unhex(line));
    }
    for (std::string line; std::getline(std::cin, line);) {
        const auto pieces = unicode_regex_split(unhex(line), patterns, false);
        std::string out;
        for (const auto & p : pieces) {
            if (p.empty()) {
                continue;
            }
            if (!out.empty()) {
                out.push_back(' ');
            }
            out += hex(p);
        }
        std::cout << out << '\n';
    }
    return 0;
}
