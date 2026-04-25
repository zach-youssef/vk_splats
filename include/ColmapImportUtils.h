// Zach Youssef, 4/21/26
// Header defining utility functions for parsing colmap data

#pragma once

#include <vector>
#include <functional>

#include <fstream>
#include <sstream>

// Trim from the end (in place)
// Taken from https://stackoverflow.com/questions/216823/how-can-i-trim-a-stdstring
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}


template<typename T>
std::vector<T> readColmapFile(const std::string& path, std::function<T(const std::vector<std::string>&)> lineParser);

template<typename T>
std::vector<T> readColmapFile(const std::string& path, std::function<T(const std::vector<std::string>&)> lineParser) {
    std::ifstream colmapFile;
    colmapFile.open(path);
    std::string line;
    std::vector<T> results;
    while(std::getline(colmapFile, line)) {
        std::vector<std::string> lineContents;
        std::stringstream lineStream(line);
        std::string val;
        while(std::getline(lineStream, val, ' ')) {
            // Skip layout comments
            if (val.compare("#") == 0) {
                break;
            }

            rtrim(val);

            lineContents.push_back(val);
        }
        if (val.compare("#") != 0 && lineContents.size() < 21) {
            results.push_back(lineParser(lineContents));
        }
    }
    return results;
}