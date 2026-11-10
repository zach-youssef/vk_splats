// splat_view's MainInputs: the scene's opener, the failure exit (every other check is in the app),
// and --inspect's accum written to $SPLAT_VIEW_ACCUM for the closed-form oracle
// (web-check/view_oracle_check.cpp).
//
// The opener (the graph's stream input) finds the file as other assets are found
// (VulkanContext::findAsset) and opens it by its first bytes:
//   - an INRIA PLY ("ply"): read whole (lib/native/ply_reader.h) and encoded in full, every group
//     f32 at the PLY's own SH degree (lib/native/stream_encoder.h), into memory, then streamed from
//     there: the GPU decodes a full stream to the PLY's values bit for bit;
//   - anything else: a stream file, its first bytes read to check the header, then opened again
//     and read as the graph asks (zgal::fileStream).
// Either way it refuses a header lib/splat_stream.zgal's checkStreamHeader refuses (or a stream
// ending within its header) with ErrorCode::InvalidData, which fails the program before its first
// frame: the graph sizes its buffers from the header.
#include "splat_view.zgal.h"

namespace splat_native {
namespace gen = ::splat_view;
}

#include "../../lib/native/ply_reader.h"
#include "../../lib/native/stream_encoder.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// The stream's header from its first bytes, refused unless checkStreamHeader passes it
Result<splat_view::StreamHeader> checkedHeader(const std::vector<std::byte>& bytes, const std::string& name) {
    if (bytes.size() < splat_view::streamHeaderSize) {
        return Error(ErrorCode::InvalidData, "'" + name + "' ends within its " +
                                                 std::to_string(splat_view::streamHeaderSize) + "-byte stream header (" +
                                                 std::to_string(bytes.size()) + " bytes)");
    }
    splat_view::StreamHeader header;
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (const uint32_t code = splat_view::checkStreamHeader(header); code != splat_view::streamOk) {
        return Error(ErrorCode::InvalidData, "'" + name + "' has a stream header checkStreamHeader refuses, code " +
                                                 std::to_string(code) + " (lib/splat_stream.zgal)");
    }
    return header;
}

// Bytes in memory as a stream, which frees them once it has given them all: the graph keeps a
// stream's source for the program's life, and a PLY's encoding can be over 100 MB
zgal::Stream memoryStream(std::shared_ptr<const std::vector<std::byte>> bytes) {
    struct State {
        std::shared_ptr<const std::vector<std::byte>> bytes;
        size_t offset = 0;
    };
    auto state = std::make_shared<State>(State{std::move(bytes)});
    return zgal::Stream([state](std::span<std::byte> out) -> Result<size_t> {
        if (!state->bytes) return size_t(0);
        const size_t count = std::min(out.size(), state->bytes->size() - state->offset);
        std::memcpy(out.data(), state->bytes->data() + state->offset, count);
        state->offset += count;
        if (state->offset == state->bytes->size()) state->bytes.reset();
        return count;
    });
}

// A PLY, encoded in full into memory
Result<zgal::Stream> plyStream(const std::filesystem::path& path) {
    const std::string name = path.string();
    std::vector<splat_view::Splat> splats;
    uint32_t shDegree = 0;
    const splat_native::ReadStatus read = splat_native::readPly(name.c_str(), splats, shDegree);
    if (read.code != splat_native::kReadOk) {
        return Error(read.code == splat_native::kReadCannotRead ? ErrorCode::NotFound
                     : read.code == splat_native::kReadOutOfMemory ? ErrorCode::OutOfMemory
                                                                 : ErrorCode::InvalidData,
                     "reading the PLY '" + name + "': " + read.message);
    }
    if (splats.size() > splat_view::streamMaxSplats) {
        return Error(ErrorCode::InvalidData, "the PLY '" + name + "' holds " + std::to_string(splats.size()) +
                                                 " splats, above the stream's " +
                                                 std::to_string(splat_view::streamMaxSplats));
    }
    splat_native::EncodeConfig config{};
    config.shDegree = shDegree;
    for (auto& group : config.groups) {
        group = splat_native::GroupConfig{splat_view::encodingF32, splat_native::kRangeNone, {}, {}};
    }
    splat_view::StreamHeader layout;
    uint32_t badGroup = 0;
    if (const uint32_t status = splat_native::streamLayout(config, uint32_t(splats.size()), layout, badGroup);
        status != splat_native::kEncodeOk) {
        return Error(ErrorCode::InvalidData, "encoding the PLY '" + name + "': status " + std::to_string(status));
    }
    auto bytes = std::make_shared<std::vector<std::byte>>(splat_native::streamSize(layout));
    splat_native::EncodeReport report;
    if (const uint32_t status = splat_native::encodeStream(
            splats, config, std::span<uint8_t>(reinterpret_cast<uint8_t*>(bytes->data()), bytes->size()), report);
        status != splat_native::kEncodeOk) {
        return Error(ErrorCode::InvalidData, "encoding the PLY '" + name + "': status " + std::to_string(status) +
                                                 " (lib/native/stream_encoder.h), group " +
                                                 std::to_string(report.group) + ", splat " +
                                                 std::to_string(report.splat));
    }
    TRY(checkedHeader(*bytes, name));
    return memoryStream(bytes);
}

// The file's first bytes, up to a stream header's size: a PLY's magic, or the header to check
Result<std::vector<std::byte>> firstBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Error(ErrorCode::NotFound, "Can't open '" + path.string() + "'");
    }
    std::vector<std::byte> bytes(splat_view::streamHeaderSize);
    file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    bytes.resize(size_t(file.gcount()));
    return bytes;
}

} // namespace

Result<splat_view::MainInputs> splat_view::mainInputs(VulkanContext& context) {
    return MainInputs{
        .open = [&context](const std::filesystem::path& asset) -> Result<zgal::Stream> {
            TRY_ASSIGN(auto path, context.findAsset(asset));
            TRY_ASSIGN(const auto first, firstBytes(path));
            if (first.size() >= 3 && std::memcmp(first.data(), "ply", 3) == 0) return plyStream(path);
            // A stream file, its header checked, then read as the graph asks
            TRY(checkedHeader(first, path.string()));
            return zgal::fileStream(path);
        },
        .fail = [] {
            std::cout.flush();
            std::exit(1);
        },
        .saveAccum = [](const zgal::Data<glm::uvec2>& accum) {
            const char* path = std::getenv("SPLAT_VIEW_ACCUM");
            if (path == nullptr || *path == '\0') return;
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(accum.elements().data()),
                      std::streamsize(accum.size() * sizeof(glm::uvec2)));
            if (!out) {
                std::cout << "splat_view: FAILED, can't write accum to " << path << std::endl;
                std::exit(1);
            }
            std::cout << "splat_view: wrote accum to " << path << std::endl;
        },
    };
}
