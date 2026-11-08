// The oracle's chat render: the prompt llama-server builds from one
// /v1/chat/completions body, at the pin, with no model loaded.
//
//   chat_render <out> <bos> <eos> [llama-server flags] < body.json
//
// writes the prompt's bytes to <out>. llama.cpp logs to stdout, so the
// prompt cannot go there.
//
// The flags go through llama.cpp's own parser as llama-server's
// (common_params_parse), so a default such as preserve_reasoning's
// (common/arg.cpp, after parsing) is upstream's code, not a copy of it.
// Pass the template with --chat-template-file. The body is handled as
// oaicompat_chat_params_parse handles it (tools/server/server-common.cpp),
// with enable_thinking resolved as server-context.cpp resolves it at
// load. Steps that cannot change the prompt (media, grammar, sampling)
// are left out.
//
// strftime_now reads the real clock in local time (jinja::context's
// current_time); inputs.now does not reach it. Run under TZ=UTC.
#include "arg.h"
#include "chat.h"
#include "common.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <out> <bos> <eos> [llama-server flags] < body.json\n", argv[0]);
        return 2;
    }
    const std::string out_path = argv[1];
    const std::string bos      = argv[2];
    const std::string eos      = argv[3];

    std::vector<char *> args = { argv[0] };
    for (int i = 4; i < argc; i++) {
        args.push_back(argv[i]);
    }
    common_params params;
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 2;
    }
    if (params.chat_template.empty()) {
        fprintf(stderr, "chat_render: pass the template with --chat-template-file\n");
        return 2;
    }

    try {
        const std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
        common_json body = common_json::parse(text);

        auto tmpls = common_chat_templates_init(/* model= */ nullptr, params.chat_template, bos, eos);

        // server-context.cpp, at load
        const bool template_supports_thinking =
            params.use_jinja && common_chat_templates_support_enable_thinking(tmpls.get());
        const bool enable_thinking = params.enable_reasoning != 0 && template_supports_thinking;

        // server-common.cpp, oaicompat_chat_params_parse
        const common_json tools       = body.value("tools", common_json());
        const std::string tool_choice = body.value("tool_choice", std::string("auto"));
        const auto        caps        = common_chat_templates_get_caps(tmpls.get());

        common_chat_templates_inputs inputs;
        inputs.messages               = common_chat_msgs_parse_oaicompat(body.at("messages"));
        inputs.tools                  = common_chat_tools_parse_oaicompat(tools);
        inputs.tool_choice            = common_chat_tool_choice_parse_oaicompat(tool_choice);
        inputs.use_jinja              = params.use_jinja;
        inputs.parallel_tool_calls    = body.value("parallel_tool_calls", caps.at("supports_parallel_tool_calls").get<bool>());
        inputs.add_generation_prompt  = body.value("add_generation_prompt", true);
        inputs.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
        if (params.prefill_assistant && !inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
            inputs.add_generation_prompt  = false;
        }
        inputs.reasoning_format = params.reasoning_format;
        if (body.contains("reasoning_format")) {
            inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
        }
        inputs.enable_thinking = enable_thinking;

        inputs.chat_template_kwargs = params.default_template_kwargs;
        if (body.contains("chat_template_kwargs")) {
            for (const auto & [key, value] : body.at("chat_template_kwargs").items()) {
                inputs.chat_template_kwargs[key] = value.dump();
            }
        }
        const auto enable_thinking_kwarg = inputs.chat_template_kwargs.find("enable_thinking");
        if (enable_thinking_kwarg != inputs.chat_template_kwargs.end()) {
            if (enable_thinking_kwarg->second == "true") {
                inputs.enable_thinking = true;
            } else if (enable_thinking_kwarg->second == "false") {
                inputs.enable_thinking = false;
            } else if (!enable_thinking_kwarg->second.empty() && enable_thinking_kwarg->second[0] == '"') {
                throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
            }
        }
        if (body.contains("reasoning_effort")) {
            const std::string effort = body.value("reasoning_effort", std::string(""));
            if (effort == "none") {
                inputs.enable_thinking = false;
                inputs.chat_template_kwargs.erase("reasoning_effort");
            } else if (!effort.empty()) {
                inputs.chat_template_kwargs["reasoning_effort"] = common_json::make(effort).dump();
            }
        }
        inputs.force_pure_content = params.force_pure_content_parser;

        const common_chat_session session(tmpls.get(), /* vocab= */ nullptr, inputs, common_chat_session_params());
        const std::string         prompt = session.prompt().str();

        FILE * out = fopen(out_path.c_str(), "wb");
        if (out == nullptr || fwrite(prompt.data(), 1, prompt.size(), out) != prompt.size() || fclose(out) != 0) {
            fprintf(stderr, "chat_render: cannot write %s\n", out_path.c_str());
            return 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "chat_render: %s\n", e.what());
        return 1;
    }
    return 0;
}
