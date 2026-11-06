import { describe, expect, it } from "bun:test";
import { convertCodexResponsesMessages } from "@oh-my-pi/pi-ai/providers/openai-codex-responses";
import type { ResponseInput } from "@oh-my-pi/pi-ai/providers/openai-responses-wire";
import { buildResponsesInput } from "@oh-my-pi/pi-ai/providers/openai-shared";
import type { Context, Model } from "@oh-my-pi/pi-ai/types";
import { buildModel } from "@oh-my-pi/pi-catalog/build";

const openRouterGemini = buildModel({
	id: "google/gemini-4-argon",
	name: "Gemini 4 Argon",
	api: "openai-responses",
	provider: "openrouter",
	baseUrl: "https://openrouter.ai/api/v1",
	reasoning: false,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 16_000,
});

const openRouterOpenAI = buildModel({
	id: "openai/gpt-5.5",
	name: "GPT-5.5",
	api: "openai-responses",
	provider: "openrouter",
	baseUrl: "https://openrouter.ai/api/v1",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 16_000,
});

const codexModel = buildModel({
	id: "gpt-5.5",
	name: "Codex Test",
	api: "openai-codex-responses",
	provider: "openai-codex",
	baseUrl: "https://chatgpt.com/backend-api/codex/responses",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 200_000,
	maxTokens: 100_000,
	compat: { supportsImageDetailOriginal: true },
});

const zeroUsage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function makeContext(model: Model): Context {
	return {
		messages: [
			{
				role: "assistant",
				content: [
					{ type: "toolCall", id: "call_read_36", name: "read", arguments: { path: "a.png" } },
					{ type: "toolCall", id: "call_read_37", name: "read", arguments: { path: "b.png" } },
					{ type: "toolCall", id: "call_bash_38", name: "bash", arguments: { command: "true" } },
				],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: zeroUsage,
				stopReason: "toolUse",
				timestamp: 1,
			},
			{
				role: "toolResult",
				toolCallId: "call_read_36",
				toolName: "read",
				content: [
					{ type: "text", text: "first" },
					{ type: "image", mimeType: "image/png", data: "AAAA" },
				],
				isError: false,
				timestamp: 2,
			},
			{
				role: "toolResult",
				toolCallId: "call_read_37",
				toolName: "read",
				content: [
					{ type: "text", text: "second" },
					{ type: "image", mimeType: "image/png", data: "BBBB" },
				],
				isError: false,
				timestamp: 3,
			},
			{
				role: "toolResult",
				toolCallId: "call_bash_38",
				toolName: "bash",
				content: [{ type: "text", text: "done" }],
				isError: false,
				timestamp: 4,
			},
			{ role: "user", content: "continue", timestamp: 5 },
		],
	};
}

function expectOrderedToolResults(items: ResponseInput): void {
	expect(items.slice(0, 3).map(item => ("call_id" in item ? item.call_id : undefined))).toEqual([
		"call_read_36",
		"call_read_37",
		"call_bash_38",
	]);
	expect(items.slice(3)).toEqual([
		{
			type: "function_call_output",
			call_id: "call_read_36",
			output: [
				{ type: "input_text", text: "first" },
				{ type: "input_image", detail: "auto", image_url: "data:image/png;base64,AAAA" },
			],
		},
		{
			type: "function_call_output",
			call_id: "call_read_37",
			output: [
				{ type: "input_text", text: "second" },
				{ type: "input_image", detail: "auto", image_url: "data:image/png;base64,BBBB" },
			],
		},
		{ type: "function_call_output", call_id: "call_bash_38", output: "done" },
		{ role: "user", content: [{ type: "input_text", text: "continue" }] },
	]);
}

describe("parallel Responses tool-result images", () => {
	it("keeps OpenRouter non-OpenAI images out of function_call_output and after the tool batch", () => {
		// OpenRouter drops an array `output` for these upstreams, so the translated
		// request ends on the function call and the upstream returns 400 (#13486).
		const items = buildResponsesInput({
			model: openRouterGemini,
			context: makeContext(openRouterGemini),
			strictResponsesPairing: true,
			supportsImageDetailOriginal: false,
		});

		expect(items.slice(0, 3).map(item => ("call_id" in item ? item.call_id : undefined))).toEqual([
			"call_read_36",
			"call_read_37",
			"call_bash_38",
		]);
		expect(items.slice(3)).toEqual([
			{ type: "function_call_output", call_id: "call_read_36", output: "first" },
			{ type: "function_call_output", call_id: "call_read_37", output: "second" },
			{ type: "function_call_output", call_id: "call_bash_38", output: "done" },
			{
				type: "message",
				role: "user",
				content: [
					{ type: "input_text", text: "Attached image(s) from tool result:" },
					{ type: "input_image", detail: "auto", image_url: "data:image/png;base64,AAAA" },
				],
			},
			{
				type: "message",
				role: "user",
				content: [
					{ type: "input_text", text: "Attached image(s) from tool result:" },
					{ type: "input_image", detail: "auto", image_url: "data:image/png;base64,BBBB" },
				],
			},
			{ role: "user", content: [{ type: "input_text", text: "continue" }] },
		]);
	});

	it("keeps a later OpenRouter tool output from jumping ahead of an earlier image", () => {
		const context = makeContext(openRouterGemini);
		context.messages.push(
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call_read_39", name: "read", arguments: { path: "c.png" } }],
				api: openRouterGemini.api,
				provider: openRouterGemini.provider,
				model: openRouterGemini.id,
				usage: zeroUsage,
				stopReason: "toolUse",
				timestamp: 6,
			},
			{
				role: "toolResult",
				toolCallId: "call_read_39",
				toolName: "read",
				content: [
					{ type: "text", text: "third" },
					{ type: "image", mimeType: "image/png", data: "CCCC" },
				],
				isError: false,
				timestamp: 7,
			},
		);
		const items = buildResponsesInput({
			model: openRouterGemini,
			context,
			strictResponsesPairing: true,
			supportsImageDetailOriginal: false,
		});
		const types = items.map(item => ("type" in item && item.type ? item.type : "role" in item ? item.role : "item"));
		expect(types).toEqual([
			"function_call",
			"function_call",
			"function_call",
			"function_call_output",
			"function_call_output",
			"function_call_output",
			"message",
			"message",
			"user",
			"function_call",
			"function_call_output",
			"message",
		]);
		const thirdOutput = items.find(
			item => item.type === "function_call_output" && "call_id" in item && item.call_id === "call_read_39",
		);
		expect(thirdOutput).toMatchObject({ output: "third" });
	});

	it("sends an image-only OpenRouter tool result as a string plus the image message", () => {
		const items = buildResponsesInput({
			model: openRouterGemini,
			context: {
				messages: [
					{
						role: "assistant",
						content: [{ type: "toolCall", id: "call_read_1", name: "read", arguments: { path: "a.png" } }],
						api: openRouterGemini.api,
						provider: openRouterGemini.provider,
						model: openRouterGemini.id,
						usage: zeroUsage,
						stopReason: "toolUse",
						timestamp: 1,
					},
					{
						role: "toolResult",
						toolCallId: "call_read_1",
						toolName: "read",
						content: [{ type: "image", mimeType: "image/webp", data: "AAAA" }],
						isError: false,
						timestamp: 2,
					},
				],
			},
			strictResponsesPairing: true,
			supportsImageDetailOriginal: false,
		});
		expect(items.filter(item => item.type === "function_call_output")).toEqual([
			{ type: "function_call_output", call_id: "call_read_1", output: "(see attached image)" },
		]);
		expect(items.at(-1)).toEqual({
			type: "message",
			role: "user",
			content: [
				{ type: "input_text", text: "Attached image(s) from tool result:" },
				{ type: "input_image", detail: "auto", image_url: "data:image/webp;base64,AAAA" },
			],
		});
	});

	it("encodes OpenRouter OpenAI-upstream images inside their tool outputs", () => {
		const items = buildResponsesInput({
			model: openRouterOpenAI,
			context: makeContext(openRouterOpenAI),
			strictResponsesPairing: true,
			supportsImageDetailOriginal: true,
		});

		expectOrderedToolResults(items);
	});

	it("encodes Codex Responses images inside their tool outputs", () => {
		const items = convertCodexResponsesMessages(codexModel, makeContext(codexModel));

		expectOrderedToolResults(items);
	});
});
