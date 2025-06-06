package aisdk_test

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/kylecarbs/aisdk-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"

	// Import the param package for NewOpt

	"github.com/openai/openai-go/packages/ssestream"
	"github.com/stretchr/testify/require"
)

func TestOpenAIToDataStream(t *testing.T) {
	t.Parallel()

	// Hardcoded example response from OpenAI API streaming endpoint
	// (tool_calls chunk followed by message chunk)
	mockResponse := `data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_acK2pxwOef03RhfTFTbuPTkR","type":"function","function":{"name":"test","arguments":""}}],"refusal":null},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"message"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"This"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" is"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" a"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" test"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" run"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" as"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" requested"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":".\""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BK4NPErLSC7PWDhqhhLSFQAFkGJvU","object":"chat.completion.chunk","created":1744123083,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}

data: [DONE]`

	// 1. Create decoder using ssestream.NewDecoder with a mock http.Response
	mockHTTPResp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(mockResponse)),
	}
	decoder := ssestream.NewDecoder(mockHTTPResp)

	// 2. Create the typed stream required by OpenAIToDataStream
	typedStream := ssestream.NewStream[openai.ChatCompletionChunk](decoder, nil)

	// 3. Pass the typed stream to OpenAIToDataStream and accumulate results
	var acc aisdk.DataStreamAccumulator
	stream := aisdk.OpenAIToDataStream(typedStream)
	stream = stream.WithToolCalling(func(toolCall aisdk.ToolCall) any {
		return map[string]any{"message": "Message printed to the console"}
	})
	stream = stream.WithAccumulator(&acc) // Accumulator is attached here

	// Iterate through the stream to drive the accumulator
	for _, err := range stream {
		require.NoError(t, err)
	}

	// 4. Define expected accumulator state based on the mockResponse
	// Based on the stream, we expect one assistant message containing both
	// the tool call and the tool result parts.
	expectedMessages := []aisdk.Message{
		{
			// ID might be derived from the stream, let accumulator handle it or check if needed
			Role: "assistant",
			// Content might be empty or contain deltas if any text parts were present
			Content: "", // No text parts in this mock response
			Parts: []aisdk.Part{
				{
					Type: aisdk.PartTypeToolInvocation,
					ToolInvocation: &aisdk.ToolInvocation{
						State:      aisdk.ToolInvocationStateResult,
						ToolCallID: "call_acK2pxwOef03RhfTFTbuPTkR",
						ToolName:   "test",
						Args:       map[string]any{"message": "This is a test run as requested."},
						Result:     map[string]any{"message": "Message printed to the console"},
					},
				},
			},
		},
	}
	expectedFinishReason := aisdk.FinishReasonToolCalls                    // From the last meaningful finish reason in the stream
	expectedUsage := aisdk.Usage{PromptTokens: nil, CompletionTokens: nil} // Mock data has no usage info

	// 5. Assert accumulator state
	// Use EqualExportedValues to ignore internal fields like 'isComplete' in Part
	require.EqualExportedValues(t, expectedMessages, acc.Messages())
	require.Equal(t, expectedFinishReason, acc.FinishReason())
	require.Equal(t, expectedUsage, acc.Usage())

	toOpenAI, err := aisdk.MessagesToOpenAI(expectedMessages)
	require.NoError(t, err)

	// Verify the converted OpenAI messages match our expectations
	// We expect two messages: an assistant message with the tool call,
	// and a tool message with the result.
	require.Len(t, toOpenAI, 2)

	// Check Assistant Message (Tool Call)
	assistantMsg := toOpenAI[0].OfAssistant
	require.NotNil(t, assistantMsg)
	require.Len(t, assistantMsg.ToolCalls, 1)
	require.Equal(t, "call_acK2pxwOef03RhfTFTbuPTkR", assistantMsg.ToolCalls[0].ID)
	require.Equal(t, "test", assistantMsg.ToolCalls[0].Function.Name)
	require.Equal(t, `{"message":"This is a test run as requested."}`, assistantMsg.ToolCalls[0].Function.Arguments)

	// Check Tool Message (Tool Result)
	toolMsg := toOpenAI[1].OfTool
	require.NotNil(t, toolMsg)
	require.Equal(t, "call_acK2pxwOef03RhfTFTbuPTkR", toolMsg.ToolCallID)
	require.Equal(t, `{"message":"Message printed to the console"}`, toolMsg.Content.OfArrayOfContentParts[0].Text)
}

func TestMessagesToOpenAI_Live(t *testing.T) {
	t.Parallel()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	ctx := context.Background()
	client := openai.NewClient(option.WithAPIKey(apiKey))

	// Ensure messages are converted correctly.
	messages, err := aisdk.MessagesToOpenAI([]aisdk.Message{
		{
			Role:    "system",
			Content: "use the 'print' tool to print 'Hello, world!' and then show the result",
		},
	})
	require.NoError(t, err)

	stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model:    openai.ChatModelGPT4o,
		Messages: messages,
	})
	require.NoError(t, err)

	dataStream := aisdk.OpenAIToDataStream(stream)
	for _, err := range dataStream {
		if err != nil {
			t.Fatal(err)
		}
	}
}
