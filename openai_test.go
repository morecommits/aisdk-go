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
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/stretchr/testify/require"
)

func TestPipeOpenAIToDataStream(t *testing.T) {
	t.Parallel()

	// openAIResponses are hardcoded responses from the OpenAI API Stream endpoint.
	openAIResponses := `data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_Qib2xfCV1qpazcvBOLg3wCNL","type":"function","function":{"name":"print","arguments":""}}],"refusal":null},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"message"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"hello"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" world"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-BIqITiyCGCjDeajdE8K0NWx18VRsO","object":"chat.completion.chunk","created":1743830633,"model":"gpt-4o-2024-08-06","service_tier":"default","system_fingerprint":"fp_898ac29719","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}

data: [DONE]`

	decoder := ssestream.NewDecoder(&http.Response{
		Body: io.NopCloser(strings.NewReader(openAIResponses)),
	})
	stream := ssestream.NewStream[openai.ChatCompletionChunk](decoder, nil)

	partsFormatted := []string{}
	oaiMessages, err := aisdk.PipeOpenAIToDataStream(stream, aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
		for _, part := range parts {
			output, err := part.Format()
			require.NoError(t, err)
			partsFormatted = append(partsFormatted, strings.TrimSpace(output))
		}
		return nil
	}), &aisdk.PipeOptions{
		HandleToolCall: func(toolCall aisdk.ToolCall) any {
			return map[string]string{
				"message": "Message printed to the console",
			}
		},
	})
	require.NoError(t, err)

	expected := `b:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","toolName":"print"}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":"{\""}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":"message"}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":"\":\""}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":"hello"}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":" world"}
c:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","argsTextDelta":"\"}"}
9:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","toolName":"print","args":{"message":"hello world"}}
a:{"toolCallId":"call_Qib2xfCV1qpazcvBOLg3wCNL","result":{"message":"Message printed to the console"}}
e:{"finishReason":"tool-calls","usage":{"promptTokens":null,"completionTokens":null},"isContinued":false}
d:{"finishReason":"tool-calls","usage":{"promptTokens":null,"completionTokens":null}}`
	require.Equal(t, expected, strings.Join(partsFormatted, "\n"))
	require.Len(t, oaiMessages.Messages, 2)

	first := oaiMessages.Messages[0]
	require.Equal(t, first.OfAssistant.ToolCalls[0].Function.Name, "print")
	second := oaiMessages.Messages[1]
	require.Equal(t, second.OfTool.Content.OfString.Value, `{"message":"Message printed to the console"}`)
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

	_, err = aisdk.PipeOpenAIToDataStream(stream, aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
		return nil
	}), nil)
	require.NoError(t, err)
}

// TestMessagesToOpenAI_Conversion tests the conversion logic without live API calls.
func TestMessagesToOpenAI_Conversion(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		inputMessages  []aisdk.Message
		expectedOutput []openai.ChatCompletionMessageParamUnion
		expectError    bool
	}{
		{
			name: "Simple User Message",
			inputMessages: []aisdk.Message{
				{Role: "user", Content: "Hello"},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
		},
		{
			name: "Simple System Message",
			inputMessages: []aisdk.Message{
				{Role: "system", Content: "You are helpful"},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are helpful"),
			},
		},
		{
			name: "Simple Assistant Message",
			inputMessages: []aisdk.Message{
				{Role: "assistant", Content: "Hi there"},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				openai.AssistantMessage("Hi there"),
			},
		},
		{
			name: "Assistant Message with Tool Call",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"SF"}`}},
						},
					},
				},
			},
		},
		{
			name: "Assistant Message with Tool Call (Partial)",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_abc", ToolName: "search", Args: map[string]any{"query": "aisdk"}, State: aisdk.ToolInvocationStatePartialCall},
					},
				},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_abc", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "search", Arguments: `{"query":"aisdk"}`}},
						},
					},
				},
			},
		},
		{
			name: "Assistant Message with Tool Result",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", State: aisdk.ToolInvocationStateResult, Result: map[string]any{"temp": 72}},
					},
				},
			},
			// Note: An assistant message *only* containing a result is a bit odd,
			// usually it follows a call. The converter creates an empty assistant message
			// and then a separate tool message.
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{OfAssistant: &openai.ChatCompletionAssistantMessageParam{ // Assistant message needs ToolCalls even for result state
					ToolCalls: []openai.ChatCompletionMessageToolCallParam{
						{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "", Arguments: "null"}}, // Need placeholder args/name if not provided
					},
				}},
				openai.ToolMessage(`{"temp":72}`, "call_123"),
			},
		},
		{
			name: "Assistant Message with Content and Tool Call",
			inputMessages: []aisdk.Message{
				{
					Role:    "assistant",
					Content: "Okay, getting weather.",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfArrayOfContentParts: []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
								{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Okay, getting weather."}},
							},
						},
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"SF"}`}},
						},
					},
				},
			},
		},
		{
			name: "Assistant Message with Content and Tool Result",
			inputMessages: []aisdk.Message{
				{
					Role:    "assistant",
					Content: "Thinking...", // Content might be present even if only results are in this specific message state
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", State: aisdk.ToolInvocationStateResult, Result: map[string]any{"temp": 72}},
					},
				},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{ // Assistant message part (with content, but no calls in this phase)
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfArrayOfContentParts: []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
								{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Thinking..."}},
							},
						},
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{ // Needs ToolCalls
							{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "", Arguments: "null"}},
						},
					},
				},
				openai.ToolMessage(`{"temp":72}`, "call_123"), // Separate Tool message for the result
			},
		},
		{
			name: "Sequence: User -> Assistant (Call) -> Assistant (Result)",
			inputMessages: []aisdk.Message{
				{Role: "user", Content: "What's the weather in SF?"},
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_weather", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
				{ // This represents the state *after* the tool result is available
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						// The original call info might technically still be here in some representations,
						// but MessagesToOpenAI only cares about the Result state for generating the ToolMessage
						{ToolCallID: "call_weather", State: aisdk.ToolInvocationStateResult, Result: map[string]any{"temp": 75, "unit": "F"}},
					},
				},
				{Role: "assistant", Content: "The weather in SF is 75F."},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("What's the weather in SF?"),
				{ // Assistant message for the call
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_weather", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"SF"}`}},
						},
					},
				},
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_weather", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "", Arguments: "null"}},
						},
					},
				},
				openai.ToolMessage(`{"temp":75,"unit":"F"}`, "call_weather"), // Tool message for the result
				openai.AssistantMessage("The weather in SF is 75F."),         // Final assistant content
			},
		},
		{
			name: "Assistant Message with Multiple Tool Calls and Results",
			inputMessages: []aisdk.Message{
				{
					Role:    "assistant",
					Content: "Okay, I need to do two things.",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_1", ToolName: "tool_a", Args: map[string]any{"p1": "v1"}, State: aisdk.ToolInvocationStateCall},
						{ToolCallID: "call_2", ToolName: "tool_b", Args: map[string]any{"p2": 123}, State: aisdk.ToolInvocationStateCall},
						{ToolCallID: "call_1", State: aisdk.ToolInvocationStateResult, Result: "Result A"},
						{ToolCallID: "call_2", State: aisdk.ToolInvocationStateResult, Result: map[string]any{"status": "ok"}},
					},
				},
			},
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{ // Assistant message with content and the two calls
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfArrayOfContentParts: []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
								{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Okay, I need to do two things."}},
							},
						},
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_1", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "tool_a", Arguments: `{"p1":"v1"}`}},
							{ID: "call_2", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "tool_b", Arguments: `{"p2":123}`}},
							// Need tool call info also for the results that follow
							{ID: "call_1", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "", Arguments: "null"}},
							{ID: "call_2", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "", Arguments: "null"}},
						},
					},
				},
				openai.ToolMessage(`"Result A"`, "call_1"),      // Result for call_1
				openai.ToolMessage(`{"status":"ok"}`, "call_2"), // Result for call_2
			},
		},
		{
			name: "Unsupported Role",
			inputMessages: []aisdk.Message{
				{Role: "invalid-role", Content: "This should fail"},
			},
			expectError: true,
		},
		{
			name: "Tool Call Arg Marshalling Error",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						// Functions cannot be marshalled to JSON
						{ToolCallID: "call_err", ToolName: "bad_args", Args: map[string]any{"fn": func() {}}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
			expectError: true,
		},
		{
			name: "Tool Result Marshalling Error",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						// Functions cannot be marshalled to JSON
						{ToolCallID: "call_err", State: aisdk.ToolInvocationStateResult, Result: func() {}},
					},
				},
			},
			expectError: true,
		},
		{
			name: "Assistant message with invalid tool args JSON",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						// Use invalid JSON (missing closing brace) to trigger unmarshal error
						{ToolCallID: "call_bad_args", ToolName: "bad_tool", Args: map[string]any{"location": "SF"}},
					},
				},
			},
			// This input map IS marshallable, so no error expected.
			expectError: false,
			expectedOutput: []openai.ChatCompletionMessageParamUnion{
				{ // Expect a valid assistant message with the marshalled tool call args
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_bad_args", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "bad_tool", Arguments: `{"location":"SF"}`}},
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		tc := tc // capture range variable
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			actualOutput, err := aisdk.MessagesToOpenAI(tc.inputMessages)

			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				// Using require.EqualValues for potentially looser comparison if direct struct comparison fails
				// due to subtle differences (e.g. nil maps vs empty maps if that were relevant)
				require.EqualValues(t, tc.expectedOutput, actualOutput)
			}
		})
	}
}

// TestOpenAIToMessages_Conversion tests the conversion from OpenAI format back to internal format.
func TestOpenAIToMessages_Conversion(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		inputMessages  []openai.ChatCompletionMessageParamUnion
		expectedOutput []aisdk.Message
		expectError    bool
	}{
		{
			name: "Simple User Message",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hello"),
			},
			expectedOutput: []aisdk.Message{
				{Role: "user", Content: "Hello"},
			},
		},
		{
			name: "Simple System Message",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("You are helpful"),
			},
			expectedOutput: []aisdk.Message{
				{Role: "system", Content: "You are helpful"},
			},
		},
		{
			name: "Simple Assistant Message",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				openai.AssistantMessage("Hi there"),
			},
			expectedOutput: []aisdk.Message{
				{Role: "assistant", Content: "Hi there"},
			},
		},
		{
			name: "Assistant Message with Tool Call",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"SF"}`}},
						},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
		},
		{
			name: "Assistant Message with Tool Call and Result",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				{ // Assistant initiates the call
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_123", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"SF"}`}},
						},
					},
				},
				openai.ToolMessage(`{"temp": 72}`, "call_123"), // Tool provides the result
			},
			expectedOutput: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						// Use map[string]interface{} for Args for assertion consistency
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]interface{}{"location": "SF"}, State: aisdk.ToolInvocationStateResult, Result: `{"temp": 72}`},
					},
				},
			},
		},
		{
			name: "Assistant Message with Content and Tool Call/Result",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				{ // Assistant initiates the call with some text
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						Content: openai.ChatCompletionAssistantMessageParamContentUnion{
							OfArrayOfContentParts: []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
								{OfText: &openai.ChatCompletionContentPartTextParam{Text: "Checking the weather..."}},
							},
						},
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_weather_check", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "get_weather", Arguments: `{"location":"London"}`}},
						},
					},
				},
				openai.ToolMessage(`{"temp": 15, "unit": "C"}`, "call_weather_check"), // Tool provides the result
			},
			expectedOutput: []aisdk.Message{
				{
					Role:    "assistant",
					Content: "Checking the weather...",
					ToolInvocations: []aisdk.ToolInvocation{
						// Use map[string]interface{} for Args for assertion consistency
						{ToolCallID: "call_weather_check", ToolName: "get_weather", Args: map[string]interface{}{"location": "London"}, State: aisdk.ToolInvocationStateResult, Result: `{"temp": 15, "unit": "C"}`},
					},
				},
			},
		},
		{
			name: "Tool Message without preceding Assistant Message",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				openai.ToolMessage(`{"result":"something"}`, "call_orphan"),
			},
			expectError: true,
		},
		{
			name: "Tool Message with incorrect ToolCallID",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				{ // Assistant initiates a call
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							{ID: "call_correct", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "tool", Arguments: `{}`}},
						},
					},
				},
				openai.ToolMessage(`{"result":"something"}`, "call_incorrect"), // Tool provides result for wrong ID
			},
			expectError: true,
		},
		{
			name: "Assistant message with invalid tool args JSON",
			inputMessages: []openai.ChatCompletionMessageParamUnion{
				{
					OfAssistant: &openai.ChatCompletionAssistantMessageParam{
						ToolCalls: []openai.ChatCompletionMessageToolCallParam{
							// Use invalid JSON (missing closing brace) to trigger unmarshal error
							{ID: "call_bad_args", Function: openai.ChatCompletionMessageToolCallFunctionParam{Name: "bad_tool", Arguments: `INVALID_JSON`}},
						},
					},
				},
			},
			// Function handles unmarshal error gracefully, so no error expected.
			expectError: false,
			expectedOutput: []aisdk.Message{
				{ // Expect args to be nil map due to unmarshal error
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_bad_args", ToolName: "bad_tool", Args: map[string]interface{}(nil), State: aisdk.ToolInvocationStateCall},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		tc := tc // Capture range variable
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			output, err := aisdk.OpenAIToMessages(tc.inputMessages)
			if tc.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)

				// Granular assertions instead of comparing the whole structure
				require.Len(t, output, len(tc.expectedOutput), "Number of messages should match")

				if len(tc.expectedOutput) > 0 && len(output) > 0 {
					for i := range tc.expectedOutput {
						expectedMsg := tc.expectedOutput[i]
						actualMsg := output[i]

						require.Equal(t, expectedMsg.Role, actualMsg.Role, "Message Role mismatch at index %d", i)
						require.Equal(t, expectedMsg.Content, actualMsg.Content, "Message Content mismatch at index %d", i)
						require.Len(t, actualMsg.ToolInvocations, len(expectedMsg.ToolInvocations), "Number of ToolInvocations mismatch at index %d", i)

						if len(expectedMsg.ToolInvocations) > 0 && len(actualMsg.ToolInvocations) > 0 {
							for j := range expectedMsg.ToolInvocations {
								expectedTI := expectedMsg.ToolInvocations[j]
								actualTI := actualMsg.ToolInvocations[j]
								require.Equal(t, expectedTI.ToolCallID, actualTI.ToolCallID, "ToolCallID mismatch at msg %d, ti %d", i, j)
								require.Equal(t, expectedTI.ToolName, actualTI.ToolName, "ToolName mismatch at msg %d, ti %d", i, j)
								require.Equal(t, expectedTI.State, actualTI.State, "State mismatch at msg %d, ti %d", i, j)
								// Compare Args using EqualValues as it involves maps
								require.EqualValues(t, expectedTI.Args, actualTI.Args, "Args mismatch at msg %d, ti %d", i, j)
								// Compare Result using EqualValues as it can be different types (string, map, etc.)
								require.EqualValues(t, expectedTI.Result, actualTI.Result, "Result mismatch at msg %d, ti %d", i, j)
							}
						}
					}
				}
			}
		})
	}
}
