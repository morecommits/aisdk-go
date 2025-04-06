package aisdk_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/kylecarbs/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestPipeAnthropicToDataStream(t *testing.T) {
	t.Parallel()

	// anthropicResponses are hardcoded responses from the Anthropic API Stream endpoint.
	anthropicResponses := `event: message_start
data: {"type":"message_start","message":{"id":"msg_01LHXQM4FBxykQGT7N1a7kJ7","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":408,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":1}}        }

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}      }

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I"}    }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'ll help you print 'hello world' to the console"}              }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" using the print function."}      }

event: content_block_stop
data: {"type":"content_block_stop","index":0  }

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01RA76iwg1LbKuDjJnc6ym45","name":"print","input":{}}            }

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}    }

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"message\""}    }

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":": \"hel"}   }

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"lo worl"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"d\"}"}       }

event: content_block_stop
data: {"type":"content_block_stop","index":1         }

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":71}             }

event: message_stop
data: {"type":"message_stop" }`

	decoder := ssestream.NewDecoder(&http.Response{
		Body: io.NopCloser(strings.NewReader(anthropicResponses)),
	})
	stream := ssestream.NewStream[anthropic.MessageStreamEventUnion](decoder, nil)

	partsFormatted := []string{}
	_, err := aisdk.PipeAnthropicToDataStream(stream, aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
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

	expected := `f:{"messageId":"msg_01LHXQM4FBxykQGT7N1a7kJ7"}
0:"I"
0:"'ll help you print 'hello world' to the console"
0:" using the print function."
b:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","toolName":"print"}
c:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","argsTextDelta":""}
c:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","argsTextDelta":"{\"message\""}
c:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","argsTextDelta":": \"hel"}
c:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","argsTextDelta":"lo worl"}
c:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","argsTextDelta":"d\"}"}
9:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","toolName":"print","args":{"message":"hello world"}}
a:{"toolCallId":"toolu_01RA76iwg1LbKuDjJnc6ym45","result":{"message":"Message printed to the console"}}
e:{"finishReason":"tool-calls","usage":{"promptTokens":408,"completionTokens":71},"isContinued":false}
d:{"finishReason":"tool-calls","usage":{"promptTokens":408,"completionTokens":71}}`

	require.EqualValues(t, expected, strings.Join(partsFormatted, "\n"))
}

func TestMessagesToAnthropic_Live(t *testing.T) {
	t.Parallel()
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY is not set")
	}
	ctx := context.Background()
	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	// Ensure messages are converted correctly.
	messages, systemPrompts, err := aisdk.MessagesToAnthropic([]aisdk.Message{
		{
			Role:    "system",
			Content: "use the 'print' tool to print 'Hello, world!' and then show the result",
		},
		{
			Role: "user", Content: "Go ahead.",
		},
	})
	require.NoError(t, err)

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Messages:  messages,
		Model:     anthropic.ModelClaude3_5SonnetLatest,
		System:    systemPrompts,
		MaxTokens: 10,
	})
	require.NoError(t, err)

	_, err = aisdk.PipeAnthropicToDataStream(stream, aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
		return nil
	}), nil)
	require.NoError(t, err)
}

func TestMessagesToAnthropic_Conversion(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                  string
		inputMessages         []aisdk.Message
		expectedOutput        []anthropic.MessageParam
		expectedSystemPrompts []anthropic.TextBlockParam
		expectError           bool
	}{
		{
			name: "Simple User Message",
			inputMessages: []aisdk.Message{
				{Role: "user", Content: "Hello"},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Hello"}},
					},
				},
			},
			expectedSystemPrompts: nil,
		},
		{
			name: "Simple Assistant Message",
			inputMessages: []aisdk.Message{
				{Role: "assistant", Content: "Hi there"},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Hi there"}},
					},
				},
			},
			expectedSystemPrompts: nil,
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
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location":"SF"}`),
							},
						},
					},
				},
			},
			expectedSystemPrompts: nil,
		},
		{
			name: "Assistant Message with Tool Call (Partial State)",
			inputMessages: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_abc", ToolName: "search", Args: map[string]any{"query": "aisdk"}, State: aisdk.ToolInvocationStatePartialCall},
					},
				},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_abc",
								Name:  "search",
								Input: json.RawMessage(`{"query":"aisdk"}`),
							},
						},
					},
				},
			},
			expectedSystemPrompts: nil,
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
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    "call_123",
							Name:  "",
							Input: json.RawMessage(`null`),
						}},
					},
				},
				{
					Role: anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_123", `{"temp":72}`, false),
					},
				},
			},
			expectedSystemPrompts: nil,
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
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Okay, getting weather."}},
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location":"SF"}`),
							},
						},
					},
				},
			},
			expectedSystemPrompts: nil,
		},
		{
			name: "Assistant Message with Content and Tool Result",
			inputMessages: []aisdk.Message{
				{
					Role:    "assistant",
					Content: "Here is the result:",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", State: aisdk.ToolInvocationStateResult, Result: map[string]any{"temp": 72}},
					},
				},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Here is the result:"}},
						{OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    "call_123",
							Name:  "",
							Input: json.RawMessage(`null`),
						}},
					},
				},
				{
					Role: anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_123", `{"temp":72}`, false),
					},
				},
			},
			expectedSystemPrompts: nil,
		},
		{
			name: "Sequence: User -> Assistant (Call) -> Assistant (Result)",
			inputMessages: []aisdk.Message{
				{Role: "user", Content: "What's the weather in SF?"},
				{ // Turn 2: Assistant initiates call
					Role:    "assistant",
					Content: "", // Optional text content
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_weather", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
				{ // Turn 3: Assistant provides result
					Role:    "assistant",
					Content: "", // Optional text content
					ToolInvocations: []aisdk.ToolInvocation{
						// Note: Need ToolName/Args here for MessagesToAnthropic to generate the required preceding ToolUse block
						{ToolCallID: "call_weather", ToolName: "get_weather", Args: map[string]any{"location": "SF"}, State: aisdk.ToolInvocationStateResult, Result: map[string]any{"temp": 75, "unit": "F"}},
					},
				},
				{ // Turn 4: Assistant final response
					Role:    "assistant",
					Content: "The weather in SF is 75F.",
				},
			},
			expectedOutput: []anthropic.MessageParam{
				{ // 1. User message
					Role: anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "What's the weather in SF?"}},
					},
				},
				{ // 2. Assistant message for the initial call
					Role: anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    "call_weather",
							Name:  "get_weather",
							Input: json.RawMessage(`{"location":"SF"}`),
						}},
					},
				},
				{ // 3. Assistant message corresponding to the result input state
					Role: anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    "call_weather",
							Name:  "get_weather",
							Input: json.RawMessage(`{"location":"SF"}`),
						}},
					},
				},
				{ // 4. User message with the actual ToolResult
					Role: anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_weather", `{"temp":75,"unit":"F"}`, false),
					},
				},
				{ // 5. Final assistant text response
					Role: anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "The weather in SF is 75F."}},
					},
				},
			},
			expectedSystemPrompts: nil,
			expectError:           false,
		},
		{
			name: "System Message Extraction",
			inputMessages: []aisdk.Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "Hello"},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Hello"}},
					},
				},
			},
			expectedSystemPrompts: []anthropic.TextBlockParam{{Text: "You are helpful."}},
			expectError:           false,
		},
		{
			name: "Multiple System Messages Extraction",
			inputMessages: []aisdk.Message{
				{Role: "system", Content: "Prompt 1"},
				{Role: "user", Content: "User Msg"},
				{Role: "system", Content: "Prompt 2"},
			},
			expectedOutput: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "User Msg"}},
					},
				},
			},
			expectedSystemPrompts: []anthropic.TextBlockParam{
				{Text: "Prompt 1"},
				{Text: "Prompt 2"},
			},
			expectError: false,
		},
		{
			name: "Unsupported Role Error",
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
						{ToolCallID: "call_err", ToolName: "bad_args", Args: func() {}, State: aisdk.ToolInvocationStateCall},
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
						{ToolCallID: "call_err", State: aisdk.ToolInvocationStateResult, Result: func() {}},
					},
				},
			},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		tc := tc // capture range variable
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			actualOutput, actualSystemPrompts, err := aisdk.MessagesToAnthropic(tc.inputMessages)

			if tc.expectError {
				require.Error(t, err)
				require.Nil(t, actualOutput)
				require.Nil(t, actualSystemPrompts)
			} else {
				require.NoError(t, err)
				require.EqualValues(t, tc.expectedOutput, actualOutput)
				if tc.expectedSystemPrompts == nil {
					require.True(t, actualSystemPrompts == nil || len(actualSystemPrompts) == 0, "Expected nil or empty system prompts, got %v", actualSystemPrompts)
				} else {
					require.EqualValues(t, tc.expectedSystemPrompts, actualSystemPrompts) // Use EqualValues for struct comparison
				}
			}
		})
	}
}

// TestAnthropicToMessages_Conversion tests the conversion from Anthropic format back to internal format.
func TestAnthropicToMessages_Conversion(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                string
		inputMessages       []anthropic.MessageParam
		expectedOutput      []aisdk.Message
		expectError         bool
		expectedErrorSubstr string // Optional substring match for expected errors
	}{
		{
			name: "Simple User Message",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Hello"}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "user", Content: "Hello", ToolInvocations: []aisdk.ToolInvocation{}},
			},
		},
		{
			name: "Simple Assistant Message",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Hi there"}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "assistant", Content: "Hi there", ToolInvocations: []aisdk.ToolInvocation{}},
			},
		},
		{
			name: "Assistant Message with Tool Call",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location":"SF"}`),
							},
						},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]interface{}{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
		},
		{
			name: "Assistant Call followed by User Result",
			inputMessages: []anthropic.MessageParam{
				{ // Assistant makes the call
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_123",
								Name:  "get_weather",
								Input: json.RawMessage(`{"location":"SF"}`),
							},
						},
					},
				},
				{ // User provides the result
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_123", `{"temp": 72}`, false),
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{ // Only the assistant message remains, modified with the result
					Role: "assistant",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_123", ToolName: "get_weather", Args: map[string]interface{}{"location": "SF"}, State: aisdk.ToolInvocationStateResult, Result: `{"temp": 72}`},
					},
				},
			},
		},
		{
			name: "Assistant Call + Text, followed by User Result + Text",
			inputMessages: []anthropic.MessageParam{
				{ // Assistant makes the call and says something
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Okay, calling tool..."}},
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_abc",
								Name:  "do_stuff",
								Input: json.RawMessage(`{}`),
							},
						},
					},
				},
				{ // User provides the result and says something else
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_abc", `{"status":"done"}`, false),
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "Tool finished."}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{ // Assistant message, modified with result
					Role:    "assistant",
					Content: "Okay, calling tool...",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "call_abc", ToolName: "do_stuff", Args: map[string]interface{}{}, State: aisdk.ToolInvocationStateResult, Result: `{"status":"done"}`},
					},
				},
				{ // User message with only the text part remaining
					Role:    "user",
					Content: "Tool finished.",
				},
			},
		},
		{
			name: "With System Prompt",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestTextBlock: &anthropic.TextBlockParam{Text: "User turn"}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "user", Content: "User turn"},
			},
		},
		{
			name: "Tool Result without preceding Assistant Call",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_orphan", `{"data":"abc"}`, false),
					},
				},
			},
			expectError:         true,
			expectedErrorSubstr: "without a preceding assistant message",
		},
		{
			name: "Tool Result with non-matching ToolCallID",
			inputMessages: []anthropic.MessageParam{
				{ // Assistant makes a call
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{ID: "call_correct", Name: "tool", Input: json.RawMessage(`{}`)}},
					},
				},
				{ // User provides result for a different ID
					Role: "user",
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewToolResultBlock("call_incorrect", `{"data":"abc"}`, false),
					},
				},
			},
			expectError:         true,
			expectedErrorSubstr: "unknown tool use ID",
		},
		{
			name: "Assistant message with invalid tool args JSON",
			inputMessages: []anthropic.MessageParam{
				{
					Role: "assistant",
					Content: []anthropic.ContentBlockParamUnion{
						{
							OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
								ID:    "call_bad_args",
								Name:  "bad_tool",
								Input: json.RawMessage(`{"location":"SF`), // Invalid JSON
							},
						},
					},
				},
			},
			// Function handles unmarshal error gracefully, expects nil args
			expectedOutput: []aisdk.Message{
				{
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
			output, err := aisdk.AnthropicToMessages(tc.inputMessages)
			if tc.expectError {
				require.Error(t, err)
				if tc.expectedErrorSubstr != "" {
					require.Contains(t, err.Error(), tc.expectedErrorSubstr)
				}
			} else {
				require.NoError(t, err)

				// Use simple EqualValues for this specific case to get a clearer diff if it fails
				if tc.name == "Assistant Message with Tool Call" {
					require.EqualValues(t, tc.expectedOutput, output)
				} else {
					// Granular assertions for other cases
					require.Len(t, output, len(tc.expectedOutput), "Number of messages should match")

					if len(tc.expectedOutput) > 0 && len(output) > 0 {
						for i := range tc.expectedOutput {
							if i >= len(output) {
								t.Fatalf("Index %d out of bounds for actual output (len %d)", i, len(output))
							}
							expectedMsg := tc.expectedOutput[i]
							actualMsg := output[i]

							require.Equal(t, expectedMsg.Role, actualMsg.Role, "Message Role mismatch at index %d", i)
							require.Equal(t, expectedMsg.Content, actualMsg.Content, "Message Content mismatch at index %d", i)
							require.Len(t, actualMsg.ToolInvocations, len(expectedMsg.ToolInvocations), "Number of ToolInvocations mismatch at index %d", i)

							if len(expectedMsg.ToolInvocations) > 0 && len(actualMsg.ToolInvocations) > 0 {
								for j := range expectedMsg.ToolInvocations {
									if j >= len(actualMsg.ToolInvocations) {
										t.Fatalf("ToolInvocation index %d out of bounds for actual output (len %d) at message index %d", j, len(actualMsg.ToolInvocations), i)
									}
									expectedTI := expectedMsg.ToolInvocations[j]
									actualTI := actualMsg.ToolInvocations[j]
									require.Equal(t, expectedTI.ToolCallID, actualTI.ToolCallID, "ToolCallID mismatch at msg %d, ti %d", i, j)
									require.Equal(t, expectedTI.ToolName, actualTI.ToolName, "ToolName mismatch at msg %d, ti %d", i, j)
									require.Equal(t, expectedTI.State, actualTI.State, "State mismatch at msg %d, ti %d", i, j)
									require.EqualValues(t, expectedTI.Args, actualTI.Args, "Args mismatch at msg %d, ti %d", i, j)
									require.EqualValues(t, expectedTI.Result, actualTI.Result, "Result mismatch at msg %d, ti %d", i, j)
								}
							}
						}
					}
				}
			}
		})
	}
}
