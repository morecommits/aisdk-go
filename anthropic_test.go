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
	"github.com/coder/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestAnthropicToDataStream(t *testing.T) {
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
	typedStream := ssestream.NewStream[anthropic.MessageStreamEventUnion](decoder, nil)

	var acc aisdk.DataStreamAccumulator
	stream := aisdk.AnthropicToDataStream(typedStream)
	stream = stream.WithToolCalling(func(toolCall aisdk.ToolCall) any {
		return map[string]any{"message": "Message printed to the console"}
	})
	stream = stream.WithAccumulator(&acc)
	for _, err := range stream {
		require.NoError(t, err)
	}

	expectedMessages := []aisdk.Message{
		{
			ID:      "msg_01LHXQM4FBxykQGT7N1a7kJ7",
			Role:    "assistant",
			Content: "I'll help you print 'hello world' to the console using the print function.",
			Parts: []aisdk.Part{{
				Type: aisdk.PartTypeStepStart,
			}, {
				Type: aisdk.PartTypeText,
				Text: "I'll help you print 'hello world' to the console using the print function.",
			}, {
				Type: aisdk.PartTypeToolInvocation,
				ToolInvocation: &aisdk.ToolInvocation{
					State:      aisdk.ToolInvocationStateResult,
					ToolCallID: "toolu_01RA76iwg1LbKuDjJnc6ym45",
					ToolName:   "print",
					Args:       map[string]any{"message": "hello world"},
					Result:     map[string]any{"message": "Message printed to the console"},
				},
			}},
		},
	}

	require.EqualExportedValues(t, expectedMessages, acc.Messages())

	// --- Add conversion back check ---
	anthropicMsgs, systemPrompts, err := aisdk.MessagesToAnthropic(acc.Messages())
	require.NoError(t, err)

	// Verify the converted Anthropic messages match expectations
	// We now expect TWO messages: assistant (text + tool_use) and user (tool_result)
	require.Empty(t, systemPrompts)
	require.Len(t, anthropicMsgs, 2)

	// Check Assistant Message (Call)
	assistantMsg := anthropicMsgs[0]
	require.Equal(t, anthropic.MessageParamRoleAssistant, assistantMsg.Role)
	require.Len(t, assistantMsg.Content, 2) // Text block + ToolUse block

	// Check Text Content Block
	textBlock := assistantMsg.Content[0].OfText
	require.NotNil(t, textBlock)
	require.Equal(t, "I'll help you print 'hello world' to the console using the print function.", textBlock.Text)

	// Check Tool Use Content Block
	toolUseBlock := assistantMsg.Content[1].OfToolUse
	require.NotNil(t, toolUseBlock)
	require.Equal(t, "toolu_01RA76iwg1LbKuDjJnc6ym45", toolUseBlock.ID)
	require.Equal(t, "print", toolUseBlock.Name)
	require.JSONEq(t, `{"message": "hello world"}`, string(toolUseBlock.Input.(json.RawMessage)))

	// Check User Message (Result) - This message is now generated by the first conversion
	userMsg := anthropicMsgs[1]
	require.Equal(t, anthropic.MessageParamRoleUser, userMsg.Role)
	require.Len(t, userMsg.Content, 1) // ToolResult block

	toolResultBlock := userMsg.Content[0].OfToolResult
	require.NotNil(t, toolResultBlock)
	require.Equal(t, "toolu_01RA76iwg1LbKuDjJnc6ym45", toolResultBlock.ToolUseID)
	require.Len(t, toolResultBlock.Content, 1)
	require.NotNil(t, toolResultBlock.Content[0].OfText)
	require.JSONEq(t, `{"message":"Message printed to the console"}`, toolResultBlock.Content[0].OfText.Text)

	// --- Second conversion check (using expectedMessages) ---
	// This part should remain the same, as it also expects 2 messages now.
	anthropicMsgsWithResult, systemPromptsWithResult, err := aisdk.MessagesToAnthropic(expectedMessages)
	require.NoError(t, err)
	require.Empty(t, systemPromptsWithResult)
	require.Len(t, anthropicMsgsWithResult, 2) // Expect assistant call + user result

	// Check Assistant Message (unchanged from above)
	assistantMsgWithResult := anthropicMsgsWithResult[0]
	require.Equal(t, anthropic.MessageParamRoleAssistant, assistantMsgWithResult.Role)
	require.Len(t, assistantMsgWithResult.Content, 2) // Text block + ToolUse block (same as before)

	// Check User Message (Tool Result)
	userMsgWithResult := anthropicMsgsWithResult[1]
	require.Equal(t, anthropic.MessageParamRoleUser, userMsgWithResult.Role)
	require.Len(t, userMsgWithResult.Content, 1) // ToolResult block

	toolResultBlockWithResult := userMsgWithResult.Content[0].OfToolResult
	require.NotNil(t, toolResultBlockWithResult)
	require.Equal(t, "toolu_01RA76iwg1LbKuDjJnc6ym45", toolResultBlockWithResult.ToolUseID)
	require.Len(t, toolResultBlockWithResult.Content, 1)
	require.NotNil(t, toolResultBlockWithResult.Content[0].OfText)
	require.JSONEq(t, `{"message":"Message printed to the console"}`, toolResultBlockWithResult.Content[0].OfText.Text)
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
	prompt := "use the 'print' tool to print 'Hello, world!' and then show the result"
	messages, systemPrompts, err := aisdk.MessagesToAnthropic([]aisdk.Message{
		{
			Role: "system",
			Parts: []aisdk.Part{
				{Text: "You are a helpful assistant.", Type: aisdk.PartTypeText},
			},
		},
		{
			Role: "user", Parts: []aisdk.Part{
				{Text: prompt, Type: aisdk.PartTypeText},
			},
		},
	})
	require.Len(t, messages, 1)
	require.Len(t, systemPrompts, 1)
	require.Len(t, messages[0].Content, 1)
	require.NotNil(t, messages[0].Content[0].OfText)
	require.Equal(t, messages[0].Content[0].OfText.Text, prompt)
	require.NoError(t, err)

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Messages:  messages,
		Model:     anthropic.ModelClaude3_5SonnetLatest,
		System:    systemPrompts,
		MaxTokens: 10,
	})
	require.NoError(t, err)

	dataStream := aisdk.AnthropicToDataStream(stream)
	var streamErr error
	dataStream(func(part aisdk.DataStreamPart, err error) bool {
		if err != nil {
			streamErr = err
			return false
		}
		return true
	})
	require.NoError(t, streamErr)
}
