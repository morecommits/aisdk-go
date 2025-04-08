package aisdk

import (
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
)

// ToolsToAnthropic converts the tool format to Anthropic's API format.
func ToolsToAnthropic(tools []Tool) []anthropic.ToolUnionParam {
	anthropicTools := []anthropic.ToolUnionParam{}
	for _, tool := range tools {
		// Construct the ToolInputSchemaParam struct directly
		inputSchema := anthropic.ToolInputSchemaParam{
			Properties: tool.Schema.Properties, // Assuming Properties is map[string]interface{}
			// Type defaults to "object" via omitempty / SDK marshalling if needed
		}
		// Add required fields if they exist
		if len(tool.Schema.Required) > 0 {
			if inputSchema.ExtraFields == nil {
				inputSchema.ExtraFields = make(map[string]interface{})
			}
			inputSchema.ExtraFields["required"] = tool.Schema.Required
		}

		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: inputSchema, // Assign the struct directly
			},
		})
	}
	return anthropicTools
}

// MessagesToAnthropic converts internal message format to Anthropic's API format.
// It extracts system messages into a separate slice of TextBlockParams and groups
// consecutive user/tool and assistant messages according to Anthropic's rules.
// It handles the case where a single assistant message part contains both the
// tool call and its result, splitting them into the required assistant tool_use
// and user tool_result blocks.
func MessagesToAnthropic(messages []Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, error) {
	anthropicMessages := []anthropic.MessageParam{}
	systemPrompts := []anthropic.TextBlockParam{}

	// Iterate through messages and process them
	for i := 0; i < len(messages); i++ {
		message := messages[i]

		if message.Role == "system" {
			// Handle system messages
			systemPrompts = append(systemPrompts, anthropic.TextBlockParam{Text: message.Content})
			for _, part := range message.Parts {
				if part.Type == PartTypeText && part.Text != "" {
					systemPrompts = append(systemPrompts, anthropic.TextBlockParam{Text: part.Text})
				}
				// Ignore other parts in system messages for now
			}
			continue
		}

		var role anthropic.MessageParamRole
		var currentContent []anthropic.ContentBlockParamUnion

		if message.Role == "assistant" {
			role = anthropic.MessageParamRoleAssistant
			// Process parts for assistant message
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					if part.Text != "" {
						currentContent = append(currentContent, anthropic.ContentBlockParamUnion{
							OfRequestTextBlock: &anthropic.TextBlockParam{Text: part.Text},
						})
					}
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, nil, fmt.Errorf("assistant message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}

					// Add the tool *call* part
					argsJSON, err := json.Marshal(part.ToolInvocation.Args)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolInvocation.ToolCallID, err)
					}
					currentContent = append(currentContent, anthropic.ContentBlockParamUnion{
						OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    part.ToolInvocation.ToolCallID,
							Name:  part.ToolInvocation.ToolName,
							Input: json.RawMessage(argsJSON),
						},
					})

					// If the state is Result, we need to immediately add a user message with the result
					if part.ToolInvocation.State == ToolInvocationStateResult {
						// Flush the current assistant message first
						if len(currentContent) > 0 {
							anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
								Role:    role,
								Content: currentContent,
							})
							currentContent = nil // Reset for the potential next message
						}

						// Now create the user message with the tool result
						resultBytes, err := json.Marshal(part.ToolInvocation.Result)
						if err != nil {
							return nil, nil, fmt.Errorf("marshalling tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
						}
						userResultContent := []anthropic.ContentBlockParamUnion{
							{
								OfRequestToolResultBlock: &anthropic.ToolResultBlockParam{
									ToolUseID: part.ToolInvocation.ToolCallID,
									Content: []anthropic.ToolResultBlockParamContentUnion{
										{
											OfRequestTextBlock: &anthropic.TextBlockParam{Text: string(resultBytes)},
										},
									},
								},
							},
						}
						anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
							Role:    anthropic.MessageParamRoleUser, // Result must be in user role
							Content: userResultContent,
						})
						// Since we added the user message, effectively skip adding the current assistant message later
						role = "" // Mark role as processed
					}
					// TODO: Add support for other part types
				}
			}
		} else if message.Role == "user" || message.Role == "tool" {
			role = anthropic.MessageParamRoleUser
			// Process parts for user/tool message
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					if part.Text != "" {
						currentContent = append(currentContent, anthropic.ContentBlockParamUnion{
							OfRequestTextBlock: &anthropic.TextBlockParam{Text: part.Text},
						})
					}
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, nil, fmt.Errorf("user/tool message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}
					// User/tool role should only contain results
					if part.ToolInvocation.State != ToolInvocationStateResult {
						return nil, nil, fmt.Errorf("non-result tool invocation found in user/tool message (ID: %s, State: %s)", message.ID, part.ToolInvocation.State)
					}
					resultBytes, err := json.Marshal(part.ToolInvocation.Result)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
					}
					currentContent = append(currentContent, anthropic.ContentBlockParamUnion{
						OfRequestToolResultBlock: &anthropic.ToolResultBlockParam{
							ToolUseID: part.ToolInvocation.ToolCallID,
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{
									OfRequestTextBlock: &anthropic.TextBlockParam{Text: string(resultBytes)},
								},
							},
						},
					})
					// TODO: Add support for other part types
				}
			}
			// Add plain content as text block if no parts were processed
			if len(currentContent) == 0 && message.Content != "" {
				currentContent = append(currentContent, anthropic.ContentBlockParamUnion{
					OfRequestTextBlock: &anthropic.TextBlockParam{Text: message.Content},
				})
			}
		} else {
			return nil, nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}

		// Add the processed message if it has content and role wasn't handled by tool result splitting
		if len(currentContent) > 0 && role != "" {
			// Check if the last message was of the same role, if so, merge content
			lastIdx := len(anthropicMessages) - 1
			if lastIdx >= 0 && anthropicMessages[lastIdx].Role == role {
				anthropicMessages[lastIdx].Content = append(anthropicMessages[lastIdx].Content, currentContent...)
			} else {
				anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
					Role:    role,
					Content: currentContent,
				})
			}
		}
	}

	return anthropicMessages, systemPrompts, nil
}

// AnthropicToDataStream pipes an Anthropic stream to a DataStream.
func AnthropicToDataStream(stream *ssestream.Stream[anthropic.MessageStreamEventUnion]) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		var lastChunk *anthropic.MessageStreamEventUnion
		var finalReason FinishReason = FinishReasonUnknown
		var finalUsage Usage
		var currentToolCall struct {
			ID   string
			Args string
		}

		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			event := chunk.AsAny()
			switch event := event.(type) {
			case anthropic.MessageStartEvent:
				if !yield(StartStepStreamPart{
					MessageID: event.Message.ID,
				}, nil) {
					return
				}

			case anthropic.ContentBlockDeltaEvent:
				switch delta := event.Delta.AsAny().(type) {
				case anthropic.TextDelta:
					if !yield(TextStreamPart{Content: delta.Text}, nil) {
						return
					}
				case anthropic.InputJSONDelta:
					// Accumulate the arguments for the current tool call
					currentToolCall.Args += delta.PartialJSON
					if !yield(ToolCallDeltaStreamPart{
						ToolCallID:    currentToolCall.ID,
						ArgsTextDelta: delta.PartialJSON,
					}, nil) {
						return
					}
				case anthropic.ThinkingDelta:
					if !yield(ReasoningStreamPart{Content: delta.Thinking}, nil) {
						return
					}
				}

			case anthropic.ContentBlockStartEvent:
				if block, ok := event.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
					currentToolCall.ID = block.ID
					currentToolCall.Args = ""

					if !yield(ToolCallStartStreamPart{
						ToolCallID: block.ID,
						ToolName:   block.Name,
					}, nil) {
						return
					}
				}

			case anthropic.MessageDeltaEvent:
				if event.Delta.StopReason == "tool_use" {
					finalReason = FinishReasonToolCalls
					if event.Usage.OutputTokens != 0 {
						tokens := event.Usage.OutputTokens
						finalUsage.CompletionTokens = &tokens
					}

					// Reset current tool call after emitting the final delta
					currentToolCall = struct {
						ID   string
						Args string
					}{}
				}

			case anthropic.MessageStopEvent:
				// Determine final reason if not already set by tool_use
				if finalReason == FinishReasonUnknown {
					finalReason = FinishReasonStop // Default if not tool_use
				}

				// Send final finish step
				if !yield(FinishStepStreamPart{
					FinishReason: finalReason,
					Usage:        finalUsage,
					IsContinued:  false,
				}, nil) {
					return
				}

				// Send final finish message
				if !yield(FinishMessageStreamPart{
					FinishReason: finalReason,
					Usage:        finalUsage,
				}, nil) {
					return
				}
			}
		}

		// Handle any errors from the stream
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("anthropic stream error: %w", err))
			return
		}

		// If we didn't get a message stop event (e.g., stream ended abruptly),
		// send a final finish message based on the last known state.
		if lastChunk == nil || lastChunk.Type != "message_stop" {
			if finalReason == FinishReasonUnknown {
				finalReason = FinishReasonError // Indicate abnormal termination
			}

			yield(FinishMessageStreamPart{
				FinishReason: finalReason,
				Usage:        finalUsage,
			}, nil)
		}
	}
}
