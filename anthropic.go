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
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: tool.Parameters,
				},
			},
		})
	}
	return anthropicTools
}

// MessagesToAnthropic converts internal message format to Anthropic's API format.
// It extracts system messages into a separate slice of TextBlockParams, suitable
// for the 'System' parameter in the Anthropic API request.
func MessagesToAnthropic(messages []Message) ([]anthropic.MessageParam, []anthropic.TextBlockParam, error) {
	anthropicMessages := []anthropic.MessageParam{}
	systemPrompts := []anthropic.TextBlockParam{}

	for _, message := range messages {
		switch message.Role {
		case "user":
			// Construct ContentBlockParamUnion for user messages.
			userContent := []anthropic.ContentBlockParamUnion{
				{OfRequestTextBlock: &anthropic.TextBlockParam{Text: message.Content}},
			}
			anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
				Role:    "user",
				Content: userContent,
			})

		case "assistant":
			// Construct ContentBlockParamUnions for the assistant's response part (text + tool calls).
			assistantContentBlocks := []anthropic.ContentBlockParamUnion{}
			// Construct ContentBlockParamUnions for the subsequent user message containing tool results.
			toolResultBlocks := []anthropic.ContentBlockParamUnion{}

			// Add text content if present.
			if message.Content != "" {
				assistantContentBlocks = append(assistantContentBlocks, anthropic.ContentBlockParamUnion{
					OfRequestTextBlock: &anthropic.TextBlockParam{Text: message.Content},
				})
			}

			// Process tool invocations.
			for _, ti := range message.ToolInvocations {
				switch ti.State {
				case ToolInvocationStateCall, ToolInvocationStatePartialCall:
					// Add tool_use block to the assistant message content.
					argsJSON, err := json.Marshal(ti.Args)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool args for call %s: %w", ti.ToolCallID, err)
					}
					assistantContentBlocks = append(assistantContentBlocks, anthropic.ContentBlockParamUnion{
						OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    ti.ToolCallID,
							Name:  ti.ToolName,
							Input: json.RawMessage(argsJSON),
						},
					})
				case ToolInvocationStateResult:
					// Add BOTH the tool_use block to the assistant message
					// AND the tool_result block to the subsequent user message.

					// 1. Add tool_use block to assistant content
					argsJSON, err := json.Marshal(ti.Args)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool args for call %s: %w", ti.ToolCallID, err)
					}
					assistantContentBlocks = append(assistantContentBlocks, anthropic.ContentBlockParamUnion{
						OfRequestToolUseBlock: &anthropic.ToolUseBlockParam{
							ID:    ti.ToolCallID,
							Name:  ti.ToolName,
							Input: json.RawMessage(argsJSON),
						},
					})

					// 2. Add tool_result block for the next user message
					resultJSON, err := json.Marshal(ti.Result)
					if err != nil {
						return nil, nil, fmt.Errorf("marshalling tool result for call %s: %w", ti.ToolCallID, err)
					}
					// Assuming result content should be stringified JSON, and isError is false.
					toolResultUnion := anthropic.NewToolResultBlock(ti.ToolCallID, string(resultJSON), false)
					toolResultBlocks = append(toolResultBlocks, toolResultUnion)
				}
			}

			// Add the assistant message only if it has actual content (text or tool use blocks)
			// relevant to the Anthropic API conversation history.
			// Reasoning-only messages (empty content, no tool calls) are skipped.
			if message.Content != "" || len(assistantContentBlocks) > 0 {
				// If the message is effectively empty *before* considering tool results
				// (no text content, no tool calls initiated in *this* turn),
				// Anthropic might still require a content block if it's an otherwise empty assistant turn.
				// Add an empty text block specifically for this case.
				if message.Content == "" && len(assistantContentBlocks) == 0 {
					assistantContentBlocks = append(assistantContentBlocks, anthropic.ContentBlockParamUnion{
						OfRequestTextBlock: &anthropic.TextBlockParam{Text: ""},
					})
				}

				// Append the message if it has any content blocks (original or the empty placeholder)
				if len(assistantContentBlocks) > 0 {
					anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
						Role:    anthropic.MessageParamRoleAssistant,
						Content: assistantContentBlocks,
					})
				}
			}

			// If there were tool results, add them in a new user message.
			if len(toolResultBlocks) > 0 {
				// Anthropic requires tool results to be in a user message.
				anthropicMessages = append(anthropicMessages, anthropic.MessageParam{
					Role:    "user",
					Content: toolResultBlocks,
				})
			}

		case "system":
			// Append as TextBlockParam
			systemPrompts = append(systemPrompts, anthropic.TextBlockParam{Text: message.Content})

		default:
			return nil, nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return anthropicMessages, systemPrompts, nil
}

// AnthropicToMessages converts Anthropic's API message format back to the internal []Message format.
// It takes both the message history and the system prompts (which Anthropic handles separately).
func AnthropicToMessages(anthropicMessages []anthropic.MessageParam) ([]Message, error) {
	messages := []Message{}
	var lastAssistantMessage *Message // Keep track of the last assistant message to potentially attach tool results

	for _, anthropicMsg := range anthropicMessages {
		var currentMessage Message
		currentMessage.Role = string(anthropicMsg.Role) // Role is directly convertible
		// toolInvocations slice will be populated within the loop
		toolInvocations := []ToolInvocation{}

		for _, contentUnion := range anthropicMsg.Content {
			// Check union fields directly instead of using AsAny
			switch {
			case contentUnion.OfRequestTextBlock != nil:
				block := contentUnion.OfRequestTextBlock
				// Append text content.
				currentMessage.Content += block.Text
			case contentUnion.OfRequestToolUseBlock != nil:
				block := contentUnion.OfRequestToolUseBlock
				// This is a tool call initiated by the assistant.
				var argsMap map[string]interface{} // Use interface{} for consistency
				// Marshal the Input interface{} back to JSON bytes
				inputBytes, err := json.Marshal(block.Input)
				if err != nil {
					// Handle marshalling error (less likely for valid input struct)
					argsMap = nil
					fmt.Printf("Warning: Failed to marshal tool use input interface for %s (%s): %v\n", block.Name, block.ID, err)
				} else {
					// Unmarshal the JSON bytes into our map
					err = json.Unmarshal(inputBytes, &argsMap)
					if err != nil {
						// Handle potential unmarshal error gracefully
						argsMap = nil
						fmt.Printf("Warning: Failed to unmarshal tool use input JSON for %s (%s): %v\n", block.Name, block.ID, err)
					}
				}
				// Add to the *local* toolInvocations slice for this message
				toolInvocations = append(toolInvocations, ToolInvocation{
					ToolCallID: block.ID,
					ToolName:   block.Name,
					Args:       argsMap,
					State:      ToolInvocationStateCall, // Mark as a call
				})
			case contentUnion.OfRequestToolResultBlock != nil:
				block := contentUnion.OfRequestToolResultBlock
				// This is a tool result provided by the user.
				// It needs to be attached to the corresponding call in the *previous* assistant message.
				if lastAssistantMessage == nil {
					return nil, fmt.Errorf("received tool result block (user role) without a preceding assistant message containing the tool call")
				}
				found := false
				for i := range lastAssistantMessage.ToolInvocations {
					if lastAssistantMessage.ToolInvocations[i].ToolCallID == block.ToolUseID {
						// Attempt to get result directly from block.Content via type assertion
						resultData := block.Content[0].OfRequestTextBlock.Text
						lastAssistantMessage.ToolInvocations[i].Result = resultData
						lastAssistantMessage.ToolInvocations[i].State = ToolInvocationStateResult
						found = true
						break
					}
				}
				if !found {
					return nil, fmt.Errorf("received tool result for unknown tool use ID: %s", block.ToolUseID)
				}
			default:
				var typeName string
				if contentUnion.OfRequestTextBlock != nil {
					typeName = fmt.Sprintf("%T", contentUnion.OfRequestTextBlock)
				} else if contentUnion.OfRequestToolUseBlock != nil {
					typeName = fmt.Sprintf("%T", contentUnion.OfRequestToolUseBlock)
				} else if contentUnion.OfRequestToolResultBlock != nil {
					typeName = fmt.Sprintf("%T", contentUnion.OfRequestToolResultBlock)
				} else {
					typeName = "unknown (nil union field)"
				}
				return nil, fmt.Errorf("unsupported Anthropic content block type: %s", typeName)
			}
		}

		// Unconditionally assign the collected tool invocations (will be empty for non-assistant or no-tool messages)
		currentMessage.ToolInvocations = toolInvocations

		// Add the composed message to our list, unless it's a user message that ONLY contained tool results.
		isUserMessage := currentMessage.Role == string(anthropic.MessageParamRoleUser)
		hasToolResultsOnly := false
		if isUserMessage {
			hasToolResultsOnly = true // Assume true initially
			for _, contentUnion := range anthropicMsg.Content {
				if contentUnion.OfRequestToolResultBlock == nil {
					hasToolResultsOnly = false // Found non-result content (e.g., text)
					break
				}
			}
		}

		if !hasToolResultsOnly {
			messages = append(messages, currentMessage)
		}

		// Update lastAssistantMessage pointer for the next iteration.
		// Important: Use the message *just added* to the slice if it was an assistant message
		if !hasToolResultsOnly && currentMessage.Role == string(anthropic.MessageParamRoleAssistant) {
			lastAssistantMessage = &messages[len(messages)-1]
		} else if !hasToolResultsOnly {
			lastAssistantMessage = nil // Reset if it wasn't an assistant or was skipped
		}
		// If hasToolResultsOnly was true, lastAssistantMessage remains unchanged from previous iteration.
	}

	return messages, nil
}

// PipeAnthropicToDataStream pipes an Anthropic message stream to a DataStream.
// It translates Anthropic stream events into DataStreamParts and calls the optional
// HandleToolCall function when a complete tool use block is received.
// It returns the final accumulated message from the stream.
func PipeAnthropicToDataStream(stream *ssestream.Stream[anthropic.MessageStreamEventUnion], dataStream DataStream, opts *PipeOptions) (*PipeResponse[[]anthropic.MessageParam], error) {
	message := anthropic.Message{}
	toolCalls := map[int]*wipToolCall{}
	toolResults := []anthropic.ContentBlockParamUnion{}
	for stream.Next() {
		eventUnion := stream.Current()

		// Accumulate the event's effects on the overall message state first.
		// This ensures message.Usage and message.StopReason are updated before generating final parts.
		err := message.Accumulate(eventUnion)
		if err != nil {
			// Log or handle accumulation error if necessary
			return nil, fmt.Errorf("failed to accumulate anthropic stream event: %w", err)
		}

		event := eventUnion.AsAny()

		switch event := event.(type) {
		case anthropic.MessageStartEvent:
			// Send the StartStepStreamPart ('f') here, using the actual message ID from Anthropic.
			err = dataStream.Write(StartStepStreamPart{
				MessageID: event.Message.ID,
			})
			if err != nil {
				return nil, err
			}
			continue

		case anthropic.ContentBlockStartEvent:
			// If it's a tool use block, start tracking it and send 'b' part.
			if block, ok := event.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
				index := int(event.Index)
				// Ensure we don't re-initialize if stream is weird, though Accumulate might error first.
				if _, exists := toolCalls[index]; !exists {
					wip := &wipToolCall{
						ID:   block.ID,
						Name: block.Name,
						Args: "",
					}
					toolCalls[index] = wip
					err := dataStream.Write(ToolCallStartStreamPart{
						ToolCallID: wip.ID,
						ToolName:   wip.Name,
					})
					if err != nil {
						return nil, err
					}
				}
			}
			// No specific Vercel part for text content_block_start.

		case anthropic.ContentBlockDeltaEvent:
			index := int(event.Index)
			switch delta := event.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				// Send text content as '0' part.
				if delta.Text != "" {
					err := dataStream.Write(TextStreamPart{Content: delta.Text})
					if err != nil {
						return nil, err
					}
				}
			case anthropic.InputJSONDelta:
				// Append to tool call args buffer and send 'c' part.
				if wip, exists := toolCalls[index]; exists && !wip.Finished {
					wip.Args += delta.PartialJSON
					// Send delta even if empty, as it's part of the stream sequence.
					err := dataStream.Write(ToolCallDeltaStreamPart{
						ToolCallID:    wip.ID,
						ArgsTextDelta: delta.PartialJSON,
					})
					if err != nil {
						return nil, err
					}
				} else {
					// Optional: Log warning for delta targeting unknown/finished tool call
					// fmt.Printf("Warning: InputJSONDelta for unknown/finished tool call index %d\n", index)
				}
			case anthropic.ThinkingDelta:
				err = dataStream.Write(ReasoningStreamPart{
					Content: delta.Thinking,
				})
				if err != nil {
					return nil, err
				}
			}

		case anthropic.ContentBlockStopEvent:
			index := int(event.Index)
			// If it's a tool call we were tracking, finalize it.
			if wip, exists := toolCalls[index]; exists && !wip.Finished {
				wip.Finished = true
				var args map[string]any
				err := json.Unmarshal([]byte(wip.Args), &args)
				if err != nil {
					// Error handling: Could send an ErrorStreamPart ('3') or log.
					// For now, log warning and don't send '9' or 'a'.
					fmt.Printf("Warning: Could not unmarshal tool args for %s (%s): %v. Args received: %s\n", wip.Name, wip.ID, err, wip.Args)
					// Optionally send an error part:
					// dataStream.Write(ErrorStreamPart{Content: fmt.Sprintf("Failed to parse arguments for tool %s", wip.Name)})
				} else {
					// Send final tool call info as '9' part.
					err = dataStream.Write(ToolCallStreamPart{
						ToolCallID: wip.ID,
						ToolName:   wip.Name,
						Args:       args,
					})
					if err != nil {
						return nil, err
					}

					// Determine and send tool result as 'a' part.
					var result any
					if opts != nil && opts.HandleToolCall != nil {
						// Execute the provided handler.
						result = opts.HandleToolCall(ToolCall{
							ID:   wip.ID,
							Name: wip.Name,
							Args: args,
						})
					} else {
						// Default result if no handler is provided.
						result = "No tool call handler provided"
					}
					err = dataStream.Write(ToolResultStreamPart{
						ToolCallID: wip.ID,
						Result:     result,
					})
					if err != nil {
						return nil, err
					}

					jsonResult, err := json.Marshal(result)
					if err != nil {
						return nil, err
					}
					toolResults = append(toolResults, anthropic.NewToolResultBlock(wip.ID, string(jsonResult), false))
				}
			}
			// No specific Vercel part for text content_block_stop.

		case anthropic.MessageDeltaEvent:
			// This event signals changes like stop reason or usage, often marking a step end.
			// Map to FinishStepStreamPart ('e').
			var stepReason FinishReason = FinishReasonUnknown
			var isContinued bool = false
			// Use the stop reason *from the delta* for the step finish reason.
			// Check if the stop reason string is non-empty.
			if event.Delta.StopReason != "" {
				switch event.Delta.StopReason {
				case "end_turn", "stop_sequence":
					stepReason = FinishReasonStop
				case "max_tokens":
					stepReason = FinishReasonLength
				case "tool_use":
					stepReason = FinishReasonToolCalls
					isContinued = false
				default:
					fmt.Printf("Warning: Unknown Anthropic stop reason in delta: %s\n", event.Delta.StopReason)
					stepReason = FinishReasonOther
				}
			}

			// Only write 'e' part if the delta provided a stop reason.
			if stepReason != FinishReasonUnknown {
				// Usage for the step: Reflect the accumulated usage *up to this point*.
				var stepInputTokens, stepOutputTokens *int64
				if message.Usage.JSON.InputTokens.IsPresent() {
					tokens := int64(message.Usage.InputTokens)
					stepInputTokens = &tokens
				}
				// Use accumulated output tokens which includes the delta's contribution.
				if message.Usage.JSON.OutputTokens.IsPresent() {
					tokens := int64(message.Usage.OutputTokens)
					stepOutputTokens = &tokens
				}

				err = dataStream.Write(FinishStepStreamPart{
					FinishReason: stepReason,
					Usage: Usage{
						PromptTokens:     stepInputTokens,
						CompletionTokens: stepOutputTokens,
					},
					IsContinued: isContinued,
				})
				if err != nil {
					return nil, err
				}
			}

		case anthropic.MessageStopEvent:
			// Message state already updated by Accumulate. Signal end of stream processing.
			goto endStreamLoop

		// case anthropic.PingEvent:
		// Ping events are typically handled by the underlying SSE client or Accumulate.
		// No specific Vercel part needed.
		// continue

		default:
			// Any other event types are handled by Accumulate, but no specific Vercel part generated here.
			// fmt.Printf("Info: Unhandled Anthropic event type in switch: %T\n", event)
			continue
		}

	}

endStreamLoop:

	// Finished processing stream events. Write the final 'd' part.
	var finalReason FinishReason
	var inputTokens, outputTokens *int64

	// Use the final accumulated usage.
	if message.Usage.JSON.InputTokens.IsPresent() {
		tokens := int64(message.Usage.InputTokens)
		inputTokens = &tokens
	}
	if message.Usage.JSON.OutputTokens.IsPresent() {
		tokens := int64(message.Usage.OutputTokens)
		outputTokens = &tokens
	}

	// Use the final accumulated stop reason.
	switch message.StopReason {
	case "end_turn", "stop_sequence":
		finalReason = FinishReasonStop
	case "max_tokens":
		finalReason = FinishReasonLength
	case "tool_use":
		finalReason = FinishReasonToolCalls
	default:
		// Determine reason if not explicitly set by Anthropic.
		if stream.Err() != nil {
			// If the stream ended with an error.
			finalReason = FinishReasonError
			fmt.Printf("Stream finished with error: %v\n", stream.Err())
		} else if message.StopReason != "" {
			// If Accumulate recorded an unknown reason.
			fmt.Printf("Warning: Unknown final Anthropic stop reason: %s\n", message.StopReason)
			finalReason = FinishReasonOther
		} else {
			// Default if stream ended cleanly without a specific reason (e.g., empty stream).
			finalReason = FinishReasonStop
		}
	}

	err := dataStream.Write(FinishMessageStreamPart{
		FinishReason: finalReason,
		Usage: Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
		},
	})
	if err != nil {
		// Don't mask stream errors if writing the final part fails.
		if stream.Err() != nil {
			return nil, fmt.Errorf("stream error (%v) and final write error (%w)", stream.Err(), err)
		}
		return nil, err
	}

	msgs := []anthropic.MessageParam{}
	msgs = append(msgs, message.ToParam())
	if len(toolResults) > 0 {
		msgs = append(msgs, anthropic.NewUserMessage(toolResults...))
	}

	return &PipeResponse[[]anthropic.MessageParam]{
		Messages:     msgs,
		FinishReason: finalReason,
	}, stream.Err()
}
