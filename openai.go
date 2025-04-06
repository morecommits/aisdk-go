package aisdk

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// MessagesToOpenAI converts internal message format to OpenAI's API format.
func MessagesToOpenAI(messages []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	openaiMessages := []openai.ChatCompletionMessageParamUnion{}

	for _, message := range messages {
		switch message.Role {
		case "user":
			// User messages just have content.
			// The helper openai.UserMessage handles creating the correct Content union (OfString or OfArrayOfContentParts).
			// Since we aren't handling attachments (which would require parts), using the string content directly is simplest.
			openaiMessages = append(openaiMessages, openai.UserMessage(message.Content))

		case "system":
			// System messages also just have content.
			openaiMessages = append(openaiMessages, openai.SystemMessage(message.Content))

		case "assistant":
			// Assistant messages can have content and/or tool calls.
			if len(message.ToolInvocations) > 0 {
				toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(message.ToolInvocations))
				hasCalls := false

				// First pass: Collect tool calls for the assistant message itself.
				for _, ti := range message.ToolInvocations {
					// OpenAI requires the tool_calls array on the assistant message
					// even if the state in our internal model is "result".
					// The subsequent "tool" role message will provide the result content.
					// So, we collect call info regardless of the state here.
					argsJSON, err := json.Marshal(ti.Args)
					if err != nil {
						return nil, fmt.Errorf("marshalling tool args for call %s: %w", ti.ToolCallID, err)
					}
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: ti.ToolCallID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      ti.ToolName,
							Arguments: string(argsJSON),
						},
					})
					hasCalls = true
				}

				// Construct the assistant message param.
				assistantMsg := openai.ChatCompletionAssistantMessageParam{}

				// Add tool calls if any were generated.
				if hasCalls {
					assistantMsg.ToolCalls = toolCalls
				}

				// Add text content part *if* content exists, even if there are tool calls.
				// Mimics Python example structure. Content type for assistant parts is different.
				if message.Content != "" {
					assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfArrayOfContentParts: []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
							{OfText: &openai.ChatCompletionContentPartTextParam{
								Text: message.Content,
							}},
						},
					}
				}

				// Append the assistant message itself.
				openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})

				// Second pass: Append separate 'tool' messages for results.
				for _, ti := range message.ToolInvocations {
					if ti.State == ToolInvocationStateResult {
						resultJSON, err := json.Marshal(ti.Result)
						if err != nil {
							return nil, fmt.Errorf("marshalling tool result for call %s: %w", ti.ToolCallID, err)
						}
						// The ToolMessage helper handles creating the Content union correctly for a string.
						openaiMessages = append(openaiMessages, openai.ToolMessage(string(resultJSON), ti.ToolCallID))
					}
				}

			} else {
				// Assistant message with no tool calls, just content.
				openaiMessages = append(openaiMessages, openai.AssistantMessage(message.Content))
			}
		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return openaiMessages, nil
}

// OpenAIToMessages converts OpenAI's API message format back to the internal []Message format.
func OpenAIToMessages(openaiMessages []openai.ChatCompletionMessageParamUnion) ([]Message, error) {
	messages := []Message{}
	var lastAssistantMessage *Message // Keep track of the last assistant message to attach tool results

	for _, msgUnion := range openaiMessages {
		var role string
		var content string
		var toolCalls []ToolInvocation // Use the internal ToolInvocation struct

		switch {
		case msgUnion.OfSystem != nil:
			role = "system"
			// System message content union likely only contains OfString.
			if msgUnion.OfSystem.Content.OfString.IsPresent() {
				content = msgUnion.OfSystem.Content.OfString.Value
			}
		case msgUnion.OfUser != nil:
			role = "user"
			// User content is a union.
			userContentUnion := msgUnion.OfUser.Content
			switch {
			case userContentUnion.OfString.IsPresent():
				if userContentUnion.OfString.Value != "" {
					content = userContentUnion.OfString.Value
				}
			case userContentUnion.OfArrayOfContentParts != nil:
				// Combine text parts.
				for _, part := range userContentUnion.OfArrayOfContentParts {
					if part.OfText != nil {
						content += part.OfText.Text
					}
					// TODO: Handle other user content part types (e.g., images) if needed.
				}
			}
		case msgUnion.OfAssistant != nil:
			role = "assistant"
			// Assistant content is a union (not a pointer to one).
			assistantContentUnion := msgUnion.OfAssistant.Content
			switch {
			case assistantContentUnion.OfString.IsPresent():
				if assistantContentUnion.OfString.Value != "" {
					content = assistantContentUnion.OfString.Value
				}
			case assistantContentUnion.OfArrayOfContentParts != nil:
				// Combine text parts.
				for _, part := range assistantContentUnion.OfArrayOfContentParts {
					if part.OfText != nil {
						content += part.OfText.Text
					}
				}
			}

			// Handle tool calls initiated by the assistant (these are outside the Content union)
			if len(msgUnion.OfAssistant.ToolCalls) > 0 {
				for _, tc := range msgUnion.OfAssistant.ToolCalls {
					// Initialize argsMap to nil to ensure it's nil if Unmarshal fails
					var argsMap map[string]interface{} = nil
					// Arguments are JSON strings, need to unmarshal
					err := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap)
					if err != nil {
						argsMap = nil
					}
					toolCalls = append(toolCalls, ToolInvocation{
						ToolCallID: tc.ID,
						ToolName:   tc.Function.Name,
						Args:       argsMap,
						State:      ToolInvocationStateCall, // Initially a call
					})
				}
			}

		case msgUnion.OfTool != nil:
			// Tool messages provide results for previous assistant tool calls.
			if lastAssistantMessage == nil {
				return nil, fmt.Errorf("received tool message without a preceding assistant message")
			}
			toolCallID := msgUnion.OfTool.ToolCallID
			found := false
			for i := range lastAssistantMessage.ToolInvocations {
				if lastAssistantMessage.ToolInvocations[i].ToolCallID == toolCallID {
					// Extract the string content from the tool message union.
					var toolResultContent string
					if msgUnion.OfTool.Content.OfString.IsPresent() {
						toolResultContent = msgUnion.OfTool.Content.OfString.Value
					}
					lastAssistantMessage.ToolInvocations[i].Result = toolResultContent // Assign extracted string
					lastAssistantMessage.ToolInvocations[i].State = ToolInvocationStateResult
					found = true
					break
				}
			}
			if !found {
				return nil, fmt.Errorf("received tool result for unknown tool call ID: %s", toolCallID)
			}
			// Tool messages don't create a new message in our internal format,
			// they modify the last assistant message. So, continue to the next OpenAI message.
			continue

		default:
			return nil, fmt.Errorf("unsupported OpenAI message type in union")
		}

		newMessage := Message{
			Role:            role,
			Content:         content,
			ToolInvocations: toolCalls, // Assign potentially empty slice
		}
		messages = append(messages, newMessage)

		// If this was an assistant message, store a pointer to it
		if role == "assistant" {
			lastAssistantMessage = &messages[len(messages)-1]
		} else {
			lastAssistantMessage = nil // Reset if the current message is not assistant
		}
	}

	return messages, nil
}

// PipeOpenAIToDataStream pipes an OpenAI stream to a DataStream.
func PipeOpenAIToDataStream(stream *ssestream.Stream[openai.ChatCompletionChunk], dataStream DataStream, opts *PipeOptions) (*PipeResponse[[]openai.ChatCompletionMessageParamUnion], error) {
	toolCalls := map[int64]*wipToolCall{}
	var lastChunk *openai.ChatCompletionChunk

	finish := func(chunk *openai.ChatCompletionChunk) error {
		if chunk == nil {
			return nil
		}
		choice := chunk.Choices[0]
		var (
			promptTokens     *int64
			completionTokens *int64
			reason           FinishReason
		)
		switch choice.FinishReason {
		case "tool_calls":
			reason = FinishReasonToolCalls
		default:
			reason = FinishReasonStop
		}
		if chunk.Usage.JSON.CompletionTokens.IsPresent() {
			completionTokens = &chunk.Usage.CompletionTokens
		}
		if chunk.Usage.JSON.PromptTokens.IsPresent() {
			promptTokens = &chunk.Usage.PromptTokens
		}
		return dataStream.Write(FinishStepStreamPart{
			FinishReason: reason,
			Usage: Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
			},
			IsContinued: false,
		})
	}

	acc := openai.ChatCompletionAccumulator{}
	msgs := []openai.ChatCompletionMessageParamUnion{}
	for stream.Next() {
		chunk := stream.Current()
		acc.AddChunk(chunk)
		lastChunk = &chunk

		if len(chunk.Choices) == 0 {
			break
		}
		// See: https://github.com/vercel/ai/blob/1789884487f11fb12206ebee1b7b0778d966e330/packages/openai/src/openai-chat-language-model.ts#L610
		choice := chunk.Choices[0]

		if choice.Delta.Content != "" {
			err := dataStream.Write(TextStreamPart{Content: choice.Delta.Content})
			if err != nil {
				return nil, err
			}
		}

		for _, toolCallDelta := range choice.Delta.ToolCalls {
			toolCall, ok := toolCalls[toolCallDelta.Index]
			if !ok {
				// Create it.
				toolCall = &wipToolCall{
					ID:       toolCallDelta.ID,
					Name:     toolCallDelta.Function.Name,
					Args:     toolCallDelta.Function.Arguments,
					Finished: false,
				}

				err := dataStream.Write(ToolCallStartStreamPart{
					ToolCallID: toolCall.ID,
					ToolName:   toolCall.Name,
				})
				if err != nil {
					return nil, err
				}

				toolCalls[toolCallDelta.Index] = toolCall
			}

			if toolCall.Finished {
				continue
			}

			if toolCallDelta.Function.Arguments != "" {
				toolCall.Args += toolCallDelta.Function.Arguments

				err := dataStream.Write(ToolCallDeltaStreamPart{
					ToolCallID:    toolCall.ID,
					ArgsTextDelta: toolCallDelta.Function.Arguments,
				})
				if err != nil {
					return nil, err
				}
			}

			var args map[string]any
			err := json.Unmarshal([]byte(toolCall.Args), &args)
			// The tool call is finished!
			if err == nil {
				err = dataStream.Write(ToolCallStreamPart{
					ToolCallID: toolCall.ID,
					ToolName:   toolCall.Name,
					Args:       args,
				})
				if err != nil {
					return nil, err
				}

				var result any = "No tool call handler provided"
				if opts.HandleToolCall != nil {
					result = opts.HandleToolCall(ToolCall{
						ID:   toolCall.ID,
						Name: toolCall.Name,
						Args: args,
					})
				}
				err = dataStream.Write(ToolResultStreamPart{
					ToolCallID: toolCall.ID,
					Result:     result,
				})
				if err != nil {
					return nil, err
				}

				jsonResult, err := json.Marshal(result)
				if err != nil {
					return nil, err
				}
				msgs = append(msgs, openai.ToolMessage(string(jsonResult), toolCall.ID))
			}
		}

		if choice.FinishReason != "" {
			err := finish(&chunk)
			if err != nil {
				return nil, err
			}
		}
	}

	finishReason := FinishReasonUnknown
	if lastChunk != nil {
		choice := lastChunk.Choices[0]
		var (
			promptTokens     *int64
			completionTokens *int64
		)
		switch choice.FinishReason {
		case "tool_calls":
			finishReason = FinishReasonToolCalls
		default:
			finishReason = FinishReasonStop
		}
		if lastChunk.Usage.JSON.CompletionTokens.IsPresent() {
			completionTokens = &lastChunk.Usage.CompletionTokens
		}
		if lastChunk.Usage.JSON.PromptTokens.IsPresent() {
			promptTokens = &lastChunk.Usage.PromptTokens
		}
		err := dataStream.Write(FinishMessageStreamPart{
			FinishReason: finishReason,
			Usage: Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
			},
		})
		if err != nil {
			return nil, err
		}
	}

	if len(acc.Choices) > 0 {
		// If the message invokes a tool call, the first message must
		// be the message that invoked the tool call, not the tool call itself.
		msgs = append([]openai.ChatCompletionMessageParamUnion{acc.Choices[0].Message.ToParam()}, msgs...)
	}

	return &PipeResponse[[]openai.ChatCompletionMessageParamUnion]{
		Messages:     msgs,
		FinishReason: finishReason,
	}, stream.Err()
}
