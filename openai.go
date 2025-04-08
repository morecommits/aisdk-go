package aisdk

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/packages/ssestream"
)

// ToolsToOpenAI converts the tool format to OpenAI's API format.
func ToolsToOpenAI(tools []Tool) []openai.ChatCompletionToolParam {
	openaiTools := []openai.ChatCompletionToolParam{}
	for _, tool := range tools {
		var schemaParams map[string]any
		if tool.Schema.Properties != nil {
			schemaParams = map[string]any{
				"type":       "object",
				"properties": tool.Schema.Properties,
				"required":   tool.Schema.Required,
			}
		}
		openaiTools = append(openaiTools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Name,
				Description: param.NewOpt[string](tool.Description),
				Parameters:  schemaParams,
			},
		})
	}
	return openaiTools
}

// MessagesToOpenAI converts internal message format to OpenAI's API format.
func MessagesToOpenAI(messages []Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	openaiMessages := []openai.ChatCompletionMessageParamUnion{}

	for _, message := range messages {
		switch message.Role {
		case "user":
			// Handle simple text content
			if len(message.Parts) == 1 && message.Parts[0].Type == PartTypeText {
				contentUnion := openai.ChatCompletionUserMessageParamContentUnion{}
				contentUnion.OfString = param.NewOpt(message.Parts[0].Text)
				openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: contentUnion,
					},
				})
				continue
			}

			// For now, just use the message content for multi-part messages
			// TODO: Support multi-modal content (images, files, etc.)
			contentUnion := openai.ChatCompletionUserMessageParamContentUnion{}
			contentUnion.OfString = param.NewOpt(message.Content)
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfUser: &openai.ChatCompletionUserMessageParam{
					Content: contentUnion,
				},
			})

		case "system":
			openaiMessages = append(openaiMessages, openai.SystemMessage(message.Content))

		case "assistant":
			assistantMsg := openai.ChatCompletionAssistantMessageParam{}
			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0)
			toolResults := make(map[string]json.RawMessage) // Store results temporarily
			hasToolCalls := false
			textContent := ""

			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					textContent += part.Text
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, fmt.Errorf("assistant message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}

					// Always create the tool call structure for the assistant message
					argsJSON, err := json.Marshal(part.ToolInvocation.Args)
					if err != nil {
						return nil, fmt.Errorf("marshalling tool input for call %s: %w", part.ToolInvocation.ToolCallID, err)
					}
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: part.ToolInvocation.ToolCallID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      part.ToolInvocation.ToolName,
							Arguments: string(argsJSON),
						},
					})
					hasToolCalls = true

					// If the state includes a result, store it for a separate ToolMessage
					if part.ToolInvocation.State == ToolInvocationStateResult {
						resultBytes, err := json.Marshal(part.ToolInvocation.Result)
						if err != nil {
							return nil, fmt.Errorf("marshalling tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
						}
						toolResults[part.ToolInvocation.ToolCallID] = resultBytes
					}
				}
			}

			contentUnion := openai.ChatCompletionAssistantMessageParamContentUnion{}
			contentUnion.OfString = param.NewOpt(textContent)
			assistantMsg.Content = contentUnion

			if hasToolCalls {
				assistantMsg.ToolCalls = toolCalls
			}
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})

			// Append ToolMessages for any results we found
			for toolCallID, resultBytes := range toolResults {
				openaiMessages = append(openaiMessages, openai.ToolMessage(string(resultBytes), toolCallID))
			}

		case "tool":
			// Convert tool messages to OpenAI format
			for _, part := range message.Parts {
				if part.Type == PartTypeToolInvocation && part.ToolInvocation != nil && part.ToolInvocation.State == ToolInvocationStateResult {
					resultBytes, err := json.Marshal(part.ToolInvocation.Result)
					if err != nil {
						return nil, fmt.Errorf("marshalling tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
					}
					openaiMessages = append(openaiMessages, openai.ToolMessage(string(resultBytes), part.ToolInvocation.ToolCallID))
				}
			}

		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return openaiMessages, nil
}

// OpenAIToDataStream pipes an OpenAI stream to a DataStream.
func OpenAIToDataStream(stream *ssestream.Stream[openai.ChatCompletionChunk]) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		var lastChunk *openai.ChatCompletionChunk
		var currentToolCallID string

		if stream.Err() != nil {
			if !yield(ErrorStreamPart{Content: stream.Err().Error()}, nil) {
				return
			}
		}

		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			if len(chunk.Choices) == 0 {
				break
			}
			choice := chunk.Choices[0]

			if choice.Delta.Content != "" {
				// Yield a Part object instead of TextStreamPart
				if !yield(TextStreamPart{Content: choice.Delta.Content}, nil) {
					return
				}
			}

			for _, toolCallDelta := range choice.Delta.ToolCalls {
				// The tool call ID is only present in the first delta.
				if toolCallDelta.ID != "" {
					currentToolCallID = toolCallDelta.ID // Update current ID when starting new tool call
					if !yield(ToolCallStartStreamPart{
						ToolCallID: currentToolCallID,
						ToolName:   toolCallDelta.Function.Name,
					}, nil) {
						return
					}
				}

				// Only emit delta parts if we have arguments
				if toolCallDelta.Function.Arguments != "" {
					if currentToolCallID == "" {
						if !yield(nil, fmt.Errorf("received tool call delta with empty ID and no current tool call")) {
							return
						}
						continue
					}
					if !yield(ToolCallDeltaStreamPart{
						ToolCallID:    currentToolCallID,
						ArgsTextDelta: toolCallDelta.Function.Arguments,
					}, nil) {
						return
					}
				}
			}
		}

		var finishReason FinishReason
		var promptTokens, completionTokens *int64

		if lastChunk != nil && len(lastChunk.Choices) > 0 {
			choice := lastChunk.Choices[0]

			switch choice.FinishReason {
			case "tool_calls":
				finishReason = FinishReasonToolCalls
			default:
				finishReason = FinishReasonStop
			}

			if lastChunk.Usage.JSON.CompletionTokens.IsPresent() {
				tokens := int64(lastChunk.Usage.CompletionTokens)
				completionTokens = &tokens
			}
			if lastChunk.Usage.JSON.PromptTokens.IsPresent() {
				tokens := int64(lastChunk.Usage.PromptTokens)
				promptTokens = &tokens
			}
		}

		yield(FinishMessageStreamPart{
			FinishReason: finishReason,
			Usage: Usage{
				PromptTokens:     promptTokens,
				CompletionTokens: completionTokens,
			},
		}, nil)
	}
}
