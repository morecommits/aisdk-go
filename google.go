package aisdk

import (
	"encoding/json"
	"fmt"
	"iter"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// GoogleStreamIterator defines the interface for iterating over Google AI stream responses.
// This allows for mocking in tests.
type GoogleStreamIterator interface {
	Next() (*genai.GenerateContentResponse, error)
}

// MessagesToGoogle converts internal message format to Google's genai.Content slice.
// System messages are ignored.
func MessagesToGoogle(messages []Message) ([]*genai.Content, error) {
	googleContents := []*genai.Content{}

	for _, message := range messages {
		switch message.Role {
		case "user":
			userContent := &genai.Content{
				Role:  "user",
				Parts: []*genai.Part{{Text: message.Content}},
			}
			googleContents = append(googleContents, userContent)

		case "assistant":
			assistantParts := []*genai.Part{}
			if message.Content != "" {
				assistantParts = append(assistantParts, &genai.Part{Text: message.Content})
			}

			functionResponseParts := []*genai.Part{}

			for _, ti := range message.ToolInvocations {
				switch ti.State {
				case ToolInvocationStateCall, ToolInvocationStatePartialCall:
					argsMap, ok := ti.Args.(map[string]any)
					if !ok && ti.Args != nil {
						return nil, fmt.Errorf("tool invocation args for %s are not map[string]any: %T", ti.ToolName, ti.Args)
					}
					fc := genai.FunctionCall{
						Name: ti.ToolName,
						Args: argsMap,
					}
					assistantParts = append(assistantParts, &genai.Part{FunctionCall: &fc})
				case ToolInvocationStateResult:
					var resultMap map[string]any
					switch v := ti.Result.(type) {
					case string:
						err := json.Unmarshal([]byte(v), &resultMap)
						if err != nil {
							return nil, fmt.Errorf("tool invocation result for %s was string but not valid JSON object: %w", ti.ToolName, err)
						}
					default:
						resultBytes, err := json.Marshal(v)
						if err != nil {
							return nil, fmt.Errorf("failed to marshal tool invocation result for %s: %w", ti.ToolName, err)
						}
						err = json.Unmarshal(resultBytes, &resultMap)
						if err != nil {
							// Fallback: wrap non-JSON-object result in a map
							resultMap = map[string]any{"result": v}
						}
					}

					if resultMap == nil {
						resultMap = make(map[string]any)
					}

					fr := genai.FunctionResponse{
						Name:     ti.ToolName,
						Response: resultMap,
					}
					functionResponseParts = append(functionResponseParts, &genai.Part{FunctionResponse: &fr})
				}
			}

			if len(assistantParts) > 0 {
				assistantContent := &genai.Content{
					Role:  "model",
					Parts: assistantParts,
				}
				googleContents = append(googleContents, assistantContent)
			}

			if len(functionResponseParts) > 0 {
				functionContent := &genai.Content{
					Role:  "function",
					Parts: functionResponseParts,
				}
				googleContents = append(googleContents, functionContent)
			}

		case "system":
			// System messages are ignored.
			// Google's genai API handles system instructions differently.

		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return googleContents, nil
}

// PipeGoogleToDataStream pipes a Google AI stream to a DataStream.
func PipeGoogleToDataStream(stream iter.Seq2[*genai.GenerateContentResponse, error], dataStream DataStream, opts *PipeOptions) (*PipeResponse[[]*genai.Content], error) {
	accumulatedContents := []*genai.Content{}
	finalReason := FinishReasonUnknown
	isNewMessageSegment := true
	var lastResp *genai.GenerateContentResponse

	for resp, err := range stream {
		if err != nil {
			return nil, fmt.Errorf("stream iteration error: %w", err)
		}

		if resp == nil {
			continue
		}

		lastResp = resp

		if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
			continue
		}
		cand := resp.Candidates[0]
		content := cand.Content

		if isNewMessageSegment {
			messageID := uuid.New().String()
			err := dataStream.Write(StartStepStreamPart{
				MessageID: messageID,
			})
			if err != nil {
				return nil, fmt.Errorf("writing start step part: %w", err)
			}
			isNewMessageSegment = false
		}

		accumulatedContents = append(accumulatedContents, content)

		for _, part := range content.Parts {
			if part.FunctionCall != nil {
				fc := part.FunctionCall
				toolCallID := uuid.New().String()
				err := dataStream.Write(ToolCallStartStreamPart{
					ToolCallID: toolCallID,
					ToolName:   fc.Name,
				})
				if err != nil {
					return nil, fmt.Errorf("writing tool start part: %w", err)
				}
				err = dataStream.Write(ToolCallStreamPart{
					ToolCallID: toolCallID,
					ToolName:   fc.Name,
					Args:       fc.Args,
				})
				if err != nil {
					return nil, fmt.Errorf("writing tool call part: %w", err)
				}

				var result any
				if opts != nil && opts.HandleToolCall != nil {
					result = opts.HandleToolCall(ToolCall{
						ID:   toolCallID,
						Name: fc.Name,
						Args: fc.Args,
					})
				} else {
					result = map[string]string{"error": "No tool call handler provided"}
				}

				var resultMap map[string]any
				resultBytes, err := json.Marshal(result)
				if err != nil {
					// Handle cases where the handler result isn't JSON-serializable
					resultMap = map[string]any{"error": fmt.Sprintf("failed to marshal tool result: %v", err)}
					resultBytes, _ = json.Marshal(resultMap)
				} else {
					err := json.Unmarshal(resultBytes, &resultMap)
					if err != nil {
						// Wrap non-map results for FunctionResponse
						resultMap = map[string]any{"result": result}
					}
				}
				if resultMap == nil {
					resultMap = make(map[string]any)
				}

				fr := genai.FunctionResponse{
					Name:     fc.Name,
					Response: resultMap,
				}
				funcResponseContent := &genai.Content{
					Role:  "function",
					Parts: []*genai.Part{{FunctionResponse: &fr}},
				}
				accumulatedContents = append(accumulatedContents, funcResponseContent)

				err = dataStream.Write(ToolResultStreamPart{
					ToolCallID: toolCallID,
					Result:     string(resultBytes),
				})
				if err != nil {
					return nil, fmt.Errorf("writing tool result part: %w", err)
				}

				err = dataStream.Write(FinishStepStreamPart{
					FinishReason: FinishReasonToolCalls,
					Usage:        Usage{},
					IsContinued:  false,
				})
				if err != nil {
					return nil, fmt.Errorf("writing tool step finish part: %w", err)
				}

				isNewMessageSegment = true
				finalReason = FinishReasonToolCalls

			} else if part.Text != "" {
				text := part.Text

				if part.Thought {
					err := dataStream.Write(ReasoningStreamPart{Content: text})
					if err != nil {
						return nil, fmt.Errorf("writing reasoning part: %w", err)
					}
				} else {
					err := dataStream.Write(TextStreamPart{Content: text})
					if err != nil {
						return nil, fmt.Errorf("writing text part: %w", err)
					}
				}
			} else {
				// Skip other part types like FunctionResponse or FileData in the stream
			}
		}
	}

	var finalUsage Usage
	var finalCand *genai.Candidate

	if lastResp != nil && len(lastResp.Candidates) > 0 {
		finalCand = lastResp.Candidates[0]
	}

	if finalReason == FinishReasonUnknown {
		if finalCand != nil {
			switch finalCand.FinishReason {
			case genai.FinishReasonStop:
				finalReason = FinishReasonStop
			case genai.FinishReasonMaxTokens:
				finalReason = FinishReasonLength
			case genai.FinishReasonSafety:
				finalReason = FinishReasonContentFilter
			case genai.FinishReasonRecitation:
				finalReason = FinishReasonContentFilter
			case genai.FinishReasonUnspecified:
				finalReason = FinishReasonUnknown
			default:
				finalReason = FinishReasonOther
			}
		} else {
			finalReason = FinishReasonStop
		}
	}

	if lastResp != nil && lastResp.UsageMetadata != nil {
		if lastResp.UsageMetadata.PromptTokenCount != nil {
			promptTokens := int64(*lastResp.UsageMetadata.PromptTokenCount)
			finalUsage.PromptTokens = &promptTokens
		}
		if lastResp.UsageMetadata.CandidatesTokenCount != nil {
			completionTokens := int64(*lastResp.UsageMetadata.CandidatesTokenCount)
			finalUsage.CompletionTokens = &completionTokens
		}
	}

	err := dataStream.Write(FinishMessageStreamPart{
		FinishReason: finalReason,
		Usage:        finalUsage,
	})
	if err != nil {
		return nil, fmt.Errorf("writing final stream part: %w", err)
	}

	return &PipeResponse[[]*genai.Content]{
		Messages:     accumulatedContents,
		FinishReason: finalReason,
	}, nil
}

// GoogleToMessages converts Google's []*genai.Content format back to the internal []Message format.
func GoogleToMessages(googleContents []*genai.Content) ([]Message, error) {
	messages := []Message{}
	var lastAssistantMessage *Message

	for _, content := range googleContents {
		var currentMessage Message
		isFunctionResponseMessage := true

		switch content.Role {
		case "user":
			currentMessage.Role = "user"
		case "model":
			currentMessage.Role = "assistant"
		case "function":
			currentMessage.Role = "user"
		case "":
			isLikelyResponseContainer := false
			if len(content.Parts) > 0 {
				for _, part := range content.Parts {
					if part.FunctionResponse != nil {
						isLikelyResponseContainer = true
						break
					}
				}
			}
			if isLikelyResponseContainer {
				currentMessage.Role = "user"
			} else {
				currentMessage.Role = "user"
			}

		default:
			return nil, fmt.Errorf("unsupported Google role: %q", content.Role)
		}

		toolInvocations := []ToolInvocation{}

		for _, part := range content.Parts {
			if part.Text != "" {
				currentMessage.Content += part.Text
				isFunctionResponseMessage = false
			} else if fc := part.FunctionCall; fc != nil {
				isFunctionResponseMessage = false
				toolInvocations = append(toolInvocations, ToolInvocation{
					ToolCallID: "",
					ToolName:   fc.Name,
					Args:       fc.Args,
					State:      ToolInvocationStateCall,
				})
			} else if fr := part.FunctionResponse; fr != nil {
				if lastAssistantMessage == nil {
					return nil, fmt.Errorf("received function response part (role %q, tool %q) without a preceding assistant ('model') message", content.Role, fr.Name)
				}

				resultMap := fr.Response

				found := false
				for i := range lastAssistantMessage.ToolInvocations {
					inv := &lastAssistantMessage.ToolInvocations[i]
					if inv.ToolName == fr.Name && inv.State == ToolInvocationStateCall {
						inv.Result = resultMap
						inv.State = ToolInvocationStateResult
						found = true
						break
					}
				}
				if !found {
					return nil, fmt.Errorf("received function response for tool %q, but could not find a matching pending call in the previous assistant message", fr.Name)
				}
			} else {
				isFunctionResponseMessage = false
			}
		}

		currentMessage.ToolInvocations = toolInvocations

		if !isFunctionResponseMessage {
			messages = append(messages, currentMessage)
			if currentMessage.Role == "assistant" {
				lastAssistantMessage = &messages[len(messages)-1]
			} else {
				lastAssistantMessage = nil
			}
		}

	}

	return messages, nil
}
