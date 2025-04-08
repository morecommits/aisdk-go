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

func ToolsToGoogle(tools []Tool) ([]*genai.Tool, error) {
	googleTools := []*genai.Tool{}
	for _, tool := range tools {
		var schema *genai.Schema
		if tool.Schema.Properties != nil {
			schema = &genai.Schema{
				Type:       genai.TypeObject,
				Properties: make(map[string]*genai.Schema),
				Required:   tool.Schema.Required,
			}

			// Convert properties
			for propName, propSchema := range tool.Schema.Properties {
				// Handle both simple type strings and complex objects
				var propMap map[string]any
				switch ps := propSchema.(type) {
				case map[string]any:
					propMap = ps
				case map[string]string:
					// Convert map[string]string to map[string]any
					propMap = make(map[string]any, len(ps))
					for k, v := range ps {
						propMap[k] = v
					}
				case string:
					// Handle simple type string (convert to a type definition)
					propMap = map[string]any{"type": ps}
				default:
					return nil, fmt.Errorf("property %q has unsupported schema type %T", propName, propSchema)
				}

				// Get the type
				typeStr, ok := propMap["type"].(string)
				if !ok {
					return nil, fmt.Errorf("property %q missing type field", propName)
				}

				prop := &genai.Schema{}
				switch typeStr {
				case "string":
					prop.Type = genai.TypeString
				case "number":
					prop.Type = genai.TypeNumber
				case "integer":
					prop.Type = genai.TypeInteger
				case "boolean":
					prop.Type = genai.TypeBoolean
				case "array":
					prop.Type = genai.TypeArray
				case "object":
					prop.Type = genai.TypeObject
				default:
					return nil, fmt.Errorf("property %q has unsupported type %q", propName, typeStr)
				}

				// Copy over common fields if they exist
				if desc, ok := propMap["description"].(string); ok {
					prop.Description = desc
				}
				if enum, ok := propMap["enum"].([]any); ok {
					prop.Enum = make([]string, len(enum))
					for i, e := range enum {
						prop.Enum[i] = fmt.Sprintf("%v", e)
					}
				}

				schema.Properties[propName] = prop
			}
		}

		googleTools = append(googleTools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  schema,
				},
			},
		})
	}
	return googleTools, nil
}

// MessagesToGoogle converts internal message format to Google's genai.Content slice.
// System messages are ignored.
func MessagesToGoogle(messages []Message) ([]*genai.Content, error) {
	googleContents := []*genai.Content{}

	for _, message := range messages {
		switch message.Role {
		case "user":
			// TODO: Handle multi-modal user content via Parts
			userContent := &genai.Content{
				Role: "user",
				// Ensure we create a slice of *pointers* to Part
				Parts: []*genai.Part{{Text: message.Content}}, // Correctly make a Part and take its address implicitly via literal
			}
			googleContents = append(googleContents, userContent)

		case "assistant": // Corresponds to 'model' role in Google AI
			assistantParts := []*genai.Part{}
			functionCallParts := []*genai.Part{}
			functionResponseParts := []*genai.Part{} // Separate list for function responses

			// First, add text content if it exists
			if message.Content != "" {
				assistantParts = append(assistantParts, &genai.Part{Text: message.Content}) // Take address
			}

			// Iterate through parts to find text, tool calls, and tool results
			for _, part := range message.Parts {
				switch part.Type {
				case PartTypeText:
					// If message.Content was empty but text parts exist, add them.
					// Avoid duplicating if message.Content already added the text.
					if message.Content == "" && part.Text != "" {
						assistantParts = append(assistantParts, &genai.Part{Text: part.Text}) // Take address
					}
				case PartTypeToolInvocation:
					if part.ToolInvocation == nil {
						return nil, fmt.Errorf("assistant message part has type tool-invocation but nil ToolInvocation field (ID: %s)", message.ID)
					}
					if part.ToolInvocation.State == ToolInvocationStateCall || part.ToolInvocation.State == ToolInvocationStatePartialCall {
						argsMap, ok := part.ToolInvocation.Args.(map[string]any)
						if !ok && part.ToolInvocation.Args != nil { // Allow nil args
							return nil, fmt.Errorf("tool call args for %s are not map[string]any: %T", part.ToolInvocation.ToolName, part.ToolInvocation.Args)
						}
						fc := genai.FunctionCall{
							Name: part.ToolInvocation.ToolName,
							Args: argsMap,
						}
						// Function calls belong in the 'model' role content
						functionCallParts = append(functionCallParts, &genai.Part{FunctionCall: &fc}) // Take address
					} else if part.ToolInvocation.State == ToolInvocationStateResult {
						// Tool results require a separate 'function' role content block later
						var resultMap map[string]any
						switch v := part.ToolInvocation.Result.(type) {
						case string:
							// Attempt to unmarshal if it's a JSON string
							err := json.Unmarshal([]byte(v), &resultMap)
							if err != nil {
								// If not valid JSON, treat it as a plain string result
								resultMap = map[string]any{"result": v}
							}
						case nil:
							resultMap = make(map[string]any) // Empty map for nil result
						default:
							// Attempt to marshal and unmarshal to get map[string]any
							resultBytes, err := json.Marshal(v)
							if err != nil {
								return nil, fmt.Errorf("failed to marshal tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
							}
							err = json.Unmarshal(resultBytes, &resultMap)
							if err != nil {
								// Fallback: wrap non-JSON-object result in a map
								resultMap = map[string]any{"result": v}
							}
						}

						fr := genai.FunctionResponse{
							Name:     part.ToolInvocation.ToolName,
							Response: resultMap,
						}
						functionResponseParts = append(functionResponseParts, &genai.Part{FunctionResponse: &fr}) // Take address
					}
				}
			}

			// Append model content (text + function calls)
			if len(assistantParts) > 0 || len(functionCallParts) > 0 {
				modelContent := &genai.Content{
					Role: "model",
					// Combine the slices of pointers
					Parts: append(assistantParts, functionCallParts...),
				}
				googleContents = append(googleContents, modelContent)
			}

			// Append function responses if any
			if len(functionResponseParts) > 0 {
				// Google expects function responses in a separate 'function' role message
				functionContent := &genai.Content{
					Role:  "function",            // Correct role for function responses
					Parts: functionResponseParts, // Already a slice of pointers
				}
				googleContents = append(googleContents, functionContent)
			}

		case "system":
			// System messages are ignored for Google's main message history.
			// They are handled separately via SystemInstruction.

		case "tool":
			// This case handles messages that *only* contain tool results.
			// Combine with the logic in 'assistant' case as Google expects
			// function responses in a separate 'function' role message
			// following the 'model' message that contained the call.
			functionResponseParts := []*genai.Part{}
			for _, part := range message.Parts {
				if part.Type == PartTypeToolInvocation && part.ToolInvocation != nil && part.ToolInvocation.State == ToolInvocationStateResult {
					var resultMap map[string]any
					switch v := part.ToolInvocation.Result.(type) {
					case string:
						err := json.Unmarshal([]byte(v), &resultMap)
						if err != nil {
							resultMap = map[string]any{"result": v}
						}
					case nil:
						resultMap = make(map[string]any)
					default:
						resultBytes, err := json.Marshal(v)
						if err != nil {
							return nil, fmt.Errorf("failed to marshal tool result for call %s: %w", part.ToolInvocation.ToolCallID, err)
						}
						err = json.Unmarshal(resultBytes, &resultMap)
						if err != nil {
							resultMap = map[string]any{"result": v}
						}
					}
					fr := genai.FunctionResponse{
						Name:     part.ToolInvocation.ToolName,
						Response: resultMap,
					}
					functionResponseParts = append(functionResponseParts, &genai.Part{FunctionResponse: &fr}) // Take address
				}
			}
			if len(functionResponseParts) > 0 {
				functionContent := &genai.Content{
					Role:  "function",
					Parts: functionResponseParts, // Already a slice of pointers
				}
				googleContents = append(googleContents, functionContent)
			} else {
				// If a 'tool' role message has no ToolResult parts, it's an error or unexpected.
				return nil, fmt.Errorf("tool message found without ToolResult parts (ID: %s)", message.ID)
			}

		default:
			return nil, fmt.Errorf("unsupported message role encountered: %s", message.Role)
		}
	}

	return googleContents, nil
}

// GoogleToDataStream pipes a Google AI stream to a DataStream.
func GoogleToDataStream(stream iter.Seq2[*genai.GenerateContentResponse, error]) DataStream {
	return func(yield func(DataStreamPart, error) bool) {
		accumulatedContents := []*genai.Content{}
		finalReason := FinishReasonUnknown
		isNewMessageSegment := true
		var lastResp *genai.GenerateContentResponse

		for resp, err := range stream {
			if err != nil {
				yield(nil, err)
				return
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
				if !yield(StartStepStreamPart{
					MessageID: messageID,
				}, nil) {
					return
				}
				isNewMessageSegment = false
			}

			accumulatedContents = append(accumulatedContents, content)

			for _, part := range content.Parts {
				if part.FunctionCall != nil {
					fc := part.FunctionCall
					// Google's stream often sends the full FunctionCall in one part.
					// We translate this into start and delta parts for the generic stream.
					toolCallID := uuid.New().String()

					// Emit start part ('b')
					if !yield(ToolCallStartStreamPart{
						ToolCallID: toolCallID,
						ToolName:   fc.Name,
					}, nil) {
						return
					}

					// Convert args to JSON string and emit as delta part ('c')
					// WithToolCalling will accumulate this (potentially single) delta.
					argsJSON, err := json.Marshal(fc.Args)
					if err == nil {
						if !yield(ToolCallDeltaStreamPart{
							ToolCallID:    toolCallID,
							ArgsTextDelta: string(argsJSON),
						}, nil) {
							return
						}
						if !yield(ErrorStreamPart{Content: fmt.Sprintf("failed to marshal function call args for %s: %s", fc.Name, err)}, nil) {
							return
						}
						continue
					}

					finalReason = FinishReasonToolCalls
				} else if part.Text != "" {
					text := part.Text
					if part.Thought {
						if !yield(ReasoningStreamPart{Content: text}, nil) {
							return
						}
					} else {
						if !yield(TextStreamPart{Content: text}, nil) {
							return
						}
					}
				} // Add handling for other part types (e.g., FunctionResponse) if necessary
			}
		}

		// Determine the final reason only *after* the loop completes.
		var actualFinalReason FinishReason
		var finalUsage Usage
		var finalCand *genai.Candidate

		if lastResp != nil && len(lastResp.Candidates) > 0 {
			finalCand = lastResp.Candidates[0]
		}

		// Use the detected tool call reason if present, otherwise determine from candidate.
		if finalReason == FinishReasonToolCalls {
			actualFinalReason = FinishReasonToolCalls
		} else if finalCand != nil {
			switch finalCand.FinishReason {
			case genai.FinishReasonStop:
				actualFinalReason = FinishReasonStop
			case genai.FinishReasonMaxTokens:
				actualFinalReason = FinishReasonLength
			case genai.FinishReasonSafety:
				actualFinalReason = FinishReasonContentFilter
			case genai.FinishReasonRecitation:
				actualFinalReason = FinishReasonContentFilter // Treat recitation as content filter
			case genai.FinishReasonUnspecified:
				actualFinalReason = FinishReasonUnknown
			default:
				actualFinalReason = FinishReasonOther
			}
		} else {
			// If no candidate and no tool call detected, assume stop or error?
			// Let's default to Stop, assuming the stream ended normally without specific reason.
			actualFinalReason = FinishReasonStop
		}

		// Extract final usage data if available
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

		// Send final finish step part
		if !yield(FinishStepStreamPart{
			FinishReason: actualFinalReason,
			Usage:        finalUsage,
			IsContinued:  false, // This is the final step
		}, nil) {
			return // Stop if yield fails
		}

		// Send final finish message part
		yield(FinishMessageStreamPart{
			FinishReason: actualFinalReason,
			Usage:        finalUsage,
		}, nil) // Ignore yield result here as we're at the very end
	}
}
