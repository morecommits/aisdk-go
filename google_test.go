package aisdk_test

import (
	"iter"
	"testing"

	"github.com/kylecarbs/aisdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

// Helper function to create genai.Content objects for testing expected values
func userContent(text string) *genai.Content {
	return &genai.Content{Role: "user", Parts: []*genai.Part{{Text: text}}}
}

func modelContent(parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: "model", Parts: parts}
}

func functionContent(parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: "function", Parts: parts}
}

func TestMessagesToGoogle_Basic(t *testing.T) {
	messages := []aisdk.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
	}

	// Expect []*genai.Content with updated structure
	expected := []*genai.Content{
		userContent("Hello"),
		modelContent(textPart("Hi there!")),
	}

	googleContents, err := aisdk.MessagesToGoogle(messages)
	assert.NoError(t, err)
	assert.Equal(t, expected, googleContents)
}

func TestMessagesToGoogle_SystemIgnored(t *testing.T) {
	messages := []aisdk.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	// Expect []*genai.Content
	expected := []*genai.Content{
		userContent("Hello"), // System message is ignored
	}

	googleContents, err := aisdk.MessagesToGoogle(messages)
	assert.NoError(t, err)
	assert.Equal(t, expected, googleContents)
}

func TestMessagesToGoogle_ToolCall(t *testing.T) {
	messages := []aisdk.Message{
		{Role: "user", Content: "What's the weather?"},
		{
			Role: "assistant",
			ToolInvocations: []aisdk.ToolInvocation{
				{
					State:      aisdk.ToolInvocationStateCall,
					ToolCallID: "call_123",
					ToolName:   "get_weather",
					Args: map[string]any{
						"location": "London",
					},
				},
			},
		},
	}

	// Expect model Content with FunctionCall part
	expected := []*genai.Content{
		userContent("What's the weather?"),
		modelContent(functionCallPart("get_weather", map[string]any{"location": "London"})),
	}

	googleContents, err := aisdk.MessagesToGoogle(messages)
	assert.NoError(t, err)
	assert.Equal(t, expected, googleContents)
}

func TestMessagesToGoogle_ToolResult(t *testing.T) {
	messages := []aisdk.Message{
		{Role: "user", Content: "What's the weather?"},
		{
			Role: "assistant", // Assistant decided to call the tool
			ToolInvocations: []aisdk.ToolInvocation{
				{
					State:      aisdk.ToolInvocationStateResult, // Provide result IN THE SAME assistant message
					ToolCallID: "call_123",                      // ID matches the call
					ToolName:   "get_weather",
					// Args are not strictly needed by MessagesToGoogle for FunctionResponse, but good practice
					Args: map[string]any{
						"location": "London",
					},
					Result: map[string]any{ // Result provided as a map, not string
						"weather": "sunny",
					},
				},
			},
		},
	}

	// Expect user msg, empty assistant msg (no text/call), function msg (with result)
	expected := []*genai.Content{
		userContent("What's the weather?"),
		// Assistant message is empty here.
		functionContent(functionResponsePart("get_weather", map[string]any{"weather": "sunny"})),
	}

	googleContents, err := aisdk.MessagesToGoogle(messages)
	require.NoError(t, err) // Use require to stop test on error
	assert.Equal(t, expected, googleContents)
}

func TestMessagesToGoogle_TextAndToolCall(t *testing.T) {
	messages := []aisdk.Message{
		{Role: "user", Content: "Check weather in London"},
		{
			Role:    "assistant",
			Content: "Okay, checking...", // Assistant provides text *and* tool call
			ToolInvocations: []aisdk.ToolInvocation{
				{
					State:      aisdk.ToolInvocationStateCall,
					ToolCallID: "call_weather",
					ToolName:   "get_weather",
					Args:       map[string]any{"location": "London"},
				},
			},
		},
	}

	// Expect user msg, assistant msg with BOTH text and FunctionCall parts
	expected := []*genai.Content{
		userContent("Check weather in London"),
		modelContent(
			textPart("Okay, checking..."),
			functionCallPart("get_weather", map[string]any{"location": "London"}),
		),
	}

	googleContents, err := aisdk.MessagesToGoogle(messages)
	assert.NoError(t, err)
	assert.Equal(t, expected, googleContents)
}

// --- PipeGoogleToDataStream Tests ---

// mockStreamSequence creates an iter.Seq2 for testing PipeGoogleToDataStream.
func mockStreamSequence(responses []*genai.GenerateContentResponse, finalErr error) iter.Seq2[*genai.GenerateContentResponse, error] {
	return func(yield func(*genai.GenerateContentResponse, error) bool) {
		for _, resp := range responses {
			if !yield(resp, nil) {
				return // Stop iteration if yield returns false
			}
		}
		// Yield the final error, if any. The response part is nil here.
		if finalErr != nil {
			yield(nil, finalErr)
		}
	}
}

// Helper to create a Text Part pointer
func textPart(text string) *genai.Part {
	return &genai.Part{Text: text}
}

// Helper to create a FunctionCall Part pointer
func functionCallPart(name string, args map[string]any) *genai.Part {
	return &genai.Part{FunctionCall: &genai.FunctionCall{Name: name, Args: args}}
}

// Helper to create a FunctionResponse Part pointer
func functionResponsePart(name string, response map[string]any) *genai.Part {
	return &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: name, Response: response}}
}

// Helper to get pointer to int32 for UsageMetadata
func int32Ptr(i int32) *int32 {
	return &i
}

func TestPipeGoogleToDataStream_Text(t *testing.T) {
	// Responses updated to use new Part structure
	mockResponses := []*genai.GenerateContentResponse{
		{Candidates: []*genai.Candidate{{Content: modelContent(textPart("Hello "))}}},
		{Candidates: []*genai.Candidate{{Content: modelContent(textPart("World"))}}},
		// Separate response for finish reason and usage
		{
			Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonStop}},
			// Use the correct type: genai.GenerateContentResponseUsageMetadata
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: int32Ptr(10), CandidatesTokenCount: int32Ptr(5)},
		},
	}
	// Use the new mock sequence generator, passing nil for final error
	mockStream := mockStreamSequence(mockResponses, nil)

	var capturedParts []aisdk.DataStreamPart
	dataStreamFunc := aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
		capturedParts = append(capturedParts, parts...)
		return nil
	})

	opts := &aisdk.PipeOptions{}

	pipeResp, err := aisdk.PipeGoogleToDataStream(mockStream, dataStreamFunc, opts)

	require.NoError(t, err)
	require.NotNil(t, pipeResp)
	assert.Equal(t, aisdk.FinishReasonStop, pipeResp.FinishReason)

	// Expected stream parts
	expectedParts := []aisdk.DataStreamPart{
		aisdk.StartStepStreamPart{MessageID: "ignored"}, // ID ignored
		aisdk.TextStreamPart{Content: "Hello "},
		aisdk.TextStreamPart{Content: "World"},
		aisdk.FinishMessageStreamPart{
			FinishReason: aisdk.FinishReasonStop,
			Usage: aisdk.Usage{
				PromptTokens:     intPtr(10), // Our Usage struct still uses *int64
				CompletionTokens: intPtr(5),
			},
		},
	}
	assertStreamPartsEqualIgnoringIDs(t, expectedParts, capturedParts)

	// Check accumulated messages (should be []*genai.Content with new structure)
	expectedAccMessages := []*genai.Content{
		modelContent(textPart("Hello ")),
		modelContent(textPart("World")),
		// The final response with only finish reason/usage doesn't add content
	}
	assert.Equal(t, expectedAccMessages, pipeResp.Messages)
}

func TestPipeGoogleToDataStream_ToolCall(t *testing.T) {
	mockResponses := []*genai.GenerateContentResponse{
		// Model responds with a FunctionCall Part
		{Candidates: []*genai.Candidate{{
			Content: modelContent(functionCallPart("get_weather", map[string]any{"location": "Tokyo"})),
		}}},
		// Stream ends after issuing the tool call.
		// Use Stop as the reason and provide usage.
		{
			Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonStop}},
			// Use the correct type: genai.GenerateContentResponseUsageMetadata
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: int32Ptr(15), CandidatesTokenCount: int32Ptr(10)},
		},
	}
	// Use the new mock sequence generator
	mockStream := mockStreamSequence(mockResponses, nil)

	var capturedParts []aisdk.DataStreamPart
	dataStreamFunc := aisdk.DataStreamFunc(func(parts ...aisdk.DataStreamPart) error {
		capturedParts = append(capturedParts, parts...)
		return nil
	})

	// Handle the tool call
	toolCallHandled := false
	handleToolCall := func(toolCall aisdk.ToolCall) any {
		assert.Equal(t, "get_weather", toolCall.Name)
		assert.Equal(t, map[string]any{"location": "Tokyo"}, toolCall.Args)
		toolCallHandled = true
		return map[string]any{"temp": 25, "unit": "C"} // Return a map
	}

	opts := &aisdk.PipeOptions{
		HandleToolCall: handleToolCall,
	}

	pipeResp, err := aisdk.PipeGoogleToDataStream(mockStream, dataStreamFunc, opts)

	require.NoError(t, err)
	require.NotNil(t, pipeResp)
	assert.True(t, toolCallHandled, "Tool call handler was not invoked")
	// Expect ToolCalls finish reason because a tool call was processed
	assert.Equal(t, aisdk.FinishReasonToolCalls, pipeResp.FinishReason)

	// Expected stream parts
	expectedParts := []aisdk.DataStreamPart{
		aisdk.StartStepStreamPart{MessageID: "ignored"},
		aisdk.ToolCallStartStreamPart{ToolCallID: "ignored", ToolName: "get_weather"},
		aisdk.ToolCallStreamPart{ToolCallID: "ignored", ToolName: "get_weather", Args: map[string]any{"location": "Tokyo"}},
		aisdk.ToolResultStreamPart{ToolCallID: "ignored", Result: "{\"temp\":25,\"unit\":\"C\"}"},
		aisdk.FinishStepStreamPart{
			FinishReason: aisdk.FinishReasonToolCalls,
			Usage:        aisdk.Usage{},
			IsContinued:  false,
		},
		aisdk.FinishMessageStreamPart{
			FinishReason: aisdk.FinishReasonToolCalls, // Final reason is ToolCalls
			Usage: aisdk.Usage{
				PromptTokens:     intPtr(15),
				CompletionTokens: intPtr(10),
			},
		},
	}
	assertStreamPartsEqualIgnoringIDs(t, expectedParts, capturedParts)

	// Check accumulated messages (includes FunctionCall AND the FunctionResponse generated by the handler)
	expectedAccMessages := []*genai.Content{
		modelContent(functionCallPart("get_weather", map[string]any{"location": "Tokyo"})),
		// FunctionResponse part is added by PipeGoogleToDataStream after handling
		functionContent(functionResponsePart("get_weather", map[string]any{"temp": 25.0, "unit": "C"})),
	}
	assert.Equal(t, expectedAccMessages, pipeResp.Messages)
}

// Helper to get pointer to int64 (still needed for Usage struct)
func intPtr(i int64) *int64 {
	return &i
}

// Custom assertion to compare stream parts while ignoring IDs
func assertStreamPartsEqualIgnoringIDs(t *testing.T, expected, actual []aisdk.DataStreamPart) {
	require.Equal(t, len(expected), len(actual), "Number of stream parts mismatch")

	for i := range expected {
		exp := expected[i]
		act := actual[i]

		// Zero out ID fields for comparison
		switch p := exp.(type) {
		case aisdk.StartStepStreamPart:
			p.MessageID = ""
			exp = p
			if a, ok := act.(aisdk.StartStepStreamPart); ok {
				a.MessageID = ""
				act = a
			}
		case aisdk.ToolCallStartStreamPart:
			p.ToolCallID = ""
			exp = p
			if a, ok := act.(aisdk.ToolCallStartStreamPart); ok {
				a.ToolCallID = ""
				act = a
			}
		case aisdk.ToolCallStreamPart:
			p.ToolCallID = ""
			exp = p
			if a, ok := act.(aisdk.ToolCallStreamPart); ok {
				a.ToolCallID = ""
				act = a
			}
		case aisdk.ToolResultStreamPart:
			p.ToolCallID = ""
			exp = p
			if a, ok := act.(aisdk.ToolResultStreamPart); ok {
				a.ToolCallID = ""
				act = a
			}
		}
		assert.Equal(t, exp, act, "Stream part mismatch at index %d", i)
	}
}

func TestGoogleToMessages_Conversion(t *testing.T) {
	// Use t as the standard testing object

	testCases := []struct {
		name             string
		inputContents    []*genai.Content
		expectedOutput   []aisdk.Message
		expectError      bool
		expectedErrorStr string
	}{
		{
			name: "Simple User Message",
			inputContents: []*genai.Content{
				{
					Role:  "user",
					Parts: []*genai.Part{{Text: "Hello"}},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "user", Content: "Hello", ToolInvocations: []aisdk.ToolInvocation{}},
			},
		},
		{
			name: "Simple Assistant Message",
			inputContents: []*genai.Content{
				{
					Role:  "model",
					Parts: []*genai.Part{{Text: "Hi there"}},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "assistant", Content: "Hi there", ToolInvocations: []aisdk.ToolInvocation{}},
			},
		},
		{
			name: "Assistant Message with Tool Call",
			inputContents: []*genai.Content{
				{
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "SF"},
						}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{
					Role: "assistant",
					// Content is empty when only tool calls are present
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "", ToolName: "get_weather", Args: map[string]interface{}{"location": "SF"}, State: aisdk.ToolInvocationStateCall},
					},
				},
			},
		},
		{
			name: "Assistant Call followed by Function Result",
			inputContents: []*genai.Content{
				{ // Assistant makes the call
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "SF"},
						}},
					},
				},
				{ // Function provides the result
					Role: "function",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							Name:     "get_weather",
							Response: map[string]any{"temp": 72, "unit": "F"},
						}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{ // Only the assistant message remains, modified with the result
					Role: "assistant",
					// Content is empty when only tool calls are present initially
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "", ToolName: "get_weather", Args: map[string]interface{}{"location": "SF"}, State: aisdk.ToolInvocationStateResult, Result: map[string]interface{}{"temp": 72, "unit": "F"}},
					},
				},
			},
		},
		{
			name: "Assistant Call + Text, followed by Function Result",
			inputContents: []*genai.Content{
				{ // Assistant makes the call and says something
					Role: "model",
					Parts: []*genai.Part{
						{Text: "Okay, getting weather..."},
						{FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "London"},
						}},
					},
				},
				{ // Function provides the result
					Role: "function",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							Name:     "get_weather",
							Response: map[string]any{"temp": 15, "unit": "C"},
						}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{ // Assistant message, modified with result
					Role:    "assistant",
					Content: "Okay, getting weather...",
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "", ToolName: "get_weather", Args: map[string]interface{}{"location": "London"}, State: aisdk.ToolInvocationStateResult, Result: map[string]interface{}{"temp": 15, "unit": "C"}},
					},
				},
				// Function result message is consumed
			},
		},
		{
			name: "Function Result without preceding Assistant Call",
			inputContents: []*genai.Content{
				{
					Role: "function",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{
							Name:     "get_weather",
							Response: map[string]any{"temp": 72},
						}},
					},
				},
			},
			expectError:      true,
			expectedErrorStr: "without a preceding assistant ('model') message",
		},
		{
			name: "Function Result with non-matching ToolName",
			inputContents: []*genai.Content{
				{ // Assistant makes a call
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{Name: "get_weather", Args: map[string]any{}}},
					},
				},
				{ // Function provides result for a different name
					Role: "function",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{Name: "get_stock_price", Response: map[string]any{}}},
					},
				},
			},
			expectError:      true,
			expectedErrorStr: "could not find a matching pending call",
		},
		{
			name: "Empty role, likely function response",
			inputContents: []*genai.Content{
				{ // Assistant makes a call
					Role: "model",
					Parts: []*genai.Part{
						{FunctionCall: &genai.FunctionCall{Name: "calculate", Args: map[string]any{}}},
					},
				},
				{ // Function provides result with empty role
					Role: "",
					Parts: []*genai.Part{
						{FunctionResponse: &genai.FunctionResponse{Name: "calculate", Response: map[string]any{"result": 42}}},
					},
				},
			},
			expectedOutput: []aisdk.Message{
				{ // Assistant message, modified with result
					Role: "assistant",
					// Content is empty when only tool calls are present initially
					ToolInvocations: []aisdk.ToolInvocation{
						{ToolCallID: "", ToolName: "calculate", Args: map[string]interface{}{}, State: aisdk.ToolInvocationStateResult, Result: map[string]interface{}{"result": 42}},
					},
				},
			},
		},
		{
			name: "Empty role, not function response (defaults to user)",
			inputContents: []*genai.Content{
				{
					Role:  "",
					Parts: []*genai.Part{{Text: "Something without role"}},
				},
			},
			expectedOutput: []aisdk.Message{
				{Role: "user", Content: "Something without role", ToolInvocations: []aisdk.ToolInvocation{}},
			},
		},
		{
			name:           "Empty Input",
			inputContents:  []*genai.Content{},
			expectedOutput: []aisdk.Message{},
		},
		{
			name: "Unsupported Role",
			inputContents: []*genai.Content{
				{
					Role:  "system", // System role is unsupported by GoogleToMessages
					Parts: []*genai.Part{{Text: "System info"}},
				},
			},
			expectError:      true,
			expectedErrorStr: "unsupported Google role: \"system\"",
		},
	}

	for _, tc := range testCases {
		tc := tc // Capture range variable
		t.Run(tc.name, func(t *testing.T) {
			r := require.New(t) // Create require instance per subtest
			t.Parallel()
			output, err := aisdk.GoogleToMessages(tc.inputContents)
			if tc.expectError {
				r.Error(err) // Use require instance 'r'
				if tc.expectedErrorStr != "" {
					r.Contains(err.Error(), tc.expectedErrorStr) // Use require instance 'r'
				}
			} else {
				r.NoError(err) // Use require instance 'r'
				// Use require.Equal for simpler slice comparison
				r.Equal(tc.expectedOutput, output) // Use require instance 'r'
			}
		})
	}
}
