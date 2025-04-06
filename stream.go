package aisdk

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type DataStream interface {
	Write(parts ...DataStreamPart) error
}

// DataStreamFunc is a function that implements the DataStream interface.
// It's just a convenience type to make it easier to implement the DataStream interface.
type DataStreamFunc func(parts ...DataStreamPart) error

// Write implements the DataStream interface.
func (f DataStreamFunc) Write(parts ...DataStreamPart) error {
	return f(parts...)
}

// NewDataStream accepts an http.ResponseWriter and makes it a DataStream.
func NewDataStream(w http.ResponseWriter) DataStream {
	f, ok := w.(http.Flusher)
	if !ok {
		panic("expected http.ResponseWriter to be an http.Flusher")
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Vercel-AI-Data-Stream", "v1")
	w.WriteHeader(http.StatusOK)

	return DataStreamFunc(func(parts ...DataStreamPart) error {
		for _, part := range parts {
			formatted, err := part.Format()
			if err != nil {
				return err
			}
			_, err = fmt.Fprint(w, formatted)
			if err != nil {
				return err
			}
			f.Flush()
		}
		return nil
	})
}

// DataStreamPart represents a part of the Vercel AI SDK data stream.
type DataStreamPart interface {
	Format() (string, error)
	TypeID() byte
}

// TextStreamPart corresponds to TYPE_ID '0'.
type TextStreamPart struct {
	Content string
}

func (p TextStreamPart) TypeID() byte { return '0' }
func (p TextStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal text content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// ReasoningStreamPart corresponds to TYPE_ID 'g'.
type ReasoningStreamPart struct {
	Content string
}

func (p ReasoningStreamPart) TypeID() byte { return 'g' }
func (p ReasoningStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal reasoning content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// RedactedReasoningStreamPart corresponds to TYPE_ID 'i'.
type RedactedReasoningStreamPart struct {
	Data string `json:"data"`
}

func (p RedactedReasoningStreamPart) TypeID() byte { return 'i' }
func (p RedactedReasoningStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ReasoningSignatureStreamPart corresponds to TYPE_ID 'j'.
type ReasoningSignatureStreamPart struct {
	Signature string `json:"signature"`
}

func (p ReasoningSignatureStreamPart) TypeID() byte { return 'j' }
func (p ReasoningSignatureStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// SourceStreamPart corresponds to TYPE_ID 'h'.
type SourceStreamPart struct {
	SourceType string `json:"sourceType"`
	ID         string `json:"id"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

func (p SourceStreamPart) TypeID() byte { return 'h' }
func (p SourceStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FileStreamPart corresponds to TYPE_ID 'k'.
type FileStreamPart struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

func (p FileStreamPart) TypeID() byte { return 'k' }
func (p FileStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// DataStreamDataPart corresponds to TYPE_ID '2'.
type DataStreamDataPart struct {
	Content []any
}

func (p DataStreamDataPart) TypeID() byte { return '2' }
func (p DataStreamDataPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal data content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// MessageAnnotationStreamPart corresponds to TYPE_ID '8'.
type MessageAnnotationStreamPart struct {
	Content []any
}

func (p MessageAnnotationStreamPart) TypeID() byte { return '8' }
func (p MessageAnnotationStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal annotation content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

// ErrorStreamPart corresponds to TYPE_ID '3'.
type ErrorStreamPart struct {
	Content string
}

func (p ErrorStreamPart) TypeID() byte { return '3' }
func (p ErrorStreamPart) Format() (string, error) {
	jsonContent, err := json.Marshal(p.Content)
	if err != nil {
		return "", fmt.Errorf("failed to marshal error content: %w", err)
	}
	return fmt.Sprintf("%c:%s\n", p.TypeID(), string(jsonContent)), nil
}

type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

// ToolCallStartStreamPart corresponds to TYPE_ID 'b'.
type ToolCallStartStreamPart struct {
	ToolCallID string `json:"toolCallId"`
	ToolName   string `json:"toolName"`
}

func (p ToolCallStartStreamPart) TypeID() byte { return 'b' }
func (p ToolCallStartStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolCallDeltaStreamPart corresponds to TYPE_ID 'c'.
type ToolCallDeltaStreamPart struct {
	ToolCallID    string `json:"toolCallId"`
	ArgsTextDelta string `json:"argsTextDelta"`
}

func (p ToolCallDeltaStreamPart) TypeID() byte { return 'c' }
func (p ToolCallDeltaStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolCallStreamPart corresponds to TYPE_ID '9'.
type ToolCallStreamPart struct {
	ToolCallID string         `json:"toolCallId"`
	ToolName   string         `json:"toolName"`
	Args       map[string]any `json:"args"`
}

func (p ToolCallStreamPart) TypeID() byte { return '9' }
func (p ToolCallStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// ToolResultStreamPart corresponds to TYPE_ID 'a'.
type ToolResultStreamPart struct {
	ToolCallID string `json:"toolCallId"`
	Result     any    `json:"result"`
}

func (p ToolResultStreamPart) TypeID() byte { return 'a' }
func (p ToolResultStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// StartStepStreamPart corresponds to TYPE_ID 'f'.
type StartStepStreamPart struct {
	MessageID string `json:"messageId"`
}

func (p StartStepStreamPart) TypeID() byte { return 'f' }
func (p StartStepStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FinishReason defines the possible reasons for finishing a step or message.
type FinishReason string

const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonContentFilter FinishReason = "content-filter"
	FinishReasonToolCalls     FinishReason = "tool-calls"
	FinishReasonError         FinishReason = "error"
	FinishReasonOther         FinishReason = "other"
	FinishReasonUnknown       FinishReason = "unknown"
)

// Usage details the token usage.
type Usage struct {
	PromptTokens     *int64 `json:"promptTokens"`
	CompletionTokens *int64 `json:"completionTokens"`
}

// FinishStepStreamPart corresponds to TYPE_ID 'e'.
type FinishStepStreamPart struct {
	FinishReason FinishReason `json:"finishReason"`
	Usage        Usage        `json:"usage"`
	IsContinued  bool         `json:"isContinued"`
}

func (p FinishStepStreamPart) TypeID() byte { return 'e' }
func (p FinishStepStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// FinishMessageStreamPart corresponds to TYPE_ID 'd'.
type FinishMessageStreamPart struct {
	FinishReason FinishReason `json:"finishReason"`
	Usage        Usage        `json:"usage"`
}

func (p FinishMessageStreamPart) TypeID() byte { return 'd' }
func (p FinishMessageStreamPart) Format() (string, error) {
	return formatJSONPart(p)
}

// formatJSONPart formats parts where the content is the JSON representation of the struct itself.
func formatJSONPart(part DataStreamPart) (string, error) {
	jsonData, err := json.Marshal(part)
	if err != nil {
		return "", fmt.Errorf("failed to marshal part type %T: %w", part, err)
	}
	return fmt.Sprintf("%c:%s\n", part.TypeID(), string(jsonData)), nil
}

// Chat is the structure sent from `useChat` to the server.
// This can be extended if you'd like to send additional data with `body`.
type Chat struct {
	ID       string    `json:"id"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Content         string           `json:"content"`
	Role            string           `json:"role"`
	ToolInvocations []ToolInvocation `json:"toolInvocations"`
}

type ToolInvocationState string

const (
	ToolInvocationStateCall        ToolInvocationState = "call"
	ToolInvocationStatePartialCall ToolInvocationState = "partial-call"
	ToolInvocationStateResult      ToolInvocationState = "result"
)

type ToolInvocation struct {
	State      ToolInvocationState `json:"state"`
	ToolCallID string              `json:"toolCallId"`
	ToolName   string              `json:"toolName"`
	Args       any                 `json:"args"`
	Result     any                 `json:"result"`
}

// wipToolCall tracks partial tool call streams.
type wipToolCall struct {
	ID       string
	Name     string
	Args     string // Accumulates InputJSONDelta
	Finished bool
}

// PipeOptions are options for piping a provider to a DataStream.
type PipeOptions struct {
	HandleToolCall func(toolCall ToolCall) any
}

type PipeResponse[T any] struct {
	Messages     T
	FinishReason FinishReason
}
