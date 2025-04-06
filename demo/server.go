package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	"github.com/kylecarbs/aisdk-go"
	"github.com/openai/openai-go"
	openaioption "github.com/openai/openai-go/option"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	err := run(ctx)
	if err != nil {
		log.Fatal(err)
	}
}

// run starts the server.
func run(ctx context.Context) error {
	listener, err := net.Listen("tcp", "127.0.0.1:5432")
	if err != nil {
		return err
	}
	defer listener.Close()

	openAIClient := openai.NewClient(openaioption.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
	anthropicClient := anthropic.NewClient(anthropicoption.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))
	// Ignore this error - the user might not use Google.
	googleClient, _ := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GOOGLE_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})

	mux := http.NewServeMux()
	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			aisdk.Chat
			Provider string `json:"provider"`
			Model    string `json:"model"`
			Thinking bool   `json:"thinking"`
		}
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		opts := &aisdk.PipeOptions{
			HandleToolCall: func(toolCall aisdk.ToolCall) any {
				return map[string]string{
					"message": "It worked!",
				}
			},
		}

		dataStream := aisdk.NewDataStream(w)

		tools := []aisdk.Tool{{
			Name:        "test",
			Description: "A test tool. Only use if the user explicitly requests it.",
			Parameters: map[string]any{
				"message": map[string]string{
					"type": "string",
				},
			},
		}}

		switch req.Provider {
		case "openai":
			messages, err := aisdk.MessagesToOpenAI(req.Messages)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			reasoningEffort := openai.ReasoningEffort("")
			if req.Thinking {
				reasoningEffort = openai.ReasoningEffortMedium
			}

			for {
				stream := openAIClient.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
					Model:               req.Model,
					Messages:            messages,
					ReasoningEffort:     reasoningEffort,
					Tools:               aisdk.ToolsToOpenAI(tools),
					MaxCompletionTokens: openai.Int(2048),
				})
				response, err := aisdk.PipeOpenAIToDataStream(stream, dataStream, opts)
				if err != nil {
					dataStream.Write(aisdk.ErrorStreamPart{
						Content: fmt.Sprintf("Error: %s", err),
					})
					return
				}

				something, err := aisdk.OpenAIToMessages(response.Messages)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				rmsgs, err := aisdk.MessagesToOpenAI(something)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}

				messages = append(messages, rmsgs...)
				if response.FinishReason == aisdk.FinishReasonToolCalls {
					continue
				}
				break
			}
			break
		case "anthropic":
			messages, system, err := aisdk.MessagesToAnthropic(req.Messages)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			for {
				dataStream.Write(aisdk.StartStepStreamPart{
					MessageID: req.ID,
				})

				thinking := anthropic.ThinkingConfigParamUnion{}
				if req.Thinking {
					thinking = anthropic.ThinkingConfigParamOfThinkingConfigEnabled(2048)
				}

				stream := anthropicClient.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
					Model:     req.Model,
					Messages:  messages,
					System:    system,
					MaxTokens: 4096,
					Thinking:  thinking,
					Tools:     aisdk.ToolsToAnthropic(tools),
				})
				response, err := aisdk.PipeAnthropicToDataStream(stream, dataStream, opts)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				messages = append(messages, response.Messages...)
				if response.FinishReason == aisdk.FinishReasonToolCalls {
					continue
				}
				break
			}

			break
		case "google":
			messages, err := aisdk.MessagesToGoogle(req.Messages)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			var thinkingConfig *genai.ThinkingConfig
			if req.Thinking {
				thinkingConfig = &genai.ThinkingConfig{
					IncludeThoughts: true,
				}
			}

			tools, err := aisdk.ToolsToGoogle(tools)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			for {
				stream := googleClient.Models.GenerateContentStream(ctx, req.Model, messages, &genai.GenerateContentConfig{
					Tools:          tools,
					ThinkingConfig: thinkingConfig,
				})

				response, err := aisdk.PipeGoogleToDataStream(stream, dataStream, opts)
				if err != nil {
					dataStream.Write(aisdk.ErrorStreamPart{
						Content: fmt.Sprintf("Error: %s", err),
					})
					return
				}
				messages = append(messages, response.Messages...)

				if response.FinishReason == aisdk.FinishReasonToolCalls {
					continue
				}
				break
			}

			break
		default:
			http.Error(w, "invalid provider", http.StatusBadRequest)
			return
		}
	})

	return http.Serve(listener, mux)
}
