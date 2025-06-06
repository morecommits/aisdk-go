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
	"github.com/coder/aisdk-go"
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

	var lastMessages []aisdk.Message

	mux := http.NewServeMux()
	mux.HandleFunc("/api/dump", func(w http.ResponseWriter, r *http.Request) {
		data, _ := json.MarshalIndent(lastMessages, "", "  ")
		os.WriteFile("dump.json", data, 0644)
		w.WriteHeader(http.StatusOK)
		fmt.Printf("dumped to dump.json\n")
	})
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

		handleToolCall := func(toolCall aisdk.ToolCall) any {
			return map[string]string{
				"message": "It worked!",
			}
		}

		tools := []aisdk.Tool{{
			Name:        "test",
			Description: "A test tool. Only use if the user explicitly requests it.",
			Schema: aisdk.Schema{
				Required: []string{"message"},
				Properties: map[string]any{
					"message": map[string]any{
						"type": "string",
					},
				},
			},
		}}

		aisdk.WriteDataStreamHeaders(w)

		for {
			var stream aisdk.DataStream
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
				stream = aisdk.OpenAIToDataStream(openAIClient.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
					Model:               req.Model,
					Messages:            messages,
					ReasoningEffort:     reasoningEffort,
					Tools:               aisdk.ToolsToOpenAI(tools),
					MaxCompletionTokens: openai.Int(2048),
				}))
				break
			case "anthropic":
				messages, system, err := aisdk.MessagesToAnthropic(req.Messages)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}

				thinking := anthropic.ThinkingConfigParamUnion{}
				if req.Thinking {
					thinking = anthropic.ThinkingConfigParamOfEnabled(2048)
				}
				stream = aisdk.AnthropicToDataStream(anthropicClient.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
					Model:     anthropic.Model(req.Model),
					Messages:  messages,
					System:    system,
					MaxTokens: 4096,
					Thinking:  thinking,
					Tools:     aisdk.ToolsToAnthropic(tools),
				}))
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
				stream = aisdk.GoogleToDataStream(googleClient.Models.GenerateContentStream(ctx, req.Model, messages, &genai.GenerateContentConfig{
					Tools:          tools,
					ThinkingConfig: thinkingConfig,
				}))
				break
			}
			if stream == nil {
				http.Error(w, "invalid provider", http.StatusBadRequest)
				return
			}
			var acc aisdk.DataStreamAccumulator
			stream = stream.WithToolCalling(handleToolCall)
			stream = stream.WithAccumulator(&acc)

			// Add system message if not present
			if len(req.Messages) == 0 || req.Messages[0].Role != "system" {
				req.Messages = append([]aisdk.Message{{
					Role:    "system",
					Content: "You are a helpful assistant. When using tools, always provide a text response after receiving the tool result to describe what happened. Do not make additional tool calls unless explicitly requested by the user.",
				}}, req.Messages...)
			}

			err = stream.Pipe(w)
			if err != nil {
				log.Println(err)
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			req.Messages = append(req.Messages, acc.Messages()...)
			lastMessages = req.Messages[:]
			if acc.FinishReason() == aisdk.FinishReasonToolCalls {
				continue
			}
			break
		}

	})

	return http.Serve(listener, mux)
}
