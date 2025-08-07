package aisdk_test

import (
	"context"
	"os"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/coder/aisdk-go"
	"github.com/stretchr/testify/require"
)

func TestMessagesToAnthropic_Live(t *testing.T) {
	t.Parallel()
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY is not set")
	}
	ctx := context.Background()
	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	// Ensure messages are converted correctly.
	prompt := "use the 'print' tool to print 'Hello, world!' and then show the result"
	messages, systemPrompts, err := aisdk.MessagesToAnthropic([]aisdk.Message{
		{
			Role: "system",
			Parts: []aisdk.Part{
				{Text: "You are a helpful assistant.", Type: aisdk.PartTypeText},
			},
		},
		{
			Role: "user", Parts: []aisdk.Part{
				{Text: prompt, Type: aisdk.PartTypeText},
			},
		},
	})
	require.Len(t, messages, 1)
	require.Len(t, systemPrompts, 1)
	require.Len(t, messages[0].Content, 1)
	require.NotNil(t, messages[0].Content[0].OfText)
	require.Equal(t, messages[0].Content[0].OfText.Text, prompt)
	require.NoError(t, err)

	stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Messages:  messages,
		Model:     anthropic.ModelClaude3_5SonnetLatest,
		System:    systemPrompts,
		MaxTokens: 10,
	})
	require.NoError(t, err)

	dataStream := aisdk.AnthropicToDataStream(stream)
	var streamErr error
	dataStream(func(part aisdk.DataStreamPart, err error) bool {
		if err != nil {
			streamErr = err
			return false
		}
		return true
	})
	require.NoError(t, streamErr)
}
