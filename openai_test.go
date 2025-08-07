package aisdk_test

import (
	"context"
	"os"
	"testing"

	"github.com/coder/aisdk-go"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/stretchr/testify/require"
)

func TestMessagesToOpenAI_Live(t *testing.T) {
	t.Parallel()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}
	ctx := context.Background()
	client := openai.NewClient(option.WithAPIKey(apiKey))

	// Ensure messages are converted correctly.
	prompt := "use the 'print' tool to print 'Hello, world!' and then show the result"
	messages, err := aisdk.MessagesToOpenAI([]aisdk.Message{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role: "user",
			Parts: []aisdk.Part{
				{Type: aisdk.PartTypeText, Text: prompt},
			},
		},
	})
	require.Len(t, messages, 2)
	require.NotNil(t, messages[1].OfUser)
	require.Len(t, messages[1].OfUser.Content.OfArrayOfContentParts, 1)
	require.NotNil(t, messages[1].OfUser.Content.OfArrayOfContentParts[0].OfText)
	require.Equal(t, messages[1].OfUser.Content.OfArrayOfContentParts[0].OfText.Text, prompt)
	require.NoError(t, err)

	stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model:    openai.ChatModelGPT4o,
		Messages: messages,
	})
	require.NoError(t, err)

	dataStream := aisdk.OpenAIToDataStream(stream)
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
