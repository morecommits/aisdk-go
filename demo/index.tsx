import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { useChat } from "@ai-sdk/react";

const modelToProvider = {
  "gpt-4o": "openai",
  "claude-3-7-sonnet-latest": "anthropic",
  o1: "openai",
  "gemini-2.5-pro-preview-03-25": "google",
};

type model = keyof typeof modelToProvider;

const Chat = () => {
  const [model, setModel] = useState<model>("gpt-4o");
  const [thinking, setThinking] = useState(false);
  const { messages, input, handleInputChange, handleSubmit, error } = useChat({
    api: "/api/chat",
    body: {
      provider: modelToProvider[model],
      model,
      thinking,
    },
  });

  console.log(messages);

  return (
    <div>
      <div>
        <label htmlFor="model-select">Model: </label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value as model)}
        >
          {Object.entries(modelToProvider).map(([model, provider]) => (
            <option key={model} value={model}>
              {model} ({provider})
            </option>
          ))}
        </select>
        <button onClick={() => fetch("/api/dump")}>Dump Messages Struct</button>
      </div>

      <div style={{ marginTop: "10px" }}>
        <label htmlFor="thinking-checkbox">Enable Thinking: </label>
        <input
          type="checkbox"
          id="thinking-checkbox"
          checked={thinking}
          onChange={(e) => setThinking(e.target.checked)}
        />
      </div>

      {error && (
        <div style={{ color: "red" }}>
          <p>Error: {error.message}</p>
        </div>
      )}

      <div>
        {messages.map((m) => {
          switch (m.role as string) {
            case "user":
              return (
                <div
                  key={m.id}
                  style={{
                    marginBottom: "10px",
                    borderBottom: "1px solid #eee",
                    paddingBottom: "5px",
                  }}
                >
                  <strong>User:</strong> {m.content}
                </div>
              );
            case "assistant":
              // Iterate over parts for text and tool invocations/results
              return (
                <div
                  key={m.id}
                  style={{
                    marginBottom: "10px",
                    borderBottom: "1px solid #eee",
                    paddingBottom: "5px",
                  }}
                >
                  <strong>AI:</strong>
                  {m.parts?.map((part, index) => {
                    switch (part.type) {
                      case "text":
                        return <div key={index}>{part.text}</div>;
                      case "tool-invocation":
                        const toolInvocation = part.toolInvocation as any; // Use type assertion for potentially missing fields in base types
                        return (
                          <div
                            key={index}
                            style={{
                              marginTop: "5px",
                              marginLeft: "15px",
                              borderLeft: "2px solid lightblue",
                              paddingLeft: "10px",
                            }}
                          >
                            <em>Tool Call:</em>
                            <div
                              style={{
                                fontFamily: "monospace",
                                fontSize: "0.9em",
                                marginTop: "3px",
                              }}
                            >
                              - Name: {toolInvocation.toolName} (ID:{" "}
                              {toolInvocation.toolCallId})<br />- Args:{" "}
                              {JSON.stringify(toolInvocation.args)}
                              <br />
                              {/* Display result if available */}
                              {toolInvocation.result && (
                                <>
                                  - Result:{" "}
                                  {JSON.stringify(toolInvocation.result)}
                                </>
                              )}
                            </div>
                          </div>
                        );
                      case "reasoning":
                        return (
                          <div
                            key={index}
                            style={{
                              marginTop: "5px",
                              marginLeft: "15px",
                              borderLeft: "2px solid lightgray",
                              paddingLeft: "10px",
                              fontStyle: "italic",
                              color: "#666",
                            }}
                          >
                            <em>Reasoning:</em>
                            <div
                              style={{
                                fontFamily: "monospace",
                                fontSize: "0.9em",
                                marginTop: "3px",
                                whiteSpace: "pre-wrap",
                              }}
                            >
                              {part.reasoning}
                            </div>
                          </div>
                        );
                      default:
                        return null;
                    }
                  })}
                </div>
              );
            default:
              return null; // Or some fallback rendering
          }
        })}
      </div>

      <form onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Say something..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
};

const root = createRoot(document.getElementById("root")!);
root.render(<Chat />);
