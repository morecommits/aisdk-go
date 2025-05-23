FROM golang:1.21

WORKDIR /app

COPY . .

RUN go mod tidy && go build ./...
