# conversation-copilot

## Run

```bash
./start
./start --list-devices
./start --smoke-llm
```

Click **QA** to open the panel and enable automatic answers to spoken questions.
Click **Summary** to open its panel and generate a fresh summary. The Q/A panel
shows when it is listening, generating an answer, or unable to complete a request.
Q/A uses an agent prompt to interpret speech from both microphone and external
audio. The model decides which questions are complete, which fragments belong
together, and when to skip; keyword and similarity rules do not gate Q/A.
While QA is enabled, it samples the latest five seconds of speech every second.
Inference runs in the background. If a request is still running, the newest
pending snapshot replaces older pending snapshots. The agent receives previously
answered questions so it can skip duplicates and paraphrases.
Summaries use the full session transcript from all speakers and cover the main
topics, decisions, action items, and unresolved questions.

Click **Chat** to ask questions about the recorded conversation. Each answer uses
the full transcript available when you send your question, with chat history for
follow-ups. Press Enter to send or Shift+Enter for a new line. Chat history lasts
for the current app session.
Use **Config → Chat** to edit the response prompt. Save an empty prompt to restore
the default. Chat answers directly and includes transcript quotes or timestamps
only when requested. The chat composer supports multiline questions and copying
answers.

## Model inference

The app sends chat requests to `FUNCTION_URL/v1/chat/completions`. Set these
values in the project root `.env` (see `.env.example`):

```dotenv
FUNCTION_URL=https://gwjg7secnplcnrbdxrb52ijbge0dwwnu.lambda-url.us-east-1.on.aws/
LLM_MODEL=nova-pro
```

Keep your endpoint's `INFERENCE_API_KEY` in `.env`. No AWS profile or AWS CLI
is needed. Shell environment variables override `.env`.

```bash
./start --smoke-llm --quick
./start --smoke-llm --quick --model ministral-8b
./start --smoke-llm
./start
```

The quick test sends one hello request; the full test also checks streaming and
Q/A reconstruction. Neither test needs the audio dependencies. To test another
endpoint, pass its plain URL without Markdown link formatting:

```bash
FUNCTION_URL='https://gwjg7secnplcnrbdxrb52ijbge0dwwnu.lambda-url.us-east-1.on.aws/' ./start --smoke-llm --quick
```

This script uses the configured URL; it does not resolve `dev` or `prod` aliases.
An inline variable assignment applies only to that command. Use `export` if you
also need the variable in later shell commands such as `curl`.

## Project layout

The project root contains:

```text
start
README.md
requirements.txt
prompt/
app/
```

- `prompt/qa_instructions.txt`: default Q/A agent instructions.
- `prompt/qa_prompt.txt`, `prompt/summary_prompt.txt`, `prompt/chat_prompt.txt`: saved prompts edited in Config.
- `app/backend/`: Python backend and overlay interface.
- `app/main.js`, `app/preload.js`: Electron window and bridge.
- `.env`: local configuration and inference credentials.
- `app/profile/`: reference notes.
- `app/log/`: session transcripts and answers.
- `app/tests/`, `app/packaging/`: tests and packaging configuration.
- `app/venv/`, `app/node_modules/`, `app/models/`: local dependencies and models.

Implementation and supporting files live in `app/`. Git configuration remains
at the root. `start` works from any directory.

## Development

```bash
app/venv/bin/python -m unittest discover -s app/tests
npm --prefix app start
npm --prefix app run backend
npm --prefix app run dist
```

Logs and build outputs are created inside `app/`.
