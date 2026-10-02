# conversation-copilot

## Run

```bash
./start
./start --list-devices
./start --smoke-llm
```

Click **QA** to extract the other speaker's latest question, with a short context
line. V1 does not generate answer hints. The model recognizes unpunctuated questions,
requests, implicit questions, and follow-ups, and returns `WAIT` while the latest
question is incomplete. A refinement updates the existing card with the same ID.

QA checks for changed speech every second and retains 45 seconds of conversation.
ASR repeats are suppressed and cumulative fragments are merged. One model call
extracts the question and decides whether it is new or an update; no agent or
second duplicate-check call is needed. While inference runs, only the newest
pending snapshot is retained. Unchanged speech is not sent again automatically.

By default, `EXT` and `EXT-*` speakers are eligible; microphone speech provides
context only. For a single-mic conversation, set `QA_TARGET_SPEAKERS=MIC-3` to the
other person's diarized label (numbers do not imply roles). Unlabeled input is
eligible. Use comma-separated labels to select multiple speakers. Configure
`QA_WINDOW_SEC` (default 45) and `QA_INTERVAL_SEC` (default 1) in `.env`.
`LLM_FAST_MODEL`, when set, handles extraction; otherwise `LLM_MODEL` is used.
Question logs contain JSON lines with stable IDs, context, and NEW/UPDATE actions.
Live event consumers receive `question_id`, `question`, `context`, `action`, and
`replaces_question`; `answer` mirrors context for compatibility with older clients.

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

- `prompt/qa_instructions.txt`: default live question extraction instructions.
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

### 持续转写与 partial/final

麦克风以 16 kHz、40 ms frame 持续采集。Silero VAD 保留起音，默认静音
500 ms 后结束语句；25 秒仅作为无停顿语音的安全上限。`STT_CHUNK_SEC=0.4`
是本地 partial 的刷新间隔，不是断句长度。每个 utterance 的 partial 在 UI
原位替换，允许修改不稳定词尾；final 才追加日志并送入问答。重连 UI 时仅回放
每个 utterance 最新状态。本地 Whisper 滚动解码保留整句音频，以及每个输入源
最近 2000 字符的 final context，并用 `STT_VOCABULARY` 提供 prompt bias；
这是提示词偏置，并不保证专业词一定正确。

设置 `STT_WS_URL` 后使用远端原生 streaming，MIC/EXT 各自保持一条 WebSocket。
需要安装 `websocket-client`（已加入 requirements）。此接口是项目的 bridge 协议，
不能直接填任意厂商的 API URL；部署的 ASR bridge 需实现：

- 首条 JSON：`{"type":"start","source":"MIC","sample_rate":16000,
  "encoding":"pcm_s16le","frame_ms":40,"vocabulary":["Bedrock","IAM","LangGraph"],"context":""}`。
- 后续 binary 消息：连续单声道 PCM16 little-endian，每条最多 640 samples（40 ms），
  包含静音；连接不随语句关闭。可用 `STT_WS_TOKEN` 发送 Bearer 鉴权。
- VAD 结束时 JSON：`{"type":"commit","utterance_id":"1","context":"前文"}`。
  bridge 以客户端 commit 为断句依据，语句 ID 从 `1` 递增（每个连接独立）。
- bridge 返回 `{"type":"partial","utterance_id":"1","text":"..."}` 或
  `{"type":"final","utterance_id":"1","text":"..."}`；同一语句 ID 的文本是整句替换。
  bridge 必须利用 vocabulary/context，并保留识别会话状态。

远端模式使用设备标签 MIC/EXT，不执行本地声纹跟踪或 Whisper 模型切换。
目前断线会报告识别错误，需要重启会话；没有自动续传。未配置服务地址时，
默认仍使用本地 Whisper 滚动解码，而非原生远端流式 ASR。
