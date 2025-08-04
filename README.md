

## Pipeline Overview
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   Processing     │───▶│    Output       │
│   (.mp4, .mov)  │    │   Pipeline       │    │  (.md/.html/.pdf)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Components:     │
                    │  • Audio Extract │
                    │  • Whisper AI    │
                    │  • Diarization   │
                    │  • Frame Extract │
                    │  • Visual Analysis│
                    │  • Export Engine │
                    └──────────────────┘