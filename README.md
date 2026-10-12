# voxlib

Turn YouTube transcripts into a queryable knowledge library.

`voxlib` fetches a video's transcript, cleans and semantically chunks it,
uses local LLMs (via Ollama) to summarize it, extract entities, and
classify it against a growing category set, and stores everything in a
database for retrieval.

## Status

Pre-alpha. The public API and database schema are still changing.

## Requirements

- Python 3.12
- [Ollama](https://ollama.com/) running locally with the models you
  configure in `voxlib.yaml` (see the PRD for defaults).

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from voxlib import KnowledgeBase

kb = KnowledgeBase.from_config()
video = kb.add_video("dQw4w9WgXcQ")
print(video.root_summary.tldr)
```

(This snippet describes the target API; not all of it exists yet.)

## License

Copyright (C) 2026 David Jiménez.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
