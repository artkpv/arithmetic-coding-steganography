# Contributing

Thanks for your interest in contributing!

- Use Python 3.11+.
- Set up a [virtualenv](virtualenv.md) and install with extras: `pip install .[cli,test]`.
- Run tests before submitting: `pytest -q`.
- Format your code and follow PEP8.
- Keep PRs focused and add tests when fixing bugs.

## Development tips

- CLI entrypoint: `stego-arith`
- Offline helpers (`text-to-bits`, `bits-to-text`) do not require an API key.
- Set `OPENAI_API_KEY` for encode/decode operations that query the API.

## Reporting issues

Please include:
- Steps to reproduce
- Expected vs actual behavior
- Versions (Python, OS) and relevant logs
