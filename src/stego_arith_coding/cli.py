"""
Command-line interface for the arithmetic steganography library.
"""

import logging
import os
import sys
from pathlib import Path
import datetime
from typing import Optional

try:
    import click
except ImportError:
    click = None

from .core import ArithmeticSteganography
from .config import ArithmeticSteganographyConfig
from .visualize import render_svg_from_json


def main():
    """Main CLI entry point."""
    if click is None:
        print(
            "Error: click is not installed. Install with: pip install stego-arith-coding[cli]",
            file=sys.stderr,
        )
        sys.exit(1)

    cli()


@click.group()
@click.option("--api-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, api_key: Optional[str], verbose: bool):
    """Arithmetic coding steganography CLI tool."""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    
    # Create default config
    ctx.obj["config"] = ArithmeticSteganographyConfig()

    # Set up logging
    # Set root logger to WARNING to suppress most third-party noise
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    
    # Configure our application logger based on verbose flag
    app_logger = logging.getLogger("stego_arith_coding")
    app_level = logging.DEBUG if verbose else logging.INFO
    app_logger.setLevel(app_level)
    
    # Suppress noisy third-party library logs unless in verbose mode
    if not verbose:
        # Suppress OpenAI client logs completely in non-verbose mode
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("requests").setLevel(logging.ERROR)
    else:
        # In verbose mode, allow WARNING+ from third-party libraries
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


@cli.command()
@click.argument("context")
@click.argument("message")
@click.option(
    "--artifacts",
    "-a",
    required=False,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    help="Artifacts directory (optional). If provided without a value, defaults to ./artifacts/<YYYY-MM-DD>_encode/."
)
@click.pass_context
def encode(ctx, context: str, message: str, artifacts: Optional[str]):
    """Encode a message into text using arithmetic steganography.

    CONTEXT: The initial text context to start encoding from
    MESSAGE: The secret message to encode
    """
    try:
        config = ctx.obj["config"]
        stego = ArithmeticSteganography(
            openai_api_key=ctx.obj["api_key"], config=config
        )

        # Determine artifacts directory (if any)
        artifacts_dir_path = None
        if artifacts is not None:
            if str(artifacts).strip():
                artifacts_dir_path = Path(artifacts)
            else:
                date_str = datetime.date.today().isoformat()
                artifacts_dir_path = Path("artifacts") / f"{date_str}_encode"
            artifacts_dir_path.mkdir(parents=True, exist_ok=True)

            # Add verbose file handler to our app logger
            app_logger = logging.getLogger("stego_arith_coding")
            logfile_path = artifacts_dir_path / "encoding-verbose.log"
            if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(logfile_path) for h in app_logger.handlers):
                fh = logging.FileHandler(logfile_path, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                app_logger.addHandler(fh)

        # Convert message to bits
        message_bits = stego.text_to_bits(message)

        # Encode the message using config precision - results printed by encode method
        encoded_text = stego.encode(
            context,
            message_bits,
            config.precision,
            artifacts_dir=str(artifacts_dir_path) if artifacts_dir_path else None,
        )

        click.echo(f"Encoded text: {repr(encoded_text)}")

        # Save artifacts if enabled
        if artifacts_dir_path:
            (artifacts_dir_path / "encoding-result.txt").write_text(encoded_text, encoding="utf-8")
            # Create visualization from coding_data.json
            json_path = artifacts_dir_path / "encoding-coding-data.json"
            if json_path.exists():
                svg_path = artifacts_dir_path / "encoding-visualization.svg"
                try:
                    render_svg_from_json(str(json_path), str(svg_path))
                except Exception as viz_err:
                    logging.getLogger("stego_arith_coding").warning(f"Failed to render SVG: {viz_err}")
            click.echo(f"Artifacts saved to {artifacts_dir_path}", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("context")
@click.argument("encoded_text", required=False)
@click.option("--input", "-i", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True), help="Read encoded text from file instead of argument")
@click.option(
    "--artifacts",
    "-a",
    required=False,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    help="Artifacts directory (optional). If provided without a value, defaults to ./artifacts/<YYYY-MM-DD>_decode/."
)
@click.pass_context
def decode(
    ctx,
    context: str,
    encoded_text: Optional[str],
    input: Optional[str],
    artifacts: Optional[str],
):
    """Decode a hidden message from encoded text using ETX marker detection.

    CONTEXT: The initial text context that was used during encoding
    ENCODED_TEXT: The encoded text containing the hidden message (or use --input)
    """
    try:
        if not encoded_text and not input:
            click.echo("Error: must provide ENCODED_TEXT or --input", err=True)
            sys.exit(2)
        if input:
            with open(input, "r", encoding="utf-8") as f:
                encoded_text = f.read().strip()

        # Determine artifacts directory (if any)
        artifacts_dir_path = None
        if artifacts is not None:
            if str(artifacts).strip():
                artifacts_dir_path = Path(artifacts)
            else:
                date_str = datetime.date.today().isoformat()
                artifacts_dir_path = Path("artifacts") / f"{date_str}_decode"
            artifacts_dir_path.mkdir(parents=True, exist_ok=True)

            # Add verbose file handler to our app logger
            app_logger = logging.getLogger("stego_arith_coding")
            logfile_path = artifacts_dir_path / "decoding-verbose.log"
            if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(logfile_path) for h in app_logger.handlers):
                fh = logging.FileHandler(logfile_path, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                app_logger.addHandler(fh)

        config = ctx.obj["config"]
        stego = ArithmeticSteganography(
            openai_api_key=ctx.obj["api_key"], config=config
        )

        # Decode the message using config precision and ETX detection - results printed by decode method
        decoded_bits = stego.decode(
            context,
            encoded_text,
            config.precision,
            artifacts_dir=str(artifacts_dir_path) if artifacts_dir_path else None,
        )

        # Convert bits back to text
        decoded_message = stego.bits_to_text(decoded_bits)

        click.echo(f"Decoded text: {repr(decoded_message)}")

        # Save artifacts if enabled
        if artifacts_dir_path:
            (artifacts_dir_path / "decoding-result.txt").write_text(decoded_message, encoding="utf-8")
            # Create visualization from coding_data.json
            json_path = artifacts_dir_path / "decoding-coding-data.json"
            if json_path.exists():
                svg_path = artifacts_dir_path / "decoding-visualization.svg"
                try:
                    render_svg_from_json(str(json_path), str(svg_path))
                except Exception as viz_err:
                    logging.getLogger("stego_arith_coding").warning(f"Failed to render SVG: {viz_err}")
            click.echo(f"Artifacts saved to {artifacts_dir_path}", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="text-to-bits")
@click.argument("text")
def text_to_bits(text: str):
    """Convert text to binary representation with ETX marker."""
    config = ArithmeticSteganographyConfig()
    stego = ArithmeticSteganography(config=config)
    bits = stego.text_to_bits(text)
    bit_string = "".join(str(b) for b in bits)
    click.echo(f"Text: {text}")
    click.echo(f"Bits: {bit_string}")
    click.echo(f"Length: {len(bits)} bits")


@cli.command(name="bits-to-text")
@click.argument("bits")
def bits_to_text(bits: str):
    """Convert binary representation to text with ETX marker handling."""
    try:
        bit_list = [int(b) for b in bits if b in "01"]
        config = ArithmeticSteganographyConfig()
        stego = ArithmeticSteganography(config=config)
        text = stego.bits_to_text(bit_list)
        click.echo(f"Bits: {bits}")
        click.echo(f"Text: {text}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="visualize")
@click.argument(
    "json_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
)
def visualize_cmd(json_path: str):
    """Render SVG visualization from a coding_data.json file into the same directory."""
    try:
        json_p = Path(json_path)
        svg_path = json_p.with_name("visualization.svg")
        render_svg_from_json(str(json_p), str(svg_path))
        click.echo(f"Visualization written to {svg_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
