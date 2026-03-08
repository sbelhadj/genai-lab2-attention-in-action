"""
Generation Utilities for Lab 2 — Prompt as Context Experiments

Provides helper functions for calling Ollama and comparing outputs.
Fully implemented — students use these as tools.
"""

import requests
import json
import os


def generate(prompt, model="llama3.2:3b", temperature=0.3, max_tokens=200, timeout=60):
    """
    Generate a completion using Ollama.

    Args:
        prompt: The input prompt string.
        model: Ollama model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        Completion string, or error message if generation fails.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"[Generation failed: {e}]"


def compare_outputs(prompts_dict, temperature=0.3, runs=1):
    """
    Generate outputs for multiple prompts and display for comparison.

    Args:
        prompts_dict: dict mapping condition labels to prompt strings.
        temperature: Sampling temperature.
        runs: Number of generations per prompt.

    Returns:
        dict mapping labels to lists of output strings.
    """
    results = {}
    for label, prompt in prompts_dict.items():
        print(f"\n{'=' * 70}")
        print(f"CONDITION: {label}")
        print(f"{'=' * 70}")
        prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
        print(f"Prompt:\n{prompt_preview}")
        print(f"{'-' * 70}")
        outputs = []
        for r in range(runs):
            output = generate(prompt, temperature=temperature)
            outputs.append(output)
            prefix = f"Run {r + 1}: " if runs > 1 else "Output: "
            print(f"{prefix}\n{output}\n")
        results[label] = outputs
    return results


def is_ollama_available():
    """Check if Ollama is reachable."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def load_precomputed(filepath="data/precomputed_outputs.json"):
    """
    Load pre-computed experiment outputs.

    Args:
        filepath: Path to the JSON file.

    Returns:
        dict of experiment data, or empty dict if file not found.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ Pre-computed file not found: {filepath}")
        return {}
