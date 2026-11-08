"""Gemma / Gemini Natural Language Shot Parser for Antigravity.

This module provides the natural-language understanding (NLU) layer that
translates free-form user or director requests (e.g., "Make me look heroic and
confident on stage") into structured cinematic shot plans executable by the
UR5e robotic cameraman and the User Digital Twin Coach.

Architecture & Fallback Strategy:
  1. Primary LLM Mode: Queries the Google GenAI SDK (`google.genai`) using the
     configured `GEMINI_API_KEY` from `.env` to extract a JSON payload matching
     the required schema (`shot_type`, `creator_instruction`, `director_rationale`).
  2. Offline Heuristic Fallback: If no API key is present, the network is
     unreachable, or the model returns malformed output, the parser seamlessly
     falls back to deterministic keyword matching so the live robotics demo
     never stalls or crashes.

Typical usage example:
  ```python
  parser = GemmaDirectorParser()
  plan = parser.parse_prompt("Give me a 360 orbit shot around the workspace")
  # plan -> {
  #   "shot_type": "orbit",
  #   "creator_instruction": "...",
  #   "director_rationale": "..."
  # }
  ```
"""

import json
import os
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class GemmaDirectorParser:
    """Translates natural-language director prompts into structured shot plans.

    Attributes:
        SUPPORTED_SHOTS: Class-level list of valid shot primitive identifiers
            supported by `RobotPathPlanner` and `UserDirectorDigitalTwin`.
        model_id: The identifier of the Gemini/Gemma model used for inference.
        client: An initialized `google.genai.Client` instance if `GEMINI_API_KEY`
            is available, or `None` when running in offline heuristic mode.
    """

    SUPPORTED_SHOTS: List[str] = [
        "orbit",
        "dolly_in",
        "low_angle_hero",
        "tracking",
    ]

    def __init__(self, model_id: str = "gemini-3.8-flash") -> None:
        """Initializes the prompt parser and attempts to connect to Google GenAI.

        Args:
            model_id: Model name to query via the Google GenAI SDK. Defaults to
                `'gemini-3.8-flash'`.
        """
        self.model_id: str = model_id
        self.client: Optional[object] = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai

                self.client = genai.Client(api_key=api_key)
            except Exception:
                # Gracefully degrade to rule-based fallback if SDK init fails.
                self.client = None

    def parse_prompt(self, user_prompt: str) -> Dict[str, str]:
        """Parses a free-form director prompt into a validated shot plan dictionary.

        First attempts structured JSON extraction via the configured LLM client.
        If the client is unavailable or returns an unsupported shot type, falls
        back to deterministic keyword classification.

        Args:
            user_prompt: Natural-language description of the desired camera shot
                or creative mood.

        Returns:
            A dictionary with the following keys:
                - `'shot_type'`: One of `'orbit'`, `'dolly_in'`,
                  `'low_angle_hero'`, or `'tracking'`.
                - `'creator_instruction'`: Actionable coaching guidance for the
                  human subject.
                - `'director_rationale'`: Brief explanation of why this camera
                  trajectory matches the prompt.
        """
        # 1. Attempt LLM-based structured parsing if GenAI client is configured.
        if self.client is not None:
            try:
                prompt_text = f"""
You are the AI Director for an autonomous robotic camera system (UR5e arm).
Analyze the user's cinematic request and select ONE optimal shot primitive from:
- 'orbit' (circular panning around subject for dynamic 360 energy)
- 'dolly_in' (smooth forward push-in for intimacy or revelation)
- 'low_angle_hero' (low perspective looking up for power/confidence)
- 'tracking' (lateral tracking following subject movement)

User Request: "{user_prompt}"

Return ONLY a valid JSON object matching this schema:
{{
  "shot_type": "orbit" | "dolly_in" | "low_angle_hero" | "tracking",
  "creator_instruction": "concise instruction for what the human creator in front of the camera should do",
  "director_rationale": "one sentence explaining why this camera trajectory fits the shot"
}}
"""
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt_text,
                )
                text = (response.text or "").strip()
                match = re.search(r"\{[\s\S]*\}", text)
                if match:
                    parsed = json.loads(match.group(0))
                    if parsed.get("shot_type") in self.SUPPORTED_SHOTS:
                        return parsed
            except Exception:
                # Fall through to deterministic heuristics on API/network error.
                pass

        # 2. Fast, deterministic rule-based fallback for offline resilience.
        prompt_lower = user_prompt.lower()
        if any(
            w in prompt_lower
            for w in ["circle", "orbit", "360", "spin", "around"]
        ):
            return {
                "shot_type": "orbit",
                "creator_instruction": (
                    "Stand in the creator zone and step in a slow clockwise "
                    "circle facing the lens."
                ),
                "director_rationale": (
                    "Circular orbit captures 360 energy and highlights "
                    "surrounding workspace."
                ),
            }
        elif any(
            w in prompt_lower
            for w in ["dolly", "zoom", "push", "closer", "intimate", "reveal"]
        ):
            return {
                "shot_type": "dolly_in",
                "creator_instruction": (
                    "Start at the rear marker and walk forward toward the "
                    "camera steadily."
                ),
                "director_rationale": (
                    "Dolly-in builds dramatic focus and draws audience "
                    "attention to the creator."
                ),
            }
        elif any(
            w in prompt_lower
            for w in ["hero", "low angle", "low-angle", "power", "epic", "strong", "confidence", "confident"]
        ):
            return {
                "shot_type": "low_angle_hero",
                "creator_instruction": (
                    "Stand tall on the green mat, plant feet wide, and deliver "
                    "your lines with power."
                ),
                "director_rationale": (
                    "Low-angle framing emphasizes authority and stature."
                ),
            }
        else:
            return {
                "shot_type": "tracking",
                "creator_instruction": (
                    "Walk naturally across the frame at a smooth, steady pace."
                ),
                "director_rationale": (
                    "Fluid lateral tracking maintains balanced subject framing."
                ),
            }


if __name__ == "__main__":
    # Standalone verification of prompt parsing across all four shot primitives.
    parser = GemmaDirectorParser()
    test_queries = [
        "Make me look heroic and confident on stage",
        "Give me a 360 orbit shot around the workspace",
        "Push in slowly as I reveal the new product",
        "Just follow me as I walk across the room",
    ]
    for query in test_queries:
        result = parser.parse_prompt(query)
        print(f"Query: '{query}'")
        print(f" -> Shot: {result['shot_type']}")
        print(f" -> Instruction: {result['creator_instruction']}\n")
