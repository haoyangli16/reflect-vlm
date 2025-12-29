"""
Unified VLM API Interface for Latest Vision-Language Models.

Supports:
- OpenAI GPT-5.1 / GPT-4o (vision)
- Google Gemini 3 Pro / Gemini 2.0 Flash
- Alibaba Qwen3-VL-235B

Each VLM can be used for:
1. Action generation (act method)
2. Reflection/principle extraction (reflect method)
3. General generation (generate method)

Usage:
    # For OpenAI
    vlm = UnifiedVLM(provider="openai", model="gpt-5.1")

    # For Gemini
    vlm = UnifiedVLM(provider="gemini", model="gemini-3-pro-preview")

    # For Qwen
    vlm = UnifiedVLM(provider="qwen", model="qwen3-vl-235b-a22b-instruct")

    # Use for action
    action = vlm.act(image, goal_image, prompt)

    # Use for reflection
    principle = vlm.reflect(prompt, image)
"""

from __future__ import annotations

import base64
import json
import os
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================================
# Image Utilities
# ============================================================================


def numpy_to_pil(image: np.ndarray) -> "Image.Image":
    """Convert numpy array to PIL Image."""
    if Image is None:
        raise ImportError("PIL is required for image processing")
    return Image.fromarray(image.astype(np.uint8))


def pil_to_base64(image_pil: "Image.Image", size: tuple = (512, 512), format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    if size:
        image_pil = image_pil.resize(
            size, Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BILINEAR
        )
    buffered = BytesIO()
    image_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_image(
    image: Union[np.ndarray, "Image.Image", str, None], size: tuple = (512, 512)
) -> Optional[str]:
    """Prepare image for API (convert to base64)."""
    if image is None:
        return None
    if isinstance(image, str):
        # Already base64 or URL
        return image
    if isinstance(image, np.ndarray):
        image = numpy_to_pil(image)
    return pil_to_base64(image, size=size)


# ============================================================================
# Base VLM Interface
# ============================================================================


# ============================================================================
# Domain-Specific Prompts for Reflect-VLM Assembly Task
# ============================================================================

# System prompt for action generation - detailed and domain-specific
ACTION_SYSTEM_PROMPT = """You are an expert robot controller for a puzzle assembly task.

## Task Description
You control a robot arm to assemble interlocking puzzle pieces onto a board. The pieces have different colors and MUST be inserted in a specific order due to physical interlocking constraints.

## Available Actions
You can execute ONE of these actions per step:
- `pick up [color]` - Grasp a piece of the specified color from the table
- `put down [color]` - Release the held piece onto the table
- `reorient [color]` - Stand the piece upright (required before insertion if lying flat)
- `insert [color]` - Insert the held piece into its slot on the board
- `done` - Signal task completion (only when ALL pieces are inserted)

## Critical Rules
1. You can only hold ONE piece at a time
2. Some pieces MUST be inserted BEFORE others due to interlocking (predecessor dependencies)
3. A piece lying flat MUST be reoriented before insertion
4. Always check the goal state to determine the correct assembly order

## Output Format
Output ONLY the action command, nothing else. Example: `pick up blue`"""


# Prompt template for action generation with context
ACTION_PROMPT_TEMPLATE = """## Goal State
The first image shows the TARGET configuration you need to achieve.

## Current State  
The second image shows the CURRENT state of the workspace.

## Recent Actions
{action_history}

## Available Pieces
Colors on the table: {available_pieces}

{principles_section}

## Task
Analyze the current state vs goal state and determine the next action.
Think step-by-step:
1. What pieces are already inserted?
2. What piece should be inserted next based on the goal?
3. Is the robot currently holding anything?
4. Does the next piece need reorienting?

Output only the action command."""


# Principles section template
PRINCIPLES_SECTION_TEMPLATE = """## Learned Principles (from past experience)
Apply these rules to avoid common mistakes:
{principles}"""


# System prompt for reflection - detailed guidance for principle extraction
REFLECTION_SYSTEM_PROMPT = """You are an expert robot learning system that extracts reusable principles from failures.

## Your Role
Analyze robot manipulation failures and extract GENERAL RULES that prevent similar failures in the future.

## Analysis Guidelines
1. **Root Cause Analysis**: Identify WHY the action failed, not just what failed
2. **Generalization**: Extract principles that apply to SIMILAR situations, not just this specific case
3. **Actionability**: Principles should tell the robot what to DO or AVOID
4. **Preconditions**: Specify WHEN the principle applies

## Domain Knowledge
This is a puzzle assembly task with interlocking pieces:
- Some pieces physically block others (predecessor/successor relationships)
- Pieces may need reorientation before insertion
- The robot can only hold one piece at a time
- Order of insertion matters due to physical constraints

## Output Format
Always output valid JSON with the specified structure."""


# Enhanced reflection prompt with chain-of-thought
REFLECTION_PROMPT_ENHANCED = """## Failure Analysis

### What Happened
- **Attempted Action**: {failed_action}
- **Failure Type**: {fail_tag}
- **Why It Failed**: {fail_description}

### Oracle Correction
- **Correct Action**: {oracle_action}
- **Why It Works**: The oracle knows the optimal insertion order based on physical constraints.

### Current Context
- **Holding**: {holding_piece}
- **Already Inserted**: {inserted_pieces}
- **Remaining**: {remaining_pieces}
- **Progress**: {progress}%

### Action History
{action_history}

---

## Your Analysis Task

Think through this step-by-step:

1. **What precondition was violated?**
   Consider: Was something blocking? Wrong order? Wrong state?

2. **What's the root cause?**
   Not "the action failed" but WHY it failed.

3. **What general rule prevents this?**
   Should work for ANY similar situation, not just this specific case.
   Avoid mentioning specific colors - use general terms like "piece", "predecessor", "target".

4. **When does this rule apply?**
   What conditions trigger this rule?

Output your analysis as JSON:
```json
{{
    "root_cause": "Brief explanation of why the action failed",
    "general_principle": "A general rule that prevents this type of failure",
    "action_types": ["list", "of", "relevant", "actions"],
    "trigger_conditions": ["conditions", "when", "rule", "applies"],
    "addresses_fail_tags": ["{fail_tag}"]
}}
```"""


class BaseVLM(ABC):
    """Abstract base class for VLM providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt and optional images."""
        pass

    def act(
        self,
        image: Any,
        goal_image: Any,
        prompt: str,
        next_image: Any = None,
        action_history: Optional[List[str]] = None,
        available_pieces: Optional[List[str]] = None,
        principles: Optional[str] = None,
    ) -> str:
        """
        Generate action from observation.

        Args:
            image: Current state image
            goal_image: Goal state image
            prompt: Original prompt (for compatibility)
            next_image: Optional predicted next state
            action_history: List of recent actions
            available_pieces: List of available piece colors
            principles: Formatted principles string

        Returns:
            Action string (e.g., "pick up blue")
        """
        images = [goal_image, image]
        if next_image is not None:
            images.append(next_image)

        # Build context-aware prompt
        history_str = ", ".join(action_history[-5:]) if action_history else "None (first action)"
        pieces_str = ", ".join(available_pieces) if available_pieces else "unknown"

        # Add principles section if available
        if principles:
            principles_section = PRINCIPLES_SECTION_TEMPLATE.format(principles=principles)
        else:
            principles_section = ""

        # Use enhanced prompt or fall back to simple prompt
        if action_history is not None or available_pieces is not None:
            user_prompt = ACTION_PROMPT_TEMPLATE.format(
                action_history=history_str,
                available_pieces=pieces_str,
                principles_section=principles_section,
            )
        else:
            # Fallback to original prompt for backward compatibility
            user_prompt = prompt + "\n\nOutput ONLY the action command (e.g., 'pick up blue')."

        response = self.generate(
            prompt=user_prompt,
            images=images,
            system_prompt=ACTION_SYSTEM_PROMPT,
            max_tokens=50,
            temperature=0.1,
        )

        # Clean response - extract action from potential reasoning
        action = self._extract_action(response)
        return action

    def _extract_action(self, response: str) -> str:
        """Extract action from VLM response, handling various output formats."""
        response = response.strip()

        # Handle multi-line responses - look for action patterns
        lines = response.split("\n")
        for line in lines:
            line = line.strip().lower()
            # Skip reasoning lines
            if any(skip in line for skip in ["think", "step", "because", "since", "note"]):
                continue
            # Check for action patterns
            for action_type in ["pick up", "put down", "insert", "reorient", "done"]:
                if line.startswith(action_type):
                    return line

        # Fallback: clean the first line
        action = lines[0].strip().lower()

        # Remove common prefixes
        prefixes = [
            "action:",
            "the action is:",
            "next action:",
            "i will",
            "robot action:",
            "execute:",
            "output:",
            "answer:",
        ]
        for prefix in prefixes:
            if action.startswith(prefix):
                action = action[len(prefix) :].strip()

        # Remove backticks or quotes
        action = action.strip("`'\"")

        return action

    def reflect(
        self,
        prompt: str,
        image: Any = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate reflection/principle from failure analysis."""
        images = [image] if image is not None else None

        return self.generate(
            prompt=prompt,
            images=images,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=temperature,
        )

    def reflect_on_failure(
        self,
        failed_action: str,
        fail_tag: str,
        fail_description: str,
        oracle_action: str,
        holding_piece: str,
        inserted_pieces: List[str],
        remaining_pieces: List[str],
        progress: float,
        action_history: List[str],
        image: Any = None,
        temperature: float = 0.5,
    ) -> str:
        """
        Generate structured reflection on a specific failure.

        Uses chain-of-thought prompting for better principle extraction.
        """
        prompt = REFLECTION_PROMPT_ENHANCED.format(
            failed_action=failed_action,
            fail_tag=fail_tag,
            fail_description=fail_description,
            oracle_action=oracle_action,
            holding_piece=holding_piece or "Nothing",
            inserted_pieces=", ".join(inserted_pieces) if inserted_pieces else "None",
            remaining_pieces=", ".join(remaining_pieces) if remaining_pieces else "None",
            progress=int(progress * 100),
            action_history="\n".join(f"  {i + 1}. {a}" for i, a in enumerate(action_history[-5:]))
            if action_history
            else "  No previous actions",
        )

        return self.reflect(prompt, image, temperature)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat interface for compatibility."""
        # Extract system and user messages
        system_prompt = None
        user_content = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                user_content.append(content)

        return self.generate(
            prompt="\n".join(user_content),
            system_prompt=system_prompt,
        )


# ============================================================================
# OpenAI GPT Implementation
# ============================================================================


class OpenAIVLM(BaseVLM):
    """
    OpenAI GPT Vision models (GPT-4o, GPT-5.1, etc.)

    Requires: openai>=1.0.0
    API Key: OPENAI_API_KEY environment variable

    Thinking/Reasoning mode:
    - Uses client.responses.create() API with reasoning={"effort": "medium/high"}
    - Only supported by GPT-5 series, o1, o3 models (NOT GPT-4o)
    - See: https://platform.openai.com/docs/guides/reasoning
    """

    # Models that support the reasoning API
    REASONING_MODELS = ["gpt-5", "gpt-5.1", "gpt-5.2", "o1", "o3", "o4-mini"]

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = True,
        reasoning_effort: str = "medium",  # "none", "low", "medium", "high"
        **kwargs,
    ):
        super().__init__(model, api_key, enable_thinking=enable_thinking, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.reasoning_effort = reasoning_effort

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai>=1.0.0")

        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        # Check if model supports reasoning
        self._supports_reasoning = any(
            self.model.startswith(prefix) for prefix in self.REASONING_MODELS
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        # Use reasoning API for thinking mode with supported models
        if self.enable_thinking and self._supports_reasoning:
            return self._generate_with_reasoning(prompt, images, system_prompt, max_tokens)
        else:
            return self._generate_chat(prompt, images, system_prompt, max_tokens, temperature)

    def _generate_with_reasoning(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate using the responses API with reasoning effort."""
        try:
            # Build input with images if present
            if images:
                # For responses API with images, build content list
                input_content = []
                for img in images:
                    if img is not None:
                        img_b64 = prepare_image(img)
                        input_content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{img_b64}",
                            }
                        )
                input_content.append({"type": "input_text", "text": prompt})

                # Add system prompt as instructions if provided
                instructions = system_prompt if system_prompt else None

                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    instructions=instructions,
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=max_tokens,
                )
            else:
                # Text-only input
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                response = self.client.responses.create(
                    model=self.model,
                    input=full_prompt,
                    reasoning={"effort": self.reasoning_effort},
                    max_output_tokens=max_tokens,
                )

            # Extract text from response
            return (
                response.output_text.strip() if hasattr(response, "output_text") else str(response)
            )

        except Exception as e:
            print(f"[OpenAI Reasoning] API error: {e}")
            # Fallback to chat API
            return self._generate_chat(prompt, images, system_prompt, max_tokens, 0.7)

    def _generate_chat(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Generate using the standard chat completions API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message with images
        user_content = []

        if images:
            for img in images:
                if img is not None:
                    img_b64 = prepare_image(img)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            # GPT-5 series uses max_completion_tokens, older models use max_tokens
            if self._supports_reasoning:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            # Provide more helpful error messages
            if (
                "401" in error_str
                or "Invalid Authentication" in error_str
                or "Unauthorized" in error_str
            ):
                print(
                    f"[OpenAI] API authentication error: {e}\n"
                    f"  Check that your API key is correct and has proper permissions.\n"
                    f"  Base URL: {getattr(self.client, 'base_url', 'default')}"
                )
            elif "404" in error_str or "Not Found" in error_str:
                print(
                    f"[OpenAI] API endpoint not found: {e}\n"
                    f"  Check that the base URL and model name are correct.\n"
                    f"  Base URL: {getattr(self.client, 'base_url', 'default')}\n"
                    f"  Model: {self.model}"
                )
            else:
                print(f"[OpenAI] API error: {e}")
            return ""


# ============================================================================
# Google Gemini Implementation
# ============================================================================


class GeminiVLM(BaseVLM):
    """
    Google Gemini Vision models (Gemini 3 Pro, Gemini 2.5 Flash, etc.)

    Requires: google-genai>=0.1.0
    API Key: GOOGLE_API_KEY environment variable

    Thinking mode:
    - Uses thinking_config with thinking_budget (0 to disable, -1 for dynamic, or token count)
    - See: https://ai.google.dev/gemini-api/docs/thinking
    """

    def __init__(
        self,
        model: str = "gemini-3-pro-preview",
        api_key: Optional[str] = None,
        enable_thinking: bool = True,
        thinking_level: str = "medium",  # For Gemini 3: "low", "medium", "high", "minimal" (Flash only)
        thinking_budget: int = 1024,  # For Gemini 2.5: 0 to disable, -1 for dynamic
        **kwargs,
    ):
        super().__init__(model, api_key, enable_thinking=enable_thinking, **kwargs)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or pass api_key.")

        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget

        try:
            from google import genai
            from google.genai import types

            self._types = types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: pip install google-genai"
            )

        self.client = genai.Client(api_key=self.api_key)
        self._genai = genai

        # Determine if this is a Gemini 3 or 2.5 model
        self._is_gemini3 = "gemini-3" in model.lower()

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        contents = []

        # Add system prompt as first text
        if system_prompt:
            contents.append(system_prompt + "\n\n")

        # Add images
        if images:
            for img in images:
                if img is not None:
                    if isinstance(img, np.ndarray):
                        img = numpy_to_pil(img)
                    elif isinstance(img, str):
                        # Base64 to PIL
                        img_data = base64.b64decode(img)
                        img = Image.open(BytesIO(img_data))
                    # Resize for API
                    img = img.resize(
                        (512, 512), Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BILINEAR
                    )
                    contents.append(img)

        # Add prompt
        contents.append(prompt)

        try:
            # Build config with thinking settings if enabled
            if self.enable_thinking:
                # google-genai SDK uses thinking_budget for all models
                # Set to -1 for dynamic thinking (model decides)
                thinking_config = self._types.ThinkingConfig(
                    thinking_budget=self.thinking_budget,
                    include_thoughts=True,
                )

                config = self._types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=thinking_config,
                )
            else:
                config = self._types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # When thinking is enabled, find the non-thought part for the answer
            if self.enable_thinking and hasattr(response, "candidates"):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and not part.thought:
                        return part.text.strip() if part.text else ""
                    elif not hasattr(part, "thought") and hasattr(part, "text"):
                        return part.text.strip()

            return response.text.strip() if hasattr(response, "text") else str(response)
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return ""


# ============================================================================
# Alibaba Qwen VL Implementation
# ============================================================================


class QwenVLM(BaseVLM):
    """
    Alibaba Qwen Vision-Language models (Qwen3-VL, Qwen2-VL, etc.)

    Uses OpenAI-compatible API with DashScope endpoint.
    Reference: https://github.com/QwenLM/Qwen

    API Key: DASHSCOPE_API_KEY or QWEN_API_KEY environment variable
    Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1

    Thinking mode: Qwen3 natively supports thinking mode via enable_thinking parameter.
    """

    # DashScope OpenAI-compatible endpoint
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str = "qwen3-vl-235b-a22b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        super().__init__(model, api_key, enable_thinking=enable_thinking, **kwargs)

        # Get API key from environment
        self.api_key = (
            api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "DashScope API key required. Set DASHSCOPE_API_KEY or QWEN_API_KEY env var."
            )

        # Use DashScope OpenAI-compatible endpoint by default
        self.base_url = base_url or self.DASHSCOPE_BASE_URL

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai>=1.0.0")

    def generate(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using OpenAI-compatible DashScope API.

        Uses the same format as OpenAI API with image_url type.
        Reference: https://github.com/QwenLM/Qwen
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user content with images (OpenAI format)
        user_content = []

        if images:
            for img in images:
                if img is not None:
                    img_b64 = prepare_image(img)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        }
                    )

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            # Qwen3 natively supports thinking mode via enable_thinking parameter
            extra_body = {"enable_thinking": self.enable_thinking}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Qwen] API error: {e}")
            return ""


# ============================================================================
# Kimi (Moonshot AI) Implementation
# ============================================================================


class KimiVLM(OpenAIVLM):
    """
    Kimi (Moonshot AI) Vision models.

    Compatible with OpenAI API.
    API Key: MOONSHOT_API_KEY or KIMI_API_KEY environment variable
    Base URL: https://api.moonshot.ai/v1 (NOTE: .ai not .cn!)

    Reference: https://platform.moonshot.ai/docs/api/chat
    """

    def __init__(
        self,
        model: str = "kimi-k2-0905-preview",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Check for API key in multiple env vars
        api_key = api_key or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise ValueError(
                "Moonshot/Kimi API key required. Set MOONSHOT_API_KEY or KIMI_API_KEY env var or pass api_key."
            )

        # IMPORTANT: Default base URL for Moonshot API is .ai NOT .cn!
        # See: https://platform.moonshot.ai/docs/api/chat
        if base_url is None:
            base_url = "https://api.moonshot.ai/v1"

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)


# ============================================================================
# HuggingFace Router Implementation (for gpt-oss-120b etc.)
# ============================================================================


class HuggingFaceVLM(OpenAIVLM):
    """
    HuggingFace Router for open-source models like gpt-oss-120b.

    Uses the HuggingFace router API which is OpenAI-compatible.
    API Key: HF_TOKEN environment variable
    Base URL: https://router.huggingface.co/v1

    Example models:
        - openai/gpt-oss-120b:cerebras
    """

    HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b:cerebras",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Get API key from environment
        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "HuggingFace API key required. Set HF_TOKEN or HUGGINGFACE_API_KEY env var."
            )

        # Use HuggingFace router endpoint
        if base_url is None:
            base_url = self.HUGGINGFACE_BASE_URL

        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)


# ============================================================================
# Unified VLM Factory
# ============================================================================


class UnifiedVLM:
    """
    Unified interface for all VLM providers.

    Automatically routes to the appropriate backend based on provider.

    Usage:
        vlm = UnifiedVLM(provider="openai", model="gpt-5.1")
        vlm = UnifiedVLM(provider="gemini", model="gemini-3-pro-preview")
        vlm = UnifiedVLM(provider="qwen", model="qwen3-vl-235b-a22b-instruct")
        vlm = UnifiedVLM(provider="kimi", model="moonshot-v1-8k-vision-preview")

        # With thinking mode enabled
        vlm = UnifiedVLM(provider="qwen", model="qwen3-vl-235b-a22b-instruct", enable_thinking=True)

        # OpenAI with reasoning effort
        vlm = UnifiedVLM(provider="openai", model="gpt-5.1", enable_thinking=True, reasoning_effort="medium")

        # Gemini with thinking budget
        vlm = UnifiedVLM(provider="gemini", model="gemini-2.5-flash", enable_thinking=True, thinking_budget=1024)

    Thinking Mode Support:
        - Qwen: Native support via enable_thinking parameter in API
        - Gemini: Uses thinking_config with thinking_budget (0 to disable, -1 for dynamic, or token count)
        - OpenAI/GPT: Uses responses API with reasoning={"effort": ...} (GPT-5 series, o1, o3 only)
    """

    PROVIDERS = {
        "openai": OpenAIVLM,
        "gpt": OpenAIVLM,
        "gemini": GeminiVLM,
        "google": GeminiVLM,
        "qwen": QwenVLM,
        "alibaba": QwenVLM,
        "dashscope": QwenVLM,
        "kimi": KimiVLM,
        "moonshot": KimiVLM,
        "huggingface": HuggingFaceVLM,
        "hf": HuggingFaceVLM,
    }

    # Default models for each provider
    DEFAULT_MODELS = {
        "openai": "gpt-5.1",
        "gpt": "gpt-5.1",
        "gemini": "gemini-3-flash-preview",
        "google": "gemini-3-flash-preview",
        # Use the model that works in your test file
        "qwen": "qwen3-vl-flash",  # "qwen3-vl-flash",
        "alibaba": "qwen3-vl-flash",
        "dashscope": "qwen3-vl-flash",
        "kimi": "kimi-k2-0905-preview",
        "moonshot": "kimi-k2-0905-preview",
        "huggingface": "openai/gpt-oss-120b:cerebras",
        "hf": "openai/gpt-oss-120b:cerebras",
    }

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = True,
        **kwargs,
    ):
        provider = provider.lower()
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. Supported: {list(self.PROVIDERS.keys())}"
            )

        model = model or self.DEFAULT_MODELS[provider]
        vlm_class = self.PROVIDERS[provider]

        self._vlm = vlm_class(
            model=model, api_key=api_key, enable_thinking=enable_thinking, **kwargs
        )
        self.provider = provider
        self.model = model
        self.enable_thinking = enable_thinking

    def generate(self, prompt: str, images: Optional[List[Any]] = None, **kwargs) -> str:
        return self._vlm.generate(prompt, images, **kwargs)

    def act(self, image: Any, goal_image: Any, prompt: str, **kwargs) -> str:
        return self._vlm.act(image, goal_image, prompt, **kwargs)

    def reflect(self, prompt: str, image: Any = None, **kwargs) -> str:
        return self._vlm.reflect(prompt, image, **kwargs)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self._vlm.chat(messages)

    def __repr__(self) -> str:
        return f"UnifiedVLM(provider={self.provider}, model={self.model})"


# ============================================================================
# Convenience functions
# ============================================================================


def create_vlm(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> UnifiedVLM:
    """
    Create a VLM instance.

    Args:
        provider: "openai", "gemini", "qwen", or "kimi"
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments (api_key, base_url, etc.)

    Returns:
        UnifiedVLM instance
    """
    return UnifiedVLM(provider=provider, model=model, **kwargs)


def get_available_models() -> Dict[str, List[str]]:
    """Get list of recommended models for each provider."""
    return {
        "openai": ["gpt-5.2", "gpt-5.1", "o4-mini-2025-04-16"],
        "gemini": [
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
        ],
        "qwen": [
            "qwen3-vl-235b-a22b-instruct",
            "qwen3-vl-flash",
            "qwen3-vl-plus",
            "qwen-vl-max",
            "qwen-vl-plus",
        ],
        "kimi": [
            "kimi-k2-0905-preview",
            "kimi-k2-0711-preview",
            "kimi-k2-turbo-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
        ],
        "huggingface": [
            "openai/gpt-oss-120b:cerebras",
        ],
    }


# ============================================================================
# Test function
# ============================================================================


# Convenience export for checking available providers
SUPPORTED_PROVIDERS = UnifiedVLM.PROVIDERS  # Dict mapping provider name -> class


def test_vlm(provider: str = "openai", model: Optional[str] = None):
    """Quick test of VLM API."""
    print(f"\nTesting {provider} VLM...")

    try:
        vlm = create_vlm(provider=provider, model=model)
        print(f"  Created: {vlm}")

        # Test text-only generation
        response = vlm.generate("What is 2 + 2? Answer with just the number.")
        print(f"  Text test: '{response}'")

        print(f"  ✅ {provider} VLM working!")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    model = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 50)
    print("VLM API Test")
    print("=" * 50)
    print(f"\nAvailable models: {json.dumps(get_available_models(), indent=2)}")

    test_vlm(provider, model)
