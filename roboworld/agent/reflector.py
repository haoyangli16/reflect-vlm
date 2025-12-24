"""
Reflector: VLM-based reflection for principle extraction.

Inspired by ExpeL but adapted for embodied VLM agents with:
- Visual observation context
- Oracle-guided reflection (we know the correct action)
- Structured failure tags
- Assembly-specific domain knowledge

The reflection pipeline:
1. Detect failure (from fail_tag)
2. Extract context (symbolic state, action history)
3. Generate reflection using VLM
4. Parse reflection into structured principle
5. Update principle store (add/upvote/downvote/edit)

Supported VLM backends:
- OpenAI GPT-5.1 / GPT-4o
- Google Gemini 3 Pro
- Alibaba Qwen3-VL-235B
- Rule-based fallback (no API needed)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import unified VLM API
try:
    from roboworld.agent.vlm_api import UnifiedVLM, create_vlm

    HAS_VLM_API = True
except ImportError:
    HAS_VLM_API = False
    UnifiedVLM = None
    create_vlm = None


# ============================================================================
# Reflection Prompts
# ============================================================================

REFLECTION_SYSTEM_PROMPT = """You are an expert robot learning system that extracts reusable principles from failures.

Your task is to analyze robot manipulation failures and extract GENERAL RULES that can prevent similar failures in the future.

Key guidelines:
1. Focus on the ROOT CAUSE, not the symptom
2. Extract GENERAL principles that transfer to new situations
3. Be SPECIFIC about when the rule applies
4. Consider DEPENDENCIES and ORDERING constraints
5. Think about PHYSICAL constraints (stability, reachability, blocking)

Remember: A good principle helps avoid the SAME TYPE of failure in DIFFERENT situations."""


REFLECTION_PROMPT_TEMPLATE = """
## Failure Analysis Task

A robot attempted an action that FAILED. Analyze why and extract a reusable principle.

### Failed Action
- **Action Taken**: {failed_action}
- **Failure Type**: {fail_tag}
- **Failure Description**: {fail_description}

### Correct Action (from Oracle)
- **Should Have Done**: {oracle_action}

### Current State
- **Holding Object**: {holding_piece}
- **Inserted Pieces**: {inserted_pieces}
- **Remaining Pieces**: {remaining_pieces}
- **Progress**: {progress}%

### Action History
{action_history}

### Your Analysis

Please provide:

1. **ROOT CAUSE**: Why did the action fail? (1-2 sentences)

2. **GENERAL PRINCIPLE**: What rule would prevent this type of failure?
   - Should be GENERAL (not specific to colors/names)
   - Should be ACTIONABLE (tells the robot what to do/avoid)
   - Should specify WHEN it applies

3. **APPLICABILITY**: When should this principle be applied?
   - Action types: (pick, insert, reorient, putdown)
   - Preconditions: (what must be true for this rule to apply)

Output your analysis in this JSON format:
```json
{{
    "root_cause": "...",
    "general_principle": "...",
    "action_types": ["insert", ...],
    "trigger_conditions": ["has_dependencies", ...],
    "addresses_fail_tags": ["{fail_tag}"]
}}
```
"""


CONTRASTIVE_REFLECTION_PROMPT = """
## Contrastive Analysis Task

Compare a FAILED attempt with a SUCCESSFUL attempt for the same goal.

### Failed Attempt
- **Action**: {failed_action}
- **Result**: FAILED - {fail_tag}
- **State**: Holding={holding_piece}, Inserted={inserted_pieces}

### Successful Attempt (Oracle Path)
- **Action**: {oracle_action}
- **Result**: Would SUCCEED
- **Why**: {oracle_explanation}

### Key Question
What did the FAILED attempt do wrong that the SUCCESSFUL path avoided?

Extract a GENERAL PRINCIPLE that explains the difference.

Output in JSON format:
```json
{{
    "difference": "The failed attempt tried to ... while the successful path first ...",
    "general_principle": "Before doing X, always ensure Y",
    "action_types": [...],
    "trigger_conditions": [...],
    "addresses_fail_tags": [...]
}}
```
"""


# Failure tag descriptions for better context
FAIL_TAG_DESCRIPTIONS = {
    "BLOCKED_BY_PREDECESSOR": "The piece cannot be inserted because a predecessor piece (that must be inserted first due to interlocking) has not been inserted yet.",
    "BLOCKED_BY_SUCCESSOR": "The piece cannot be inserted because a successor piece is already in the way, blocking the slot.",
    "BAD_PLACEMENT": "The piece was placed incorrectly, blocking other pieces from being inserted.",
    "NEEDS_REORIENT": "The piece is lying flat and needs to be reoriented (stood up) before insertion.",
    "HAND_FULL": "Cannot pick up a new object because the robot is already holding something.",
    "EMPTY_HAND": "Cannot put down or insert because the robot is not holding anything.",
    "GRASP_FAILED": "The grasp action failed - object not graspable due to geometry or reach issues.",
    "INSERT_TIMEOUT": "The insertion took too long, likely due to alignment or precision issues.",
    "BAD_DONE": "Called 'done' but the task is not actually complete.",
    "REDUNDANT_PICKUP": "Tried to pick up an object that is already in hand.",
    "NOT_HOLDING_FOR_INSERT": "Tried to insert but not holding the object.",
    "NOT_HOLDING_FOR_REORIENT": "Tried to reorient but not holding the object.",
    "PARSE_ERROR": "Could not parse the action format.",
    "UNKNOWN_OBJECT": "Referenced an object that doesn't exist in the environment.",
}


@dataclass
class ReflectionInput:
    """Input data for generating a reflection."""

    failed_action: str
    fail_tag: str
    oracle_action: Optional[str]
    symbolic_state: Dict[str, Any]
    action_history: List[str]
    experience_id: str

    # Optional: visual observation for VLM
    image: Optional[np.ndarray] = None


@dataclass
class ReflectionOutput:
    """Structured output from reflection."""

    root_cause: str
    general_principle: str
    action_types: List[str]
    trigger_conditions: List[str]
    addresses_fail_tags: List[str]
    confidence: float = 0.5

    # For principle store update
    operation: str = "ADD"  # ADD, UPVOTE, DOWNVOTE, EDIT
    principle_id: Optional[str] = None  # For UPVOTE/DOWNVOTE/EDIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "principle_id": self.principle_id,
            "content": self.general_principle,
            "action_types": self.action_types,
            "trigger_conditions": self.trigger_conditions,
            "addresses_fail_tags": self.addresses_fail_tags,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
        }


class Reflector:
    """
    VLM-based reflector for extracting principles from failures.

    Supports multiple reflection modes:
    - oracle_guided: Uses oracle_action as ground truth (best quality)
    - contrastive: Compares failed vs successful trajectories
    - self_reflection: Pure VLM reflection without oracle

    Supports multiple VLM backends:
    - OpenAI GPT-5.1 / GPT-4o
    - Google Gemini 3 Pro
    - Alibaba Qwen3-VL-235B
    - Rule-based fallback (no API needed)

    Usage:
        # With VLM API
        reflector = Reflector.create(provider="openai", model="gpt-5.1")

        # With existing agent
        reflector = Reflector(vlm_agent=my_agent)

        # Without VLM (uses rule-based fallback)
        reflector = Reflector()
    """

    def __init__(
        self,
        vlm_agent=None,
        mode: str = "oracle_guided",
        use_visual: bool = True,
        verbose: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize reflector.

        Args:
            vlm_agent: VLM agent with .generate() or similar method (optional)
            mode: Reflection mode - "oracle_guided", "contrastive", "self_reflection"
            use_visual: Whether to include visual observations in reflection
            verbose: Print debug information
            provider: VLM provider ("openai", "gemini", "qwen") - creates VLM if specified
            model: VLM model name (uses default if not specified)
            api_key: API key for the VLM provider
        """
        # If provider specified, create VLM agent
        if provider and vlm_agent is None:
            if not HAS_VLM_API:
                raise ImportError(
                    "VLM API not available. Install required packages or use vlm_agent parameter."
                )
            vlm_agent = create_vlm(provider=provider, model=model, api_key=api_key)
            if verbose:
                print(f"[Reflector] Created VLM: {vlm_agent}")

        self.vlm_agent = vlm_agent
        self.mode = mode
        self.use_visual = use_visual
        self.verbose = verbose
        self.provider = provider
        self.model = model

        # Track VLM usage
        self._vlm_call_count = 0
        self._fallback_count = 0

        # Cache for batch processing
        self._reflection_cache: Dict[str, ReflectionOutput] = {}

    @classmethod
    def create(
        cls,
        provider: str = "openai",
        model: Optional[str] = None,
        mode: str = "oracle_guided",
        use_visual: bool = True,
        verbose: bool = False,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "Reflector":
        """
        Factory method to create Reflector with VLM backend.

        Args:
            provider: VLM provider - "openai", "gemini", "qwen"
            model: Model name (uses default if not specified)
                   - OpenAI: "gpt-5.1", "gpt-4o", "gpt-4-turbo"
                   - Gemini: "gemini-3-pro-preview", "gemini-2.0-flash"
                   - Qwen: "qwen3-vl-235b-a22b-instruct", "qwen-vl-max"
            mode: Reflection mode
            use_visual: Include visual observations
            verbose: Debug output
            api_key: API key (uses env var if not specified)

        Returns:
            Reflector instance with VLM backend

        Example:
            reflector = Reflector.create(provider="gemini", model="gemini-3-pro-preview")
            output = reflector.reflect(input_data)
        """
        if not HAS_VLM_API:
            raise ImportError(
                "VLM API not available. Please install required packages:\n"
                "  pip install openai  # for OpenAI\n"
                "  pip install google-genai  # for Gemini\n"
                "  pip install dashscope  # for Qwen"
            )

        vlm = create_vlm(provider=provider, model=model, api_key=api_key, **kwargs)

        return cls(
            vlm_agent=vlm,
            mode=mode,
            use_visual=use_visual,
            verbose=verbose,
            provider=provider,
            model=model,
        )

    @classmethod
    def create_openai(cls, model: str = "gpt-5.1", **kwargs) -> "Reflector":
        """Create Reflector with OpenAI backend."""
        return cls.create(provider="openai", model=model, **kwargs)

    @classmethod
    def create_gemini(cls, model: str = "gemini-3-pro-preview", **kwargs) -> "Reflector":
        """Create Reflector with Gemini backend."""
        return cls.create(provider="gemini", model=model, **kwargs)

    @classmethod
    def create_qwen(cls, model: str = "qwen3-vl-235b-a22b-instruct", **kwargs) -> "Reflector":
        """Create Reflector with Qwen backend."""
        return cls.create(provider="qwen", model=model, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "vlm_calls": self._vlm_call_count,
            "fallback_calls": self._fallback_count,
            "cache_size": len(self._reflection_cache),
            "provider": self.provider,
            "model": self.model,
            "has_vlm": self.vlm_agent is not None,
        }

    def reflect(self, input_data: ReflectionInput) -> ReflectionOutput:
        """
        Generate reflection from failure.

        Args:
            input_data: ReflectionInput with failure context

        Returns:
            ReflectionOutput with extracted principle
        """
        # Check cache
        cache_key = f"{input_data.failed_action}_{input_data.fail_tag}"
        if cache_key in self._reflection_cache:
            if self.verbose:
                print(f"[Reflector] Cache hit for {cache_key}")
            return self._reflection_cache[cache_key]

        # Choose reflection mode
        if self.mode == "oracle_guided" and input_data.oracle_action:
            output = self._reflect_oracle_guided(input_data)
        elif self.mode == "contrastive":
            output = self._reflect_contrastive(input_data)
        else:
            output = self._reflect_self(input_data)

        # Cache result
        self._reflection_cache[cache_key] = output

        return output

    def _reflect_oracle_guided(self, input_data: ReflectionInput) -> ReflectionOutput:
        """
        Reflection using oracle action as ground truth.

        This is the most reliable mode since we know the correct action.
        """
        # Build prompt
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            failed_action=input_data.failed_action,
            fail_tag=input_data.fail_tag,
            fail_description=FAIL_TAG_DESCRIPTIONS.get(input_data.fail_tag, "Unknown failure type"),
            oracle_action=input_data.oracle_action or "Unknown",
            holding_piece=input_data.symbolic_state.get("holding_piece", "None"),
            inserted_pieces=", ".join(input_data.symbolic_state.get("inserted_pieces", []))
            or "None",
            remaining_pieces=", ".join(input_data.symbolic_state.get("remaining_pieces", []))
            or "None",
            progress=int(input_data.symbolic_state.get("progress", 0) * 100),
            action_history=self._format_action_history(input_data.action_history),
        )

        # Generate reflection
        if self.vlm_agent:
            response = self._call_vlm(prompt, input_data.image)
        else:
            # Fallback: rule-based reflection
            response = self._rule_based_reflection(input_data)

        # Parse response
        output = self._parse_reflection_response(response, input_data)

        return output

    def _reflect_contrastive(self, input_data: ReflectionInput) -> ReflectionOutput:
        """
        Reflection by contrasting failed vs oracle path.
        """
        # Generate explanation for why oracle action would work
        oracle_explanation = self._explain_oracle_action(
            input_data.oracle_action,
            input_data.fail_tag,
            input_data.symbolic_state,
        )

        prompt = CONTRASTIVE_REFLECTION_PROMPT.format(
            failed_action=input_data.failed_action,
            fail_tag=input_data.fail_tag,
            holding_piece=input_data.symbolic_state.get("holding_piece", "None"),
            inserted_pieces=", ".join(input_data.symbolic_state.get("inserted_pieces", []))
            or "None",
            oracle_action=input_data.oracle_action or "Unknown",
            oracle_explanation=oracle_explanation,
        )

        if self.vlm_agent:
            response = self._call_vlm(prompt, input_data.image)
        else:
            response = self._rule_based_reflection(input_data)

        return self._parse_reflection_response(response, input_data)

    def _reflect_self(self, input_data: ReflectionInput) -> ReflectionOutput:
        """
        Self-reflection without oracle guidance.

        Less reliable but useful when oracle is not available.
        """
        prompt = f"""
The robot tried: {input_data.failed_action}
Result: FAILED with error {input_data.fail_tag}

Based on this failure, what general principle should the robot learn?
Think about what preconditions were likely violated.

Output JSON with: root_cause, general_principle, action_types, trigger_conditions, addresses_fail_tags
"""

        if self.vlm_agent:
            response = self._call_vlm(prompt, input_data.image)
        else:
            response = self._rule_based_reflection(input_data)

        return self._parse_reflection_response(response, input_data)

    def _call_vlm(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        """Call VLM to generate reflection."""
        self._vlm_call_count += 1

        try:
            # Check for UnifiedVLM (our new interface)
            if hasattr(self.vlm_agent, "reflect"):
                # Use the dedicated reflect method
                if image is not None and self.use_visual:
                    result = self.vlm_agent.reflect(prompt, image=image)
                else:
                    result = self.vlm_agent.reflect(prompt)
                if self.verbose:
                    print(f"[Reflector] VLM response ({len(result)} chars)")
                return result

            # Try generate with images list (standard interface)
            if hasattr(self.vlm_agent, "generate"):
                if image is not None and self.use_visual:
                    # Try with images as list
                    try:
                        return self.vlm_agent.generate(
                            prompt,
                            images=[image],
                            system_prompt=REFLECTION_SYSTEM_PROMPT,
                        )
                    except TypeError:
                        # Fallback: try with image= parameter
                        return self.vlm_agent.generate(prompt, image=image)
                return self.vlm_agent.generate(prompt, system_prompt=REFLECTION_SYSTEM_PROMPT)

            elif hasattr(self.vlm_agent, "chat"):
                messages = [
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                return self.vlm_agent.chat(messages)

            elif hasattr(self.vlm_agent, "__call__"):
                return self.vlm_agent(prompt)

            else:
                if self.verbose:
                    print("[Reflector] Warning: Unknown VLM interface, using fallback")
                return ""

        except Exception as e:
            if self.verbose:
                print(f"[Reflector] VLM call failed: {e}")
            return ""

    def _rule_based_reflection(self, input_data: ReflectionInput) -> str:
        """
        Fallback rule-based reflection when VLM is not available.

        Uses domain knowledge about assembly tasks.
        """
        self._fallback_count += 1
        if self.verbose:
            print(f"[Reflector] Using rule-based fallback for {input_data.fail_tag}")

        fail_tag = input_data.fail_tag
        failed_action = input_data.failed_action
        oracle_action = input_data.oracle_action or ""

        # Domain-specific rules for assembly task
        rules = {
            "BLOCKED_BY_PREDECESSOR": {
                "root_cause": "Attempted to insert a piece before its predecessor pieces were inserted. In interlocking assemblies, some pieces must be inserted in a specific order.",
                "general_principle": "Before inserting any piece, verify that all its predecessor pieces (pieces that must be inserted first due to interlocking constraints) are already inserted.",
                "action_types": ["insert"],
                "trigger_conditions": ["piece_has_dependencies", "attempting_insert"],
            },
            "BLOCKED_BY_SUCCESSOR": {
                "root_cause": "A successor piece is already in the board, blocking the target piece's slot.",
                "general_principle": "If a piece cannot be inserted because a successor is blocking, first remove or reposition the blocking piece.",
                "action_types": ["insert", "putdown"],
                "trigger_conditions": ["successor_in_board", "slot_blocked"],
            },
            "NEEDS_REORIENT": {
                "root_cause": "The piece is lying flat and cannot be inserted in this orientation.",
                "general_principle": "Before inserting a piece, check if it needs to be reoriented (stood upright). If the piece is lying down, reorient it first.",
                "action_types": ["insert", "reorient"],
                "trigger_conditions": ["piece_not_upright", "attempting_insert"],
            },
            "HAND_FULL": {
                "root_cause": "Tried to pick up a new object while already holding something.",
                "general_principle": "Before picking up a new object, ensure your gripper is empty. If holding something, put it down or insert it first.",
                "action_types": ["pick"],
                "trigger_conditions": ["gripper_occupied", "attempting_pickup"],
            },
            "GRASP_FAILED": {
                "root_cause": "The grasp failed due to object geometry, position, or reachability issues.",
                "general_principle": "If a grasp fails, consider approaching the object from a different angle or repositioning before retrying.",
                "action_types": ["pick"],
                "trigger_conditions": ["grasp_attempted", "object_not_graspable"],
            },
            "BAD_DONE": {
                "root_cause": "Declared the task complete when it was not actually finished.",
                "general_principle": "Only declare 'done' when ALL pieces have been successfully inserted. Verify completion by checking that no remaining pieces exist.",
                "action_types": ["done"],
                "trigger_conditions": ["pieces_remaining", "task_incomplete"],
            },
        }

        if fail_tag in rules:
            rule = rules[fail_tag]
            return json.dumps(
                {
                    "root_cause": rule["root_cause"],
                    "general_principle": rule["general_principle"],
                    "action_types": rule["action_types"],
                    "trigger_conditions": rule["trigger_conditions"],
                    "addresses_fail_tags": [fail_tag],
                }
            )

        # Generic fallback
        return json.dumps(
            {
                "root_cause": f"Action '{failed_action}' failed with error '{fail_tag}'",
                "general_principle": f"When encountering '{fail_tag}', consider checking preconditions before attempting the action.",
                "action_types": [self._extract_action_type(failed_action)],
                "trigger_conditions": ["action_precondition_violated"],
                "addresses_fail_tags": [fail_tag],
            }
        )

    def _parse_reflection_response(
        self,
        response: str,
        input_data: ReflectionInput,
    ) -> ReflectionOutput:
        """Parse VLM response into structured ReflectionOutput."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReflectionOutput(
                    root_cause=data.get("root_cause", "Unknown"),
                    general_principle=data.get("general_principle", ""),
                    action_types=data.get("action_types", []),
                    trigger_conditions=data.get("trigger_conditions", []),
                    addresses_fail_tags=data.get("addresses_fail_tags", [input_data.fail_tag]),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: extract key information from text
        return ReflectionOutput(
            root_cause=f"Failed action: {input_data.failed_action}",
            general_principle=self._extract_principle_from_text(response),
            action_types=[self._extract_action_type(input_data.failed_action)],
            trigger_conditions=[],
            addresses_fail_tags=[input_data.fail_tag],
        )

    def _format_action_history(self, history: List[str]) -> str:
        """Format action history for prompt."""
        if not history:
            return "No previous actions"

        # Show last 5 actions
        recent = history[-5:]
        lines = [f"  {i + 1}. {action}" for i, action in enumerate(recent)]

        if len(history) > 5:
            lines.insert(0, f"  ... ({len(history) - 5} earlier actions)")

        return "\n".join(lines)

    def _explain_oracle_action(
        self,
        oracle_action: Optional[str],
        fail_tag: str,
        symbolic_state: Dict[str, Any],
    ) -> str:
        """Generate explanation for why oracle action would succeed."""
        if not oracle_action:
            return "Oracle action unknown"

        # Domain-specific explanations
        if fail_tag == "BLOCKED_BY_PREDECESSOR":
            return f"'{oracle_action}' would insert a predecessor piece first, satisfying the dependency constraint."
        elif fail_tag == "NEEDS_REORIENT":
            return (
                f"'{oracle_action}' would reorient the piece to stand it upright before insertion."
            )
        elif fail_tag == "HAND_FULL":
            return f"'{oracle_action}' would first put down or insert the currently held object."

        return f"'{oracle_action}' satisfies the required preconditions."

    def _extract_action_type(self, action: str) -> str:
        """Extract action type from action string."""
        action_lower = action.lower().strip()

        if action_lower.startswith("pick up"):
            return "pick"
        elif action_lower.startswith("insert"):
            return "insert"
        elif action_lower.startswith("reorient"):
            return "reorient"
        elif action_lower.startswith("put down"):
            return "putdown"
        elif action_lower == "done":
            return "done"

        return "unknown"

    def _extract_principle_from_text(self, text: str) -> str:
        """Extract principle from unstructured text."""
        # Look for common patterns
        patterns = [
            r"principle[:\s]+([^.]+\.)",
            r"rule[:\s]+([^.]+\.)",
            r"should\s+([^.]+\.)",
            r"must\s+([^.]+\.)",
            r"always\s+([^.]+\.)",
            r"never\s+([^.]+\.)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return first sentence
        sentences = text.split(".")
        if sentences:
            return sentences[0].strip() + "."

        return "Unable to extract principle."

    def batch_reflect(
        self,
        inputs: List[ReflectionInput],
    ) -> List[ReflectionOutput]:
        """
        Batch reflection for multiple failures.

        More efficient than calling reflect() multiple times.
        """
        outputs = []
        for input_data in inputs:
            output = self.reflect(input_data)
            outputs.append(output)
        return outputs

    def clear_cache(self) -> None:
        """Clear reflection cache."""
        self._reflection_cache.clear()


# ============================================================================
# Utility Functions
# ============================================================================


def create_reflection_input(
    failed_action: str,
    fail_tag: str,
    oracle_action: Optional[str],
    env,
    action_history: List[str],
    experience_id: str,
    image: Optional[np.ndarray] = None,
) -> ReflectionInput:
    """
    Create ReflectionInput from environment state.

    Helper function to extract symbolic state from environment.
    """
    from roboworld.agent.romemo_stack import extract_symbolic_state

    symbolic_state = extract_symbolic_state(env, failed_action)

    return ReflectionInput(
        failed_action=failed_action,
        fail_tag=fail_tag,
        oracle_action=oracle_action,
        symbolic_state=symbolic_state,
        action_history=action_history,
        experience_id=experience_id,
        image=image,
    )
