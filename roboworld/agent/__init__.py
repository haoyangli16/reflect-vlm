"""
RoboWorld Agent Module

Contains various agent implementations for the assembly task:
- LlavaAgent: VLM-based agent using LLaVA
- GPTAgent: OpenAI GPT-4o agent
- GeminiAgent: Google Gemini agent
- OracleAgent: Perfect agent with ground truth
- RoMemoDiscreteAgent: Memory-augmented agent

New components:
- UnifiedVLM: Unified interface for VLM APIs
- Reflector: Principle extraction from failures

Note: Legacy agents (LlavaAgent, GPTAgent, etc.) are imported lazily
to avoid import conflicts with transformers.
"""

# Legacy agents are imported lazily to avoid conflicts
# Use: from roboworld.agent.llava import LlavaAgent
# Use: from roboworld.agent.gpt import GPTAgent
# Use: from roboworld.agent.gemini import GeminiAgent
# Use: from roboworld.agent.oracle import AssemblyOracle

LlavaAgent = None  # Lazy import
GPTAgent = None  # Lazy import
GeminiAgent = None  # Lazy import
AssemblyOracle = None  # Lazy import

# New unified VLM interface
try:
    from roboworld.agent.vlm_api import (
        UnifiedVLM,
        OpenAIVLM,
        GeminiVLM,
        QwenVLM,
        create_vlm,
        get_available_models,
    )
except ImportError:
    UnifiedVLM = None
    OpenAIVLM = None
    GeminiVLM = None
    QwenVLM = None
    create_vlm = None
    get_available_models = None

# Reflector for principle extraction
try:
    from roboworld.agent.reflector import (
        Reflector,
        ReflectionInput,
        ReflectionOutput,
        FAIL_TAG_DESCRIPTIONS,
    )
except ImportError:
    Reflector = None
    ReflectionInput = None
    ReflectionOutput = None
    FAIL_TAG_DESCRIPTIONS = None

# RoMemo stack
try:
    from roboworld.agent.romemo_stack import (
        RoMemoDiscreteAgent,
        RoMemoDiscreteConfig,
        extract_symbolic_state,
    )
except ImportError:
    RoMemoDiscreteAgent = None
    RoMemoDiscreteConfig = None
    extract_symbolic_state = None

__all__ = [
    # Legacy agents
    "LlavaAgent",
    "GPTAgent",
    "GeminiAgent",
    "AssemblyOracle",
    # New VLM interface
    "UnifiedVLM",
    "OpenAIVLM",
    "GeminiVLM",
    "QwenVLM",
    "create_vlm",
    "get_available_models",
    # Reflector
    "Reflector",
    "ReflectionInput",
    "ReflectionOutput",
    "FAIL_TAG_DESCRIPTIONS",
    # RoMemo
    "RoMemoDiscreteAgent",
    "RoMemoDiscreteConfig",
    "extract_symbolic_state",
]
