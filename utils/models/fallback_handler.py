"""
Unified Fallback Handler for Model Loading

Provides a generic fallback chain mechanism that any TTS engine can use
to gracefully degrade when primary model loading fails.

Example: If French model not found, try English. If local not available, try HuggingFace.
"""

from typing import Callable, List, Tuple, Optional, Any
from utils.models.exceptions import TTSModelLoadingError


class FallbackHandler:
    """
    Generic fallback chain for model loading.

    Allows engines to define a sequence of fallback loaders to try.
    Each fallback has a condition that determines if it should be attempted.

    Example:
        handler = FallbackHandler(primary_loader=load_french_model)
        handler.add_fallback(load_english_model, condition=lambda e: "French" in str(e))
        handler.add_fallback(load_from_huggingface, condition=lambda e: "not found" in str(e))
        model = handler.load()
    """

    def __init__(self, primary_loader: Callable, name: str = "Model"):
        """
        Initialize fallback handler with primary loader.

        Args:
            primary_loader: Function to call first to load model
            name: Name of what's being loaded (for logging/errors)
        """
        self.primary_loader = primary_loader
        self.name = name
        self.fallbacks: List[Tuple[Callable, Optional[Callable]]] = []
        self.attempted_loaders: List[str] = []
        self.last_error: Optional[Exception] = None

    def add_fallback(
        self,
        fallback_loader: Callable,
        condition: Optional[Callable] = None,
        name: Optional[str] = None
    ) -> "FallbackHandler":
        """
        Add a fallback loader to the chain.

        Args:
            fallback_loader: Function to call if previous loading fails
            condition: Optional function that returns True if fallback should be attempted
                      Receives the exception from previous failure
            name: Optional name for this fallback (for logging)

        Returns:
            self for chaining
        """
        self.fallbacks.append((fallback_loader, condition))
        return self

    def load(self) -> Any:
        """
        Execute fallback chain.

        Tries primary loader first, then fallbacks in order until one succeeds.

        Returns:
            Loaded model from first successful loader

        Raises:
            TTSModelLoadingError: If all loaders fail
        """
        loaders_to_try = [(self.primary_loader, f"{self.name} (primary)", None)]
        loaders_to_try.extend([
            (loader, f"{self.name} (fallback {i+1})", condition)
            for i, (loader, condition) in enumerate(self.fallbacks)
        ])

        for loader, loader_name, condition in loaders_to_try:
            try:
                print(f"🔄 Attempting to load: {loader_name}")
                result = loader()
                if result is not None:
                    print(f"✅ Successfully loaded {loader_name}")
                    return result
            except Exception as e:
                self.last_error = e
                self.attempted_loaders.append(loader_name)

                # Check if this fallback's condition matches
                if condition is not None:
                    try:
                        if not condition(e):
                            # Condition doesn't match, skip this fallback
                            print(f"⏭️  Skipping {loader_name} (condition not met)")
                            continue
                    except Exception:
                        pass  # If condition check fails, continue anyway

                print(f"⚠️  Failed to load {loader_name}: {str(e)}")

        # All loaders failed
        error_msg = f"Failed to load {self.name} after trying {len(self.attempted_loaders)} sources:\n"
        error_msg += "\n".join(f"  - {loader}" for loader in self.attempted_loaders)
        if self.last_error:
            error_msg += f"\n\nLast error: {self.last_error}"

        raise TTSModelLoadingError(error_msg)

    def with_language_fallback(
        self,
        primary_language: str,
        fallback_languages: List[str],
        loader_func: Callable
    ) -> "FallbackHandler":
        """
        Convenience method for language-based fallbacks.

        Automatically adds fallback loaders for alternative languages.

        Args:
            primary_language: Primary language to try
            fallback_languages: List of languages to try in order
            loader_func: Function that takes language and returns loader
                        e.g., lambda lang: lambda: load_model(language=lang)

        Returns:
            self for chaining
        """
        for lang in fallback_languages:
            if lang != primary_language:
                self.add_fallback(
                    fallback_loader=loader_func(lang),
                    name=f"{lang} model"
                )
        return self

    def with_local_then_remote(
        self,
        local_loader: Callable,
        remote_loader: Callable
    ) -> "FallbackHandler":
        """
        Convenience method for local-first, remote-fallback loading.

        Args:
            local_loader: Try loading from local filesystem first
            remote_loader: Fall back to downloading from remote

        Returns:
            self for chaining
        """
        self.add_fallback(
            fallback_loader=remote_loader,
            condition=lambda e: "not found" in str(e) or "FileNotFoundError" in type(e).__name__,
            name="Remote (HuggingFace)"
        )
        return self


def create_language_fallback_chain(
    primary_language: str,
    fallback_languages: List[str],
    load_model_func: Callable,
    name: str = "Model"
) -> FallbackHandler:
    """
    Create a pre-configured fallback handler for language switching.

    Args:
        primary_language: Primary language to load
        fallback_languages: Languages to try if primary fails
        load_model_func: Function that takes language parameter and loads model
        name: Display name for logging

    Returns:
        Configured FallbackHandler ready to use
    """
    handler = FallbackHandler(
        primary_loader=lambda: load_model_func(primary_language),
        name=name
    )

    for lang in fallback_languages:
        if lang != primary_language:
            handler.add_fallback(
                fallback_loader=lambda l=lang: load_model_func(l),
                name=f"{lang} model"
            )

    return handler
