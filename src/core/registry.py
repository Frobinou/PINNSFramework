from typing import TypeVar, Type, Callable

T = TypeVar("T")


class Registry:
    """
    Generic registry mapping string keys to classes.
    Supports decorator-based registration and metadata.
    """

    def __init__(self, name: str):
        self._name = name
        self._items: dict[str, type] = {}
        self._metadata: dict[str, dict] = {}

    def register(self, name: str, **meta):
        """
        Decorator to register a class under a given name.

        Usage:
            @REGISTRY.models.register("mlp", description="Simple MLP")
            class MLPModel: ...
        """
        def wrapper(cls: Type[T]) -> Type[T]:
            if name in self._items:
                raise ValueError(
                    f"[{self._name}] '{name}' is already registered. "
                    f"Existing: {self._items[name].__name__}, "
                    f"New: {cls.__name__}"
                )
            self._items[name] = cls
            self._metadata[name] = {"class": cls.__name__, **meta}
            return cls
        return wrapper

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"[{self._name}] Unknown key: '{name}'. "
                f"Available: {self.list()}"
            )
        return self._items[name]

    def build(self, name: str, **kwargs):
        """Instantiate a registered class directly."""
        return self.get(name)(**kwargs)

    def list(self) -> list[str]:
        return list(self._items.keys())

    def info(self) -> dict:
        """Return metadata for all registered items."""
        return dict(self._metadata)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={self.list()})"


class GlobalRegistry:
    def __init__(self):
        self.odes = Registry("odes")
        self.models = Registry("models")
        self.losses = Registry("losses")
        self.data_loaders = Registry("data_loaders")
        self.callbacks = Registry("callbacks")
        self.evaluators = Registry("evaluators")

    def __repr__(self) -> str:
        sections = "\n".join(
            f"  {r}: {getattr(self, r).list()}"
            for r in ("odes", "models", "losses", "data_loaders", "callbacks", "evaluators")
        )
        return f"GlobalRegistry:\n{sections}"


REGISTRY = GlobalRegistry()
