"""Base adapter types for realtime SLAM / occupancy engines."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type


@dataclass(frozen=True)
class EngineDescriptor:
    name: str
    api_url: str
    role: str
    provider: str = "selfsuvis"
    open_source: bool = True
    service_name: str = ""
    env_image_var: str = ""
    default_image: str = ""
    hardware_profile: str = ""
    required_modalities: Tuple[str, ...] = ()
    recommended_modalities: Tuple[str, ...] = ()
    pros: Tuple[str, ...] = ()
    cons: Tuple[str, ...] = ()
    integration_doc: str = ""
    notes: str = ""


class RealtimeEngineAdapter:
    descriptor: EngineDescriptor

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def api_url(self) -> str:
        return self.descriptor.api_url

    @property
    def configured(self) -> bool:
        return bool(self.api_url)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.descriptor.name,
            "role": self.descriptor.role,
            "provider": self.descriptor.provider,
            "open_source": self.descriptor.open_source,
            "service_name": self.descriptor.service_name,
            "api_url": self.descriptor.api_url,
            "env_image_var": self.descriptor.env_image_var,
            "default_image": self.descriptor.default_image,
            "hardware_profile": self.descriptor.hardware_profile,
            "required_modalities": list(self.descriptor.required_modalities),
            "recommended_modalities": list(self.descriptor.recommended_modalities),
            "pros": list(self.descriptor.pros),
            "cons": list(self.descriptor.cons),
            "integration_doc": self.descriptor.integration_doc,
            "notes": self.descriptor.notes,
        }


def build_descriptor(
    *,
    name: str,
    api_url: str,
    role: str,
    provider: str = "selfsuvis",
    open_source: bool = True,
    service_name: str = "",
    env_image_var: str = "",
    default_image: str = "",
    hardware_profile: str = "",
    required_modalities: Tuple[str, ...] = (),
    recommended_modalities: Tuple[str, ...] = (),
    pros: Tuple[str, ...] = (),
    cons: Tuple[str, ...] = (),
    integration_doc: str = "",
    notes: str = "",
) -> EngineDescriptor:
    return EngineDescriptor(
        name=name,
        api_url=api_url,
        role=role,
        provider=provider,
        open_source=open_source,
        service_name=service_name,
        env_image_var=env_image_var,
        default_image=default_image,
        hardware_profile=hardware_profile,
        required_modalities=required_modalities,
        recommended_modalities=recommended_modalities,
        pros=pros,
        cons=cons,
        integration_doc=integration_doc,
        notes=notes,
    )


def instantiate_adapter(
    registry: Dict[str, Type[RealtimeEngineAdapter]],
    name: str,
    *,
    default_name: str = "stub",
) -> RealtimeEngineAdapter:
    normalized = str(name or default_name).strip().lower()
    adapter_cls = registry.get(normalized) or registry[default_name]
    return adapter_cls()


def available_backend_urls(registry: Dict[str, Type[RealtimeEngineAdapter]]) -> Dict[str, str]:
    return {name: adapter_cls().api_url for name, adapter_cls in registry.items()}


def describe_backends(registry: Dict[str, Type[RealtimeEngineAdapter]]) -> Dict[str, Dict[str, object]]:
    return {name: adapter_cls().describe() for name, adapter_cls in registry.items()}
