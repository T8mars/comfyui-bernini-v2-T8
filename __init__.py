"""ComfyUI Bernini v2 custom-node entry point."""


async def comfy_entrypoint():
    # Lazy import keeps conversion/unit tests independent from a ComfyUI
    # installation and also works when pytest collects this root file as a
    # top-level ``__init__`` module from a hyphenated custom-node directory.
    from comfy_api.latest import ComfyExtension
    from typing_extensions import override

    from .nodes.loaders import BerniniV2PlannerLoader, BerniniV2T5Loader, BerniniV2WanLoader
    from .nodes.planning import BerniniV2PlanNode
    from .nodes.rendering import BerniniV2RendererGuider, BerniniV2Scheduler, BerniniV2UniPCSampler

    class BerniniV2Extension(ComfyExtension):
        @override
        async def get_node_list(self):
            return [
                BerniniV2WanLoader,
                BerniniV2PlannerLoader,
                BerniniV2T5Loader,
                BerniniV2PlanNode,
                BerniniV2RendererGuider,
                BerniniV2Scheduler,
                BerniniV2UniPCSampler,
            ]

    return BerniniV2Extension()


__all__ = ["comfy_entrypoint"]
