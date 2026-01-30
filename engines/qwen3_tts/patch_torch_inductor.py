"""
TTS Audio Suite - PyTorch Inductor Memory Patch
Patches torch._inductor in MEMORY only (doesn't modify files)
Fixes "duplicate template name" bug in PyTorch 2.8.0 + triton-windows 3.2.0
"""

import sys

def patch_torch_inductor_memory():
    """
    Patches torch._inductor in memory to allow duplicate registrations.
    This doesn't modify any files - only the loaded modules in sys.modules.
    """
    try:
        # Check if already imported - if so, patch the already-loaded module
        already_imported = 'torch._inductor.select_algorithm' in sys.modules

        if already_imported:
            # Too late for pre-import patching, but we can still patch the loaded module
            print("ℹ️ torch._inductor already imported - patching existing module")
            import torch._inductor.select_algorithm as select_algo
        else:
            # Import it for the first time
            import torch._inductor.select_algorithm as select_algo

        # Patch 1: TritonTemplate - clear existing registrations and patch __init__
        if hasattr(select_algo.TritonTemplate, 'all_templates'):
            # Clear existing templates to allow re-registration
            select_algo.TritonTemplate.all_templates.clear()

        original_template_init = select_algo.TritonTemplate.__init__
        def patched_template_init(self, name, *args, **kwargs):
            if name in select_algo.TritonTemplate.all_templates:
                del select_algo.TritonTemplate.all_templates[name]
            return original_template_init(self, name, *args, **kwargs)
        select_algo.TritonTemplate.__init__ = patched_template_init

        # Patch 2: ExternKernelChoice - clear existing registrations and patch __init__
        if hasattr(select_algo.ExternKernelChoice, 'all_extern_kernel_choices'):
            # Clear existing extern kernels to allow re-registration
            select_algo.ExternKernelChoice.all_extern_kernel_choices.clear()

        original_extern_init = select_algo.ExternKernelChoice.__init__
        def patched_extern_init(self, kernel, *args, **kwargs):
            if hasattr(select_algo.ExternKernelChoice, 'all_extern_kernel_choices'):
                if kernel in select_algo.ExternKernelChoice.all_extern_kernel_choices:
                    del select_algo.ExternKernelChoice.all_extern_kernel_choices[kernel]
            return original_extern_init(self, kernel, *args, **kwargs)
        select_algo.ExternKernelChoice.__init__ = patched_extern_init

        # Patch 3: Artifact registration in remote cache
        try:
            # Import remote_cache if not already imported
            if 'torch._inductor.remote_cache' in sys.modules:
                import torch._inductor.remote_cache as remote_cache
            else:
                import torch._inductor.remote_cache as remote_cache

            if hasattr(remote_cache, 'RemoteCacheArtifactFactory'):
                factory = remote_cache.RemoteCacheArtifactFactory

                # Clear existing artifact registrations to allow re-registration
                if hasattr(factory, '_artifacts'):
                    factory._artifacts.clear()

                # Patch the register method to allow future duplicates
                original_register = factory.register.__func__ if hasattr(factory.register, '__func__') else factory.register

                def patched_register(cls, artifact_cls):
                    artifact_type = artifact_cls.__name__.lower()
                    if hasattr(cls, '_artifacts') and artifact_type in cls._artifacts:
                        del cls._artifacts[artifact_type]
                    return original_register(cls, artifact_cls)

                factory.register = classmethod(patched_register)
        except Exception as e:
            # Remote cache might not exist in this PyTorch version
            pass

        return True

    except Exception as e:
        print(f"⚠️ Memory patch failed: {e}")
        return False

if __name__ == '__main__':
    patch_torch_inductor_memory()
