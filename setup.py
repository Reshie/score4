import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Allow installation without a compiler while keeping a strict opt-in."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            if os.environ.get("SCORE4_REQUIRE_CPP") == "1":
                raise
            self.warn(f"optional Score4 C++ self-play extension was not built: {exc}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            if os.environ.get("SCORE4_REQUIRE_CPP") == "1":
                raise
            self.warn(f"optional extension {ext.name} was not built: {exc}")


setup(
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=[
        Extension(
            "score4._self_play_cpp",
            ["src/score4/_self_play_cpp.cpp"],
            language="c++",
            extra_compile_args=["/O2", "/std:c++17"] if __import__("sys").platform == "win32" else ["-O3", "-std=c++17"],
        )
    ]
)
