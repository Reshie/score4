import os
import sys

from setuptools import Extension, find_packages, setup
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
    name="score4",
    version="0.1.0",
    description="AlphaZero-style reinforcement learning for 4x4x4 Score Four.",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=[
        Extension(
            "score4._self_play_cpp",
            ["src/score4/_self_play_cpp.cpp"],
            language="c++",
            extra_compile_args=(
                ["/O2", "/std:c++17", "/openmp"]
                if sys.platform == "win32"
                else ["-O3", "-std=c++17", "-fopenmp"]
                if sys.platform.startswith("linux")
                else ["-O3", "-std=c++17"]
            ),
            extra_link_args=(
                ["-fopenmp"] if sys.platform.startswith("linux") else []
            ),
        )
    ]
)
