# The app and the eval commands in the locked `cap6938` environment (pixi.lock), on any x86-64 machine with
# Docker or Podman. No GPU needed: renders use Mesa's software OpenGL, and local models run in Ollama outside
# the container. See docs/docker.md.
#
#   docker build -t stl-agent --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) .
#   docker run --rm -p 7860:7860 --env-file .env stl-agent
FROM ghcr.io/prefix-dev/pixi:0.66.0-noble

# OpenGL for CadQuery (OCP) and VTK's offscreen renders: Mesa's software EGL/OSMesa, no display or GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libgl1-mesa-dri libosmesa6 libglu1-mesa \
        libx11-6 libxext6 libxrender1 libxt6 libsm6 libxi6 libxcursor1 libxrandr2 libxinerama1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change doesn't reinstall them. --locked: exactly the versions in pixi.lock.
COPY pixi.toml pixi.lock ./
RUN pixi install --locked -e cap6938 && rm -rf /root/.cache/rattler

COPY . .
RUN mkdir -p /app/data /app/outputs

# Benchmark rows record the git commit; .git is not copied, so it is passed in at build time.
ARG GIT_COMMIT=unknown
ENV STL_AGENT_COMMIT=${GIT_COMMIT} \
    GRADIO_SERVER_NAME=0.0.0.0 \
    CAD_SETTINGS=/app/data/settings.toml

EXPOSE 7860
ENTRYPOINT ["pixi", "run", "--locked", "-e", "cap6938"]
CMD ["app"]
