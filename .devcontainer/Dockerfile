FROM mcr.microsoft.com/devcontainers/base:noble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Conda-forge Miniforge)
ENV CONDA_DIR=/opt/conda
RUN curl -L -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN conda config --system --set auto_activate_base false \
    && conda clean -afy

# Install PyInstaller for backend packaging (used by the app build scripts).
# Use conda-forge to ensure a compatible binary.
RUN conda install -y -c conda-forge pyinstaller \
    && conda clean -afy

# Install Node.js LTS and Tauri CLI for building the desktop frontend
# Use NodeSource to get a recent LTS and then install the Tauri CLI globally.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get update && apt-get install -y --no-install-recommends \
       nodejs \
       build-essential \
       pkg-config \
    libwebkit2gtk-4.1-dev \
       libgtk-3-dev \
       python3 \
    && NPM_CONFIG_UNSAFE_PERM=true npm install -g @tauri-apps/cli \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain into the non-root `vscode` user's home (so `cargo` is runnable)
# Run rustup install as the `vscode` user and ensure ownership of the toolchain directories.
RUN su -s /bin/bash vscode -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y" \
    && su -s /bin/bash vscode -c "source /home/vscode/.cargo/env && rustup default stable && rustup component add rustfmt clippy" \
    && chown -R 1000:1000 /home/vscode/.cargo /home/vscode/.rustup || true

# Ensure cargo bin on PATH for later steps (vscode user's cargo first)
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# Ensure non-root developer user can use conda (devcontainer default user is 'vscode')
RUN chown -R $(id -u 1000 2>/dev/null || echo 1000):$(id -g 1000 2>/dev/null || echo 1000) ${CONDA_DIR} || true

CMD ["/bin/bash"]
