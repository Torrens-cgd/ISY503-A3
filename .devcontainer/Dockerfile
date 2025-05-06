FROM mcr.microsoft.com/devcontainers/base:noble


ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages
RUN apt-get update && apt-get install -y \
    git curl zsh unzip sudo stow fonts-powerline wget fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Install CaskaydiaCove Nerd Font
RUN mkdir -p /usr/share/fonts/truetype/caskaydia && \
    wget -q https://github.com/ryanoasis/nerd-fonts/releases/download/v3.3.0/CascadiaCode.zip -O /tmp/CaskaydiaCove.zip && \
    unzip /tmp/CaskaydiaCove.zip -d /usr/share/fonts/truetype/caskaydia/ && \
    rm /tmp/CaskaydiaCove.zip && \
    fc-cache -fv


# Clone dotfiles repository and apply configurations
# RUN git clone https://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@github.com/shinrinyoku-tech/dotfiles.git ~/.dotfiles && \
#     cd ~/.dotfiles

# # Run the setup script
# RUN BASH ~/.dotfiles/setup-dotfiles.sh