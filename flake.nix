{
  description = "Krystal WebGPU + UV (Python/CUDA) dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      # Biblioteki wymagane dla sterownika NVIDIA, Vulkana oraz binarek Pythona pobieranych przez UV (np. PyTorch/CUDA)
      gpuLibs = with pkgs; [
        vulkan-loader
        stdenv.cc.cc.lib  # Daje libstdc++.so.6 dla pre-built wheels z PyPI
        libxkbcommon
        wayland
        libX11
        libXcursor
        libXrandr
        libXi
        # Biblioteki systemowe wymagane przez pakiety PyPI (np. torch):
        zlib
        glib
        cudaPackages.cuda_cudart
        cudaPackages.cudatoolkit
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Runtimes / JS
          nodejs
          deno
          bun

          # C/C++ & GPU Tools
          gcc
          clang          # wymagany przez scriptc (kompilacja TS -> LLVM -> native)
          lld            # linkowanie generowanego kodu przez scriptc
          vulkan-loader
          vulkan-tools
          cudaPackages.cuda_nvcc
          cudaPackages.cudatoolkit

          unzip

          # UV & podstawowy Python systemowy jako baza
          uv
          python3
        ];

        shellHook = ''
          # 1. Podłączenie bibliotek Nix, sterowników NVIDIA z NixOS oraz CUDA do LD_LIBRARY_PATH dla UV
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath gpuLibs}:$LD_LIBRARY_PATH"

          # 2. Konfiguracja zmiennych środowiskowych CUDA dla kompilacji / instalacji
          export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
          export CUDA_ROOT="${pkgs.cudaPackages.cudatoolkit}"
          export EXTRA_LDFLAGS="-L/run/opengl-driver/lib -L${pkgs.cudaPackages.cuda_cudart}/lib"

          # 3. Wskazanie loaderowi Vulkan pliku ICD sterownika NVIDIA na NixOS
          if [ -f "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json" ]; then
            export VK_DRIVER_FILES="/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json"
            export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json"
          fi

          echo "================================================="
          echo " WebGPU + UV Python DevShell ready!"
          echo " UV version:    $(uv --version)"
          echo " System Python: $(python3 --version)"
          echo "================================================="
        '';
      };
    };
}