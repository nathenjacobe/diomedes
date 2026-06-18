{
  description = "diomedes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        toolchain = fenix.packages.${system}.complete.withComponents [
          "rust-src"
          "rustc"
          "cargo"
        ];

        rust-analyzer = fenix.packages.${system}.rust-analyzer;
      in
      {

        packages.mesa = pkgs.mesa;
        devShells.default = pkgs.mkShell {
          packages = [
            toolchain
            rust-analyzer

            pkgs.pkg-config
            pkgs.cmake

            pkgs.vulkan-loader
            pkgs.vulkan-tools
            pkgs.vulkan-validation-layers
            pkgs.shaderc

            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.libX11
            pkgs.libXcursor
            pkgs.libXrandr
            pkgs.libXi
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.vulkan-loader}/lib:${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib:${pkgs.shaderc.lib}/lib"
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
          '';
        };
      });
}
