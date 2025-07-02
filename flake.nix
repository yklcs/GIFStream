{
  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
    permittedInsecurePackages = [
      "freeimage-3.18.0-unstable-2024-04-18"
    ];
  };
  inputs.nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
  outputs = { nixpkgs, self }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          cudaSupport = true;
          allowUnfree = true;
          permittedInsecurePackages = [
            "freeimage-3.18.0-unstable-2024-04-18"
          ];
        };
      };
      packages = with pkgs; [
        libxcrypt
        ninja
        colmap
        ffmpeg
        libGL
        glib
        cmake
        gcc11
        python310
      ] ++ (
        with cudaPackages_12_8; [
          cuda_cccl
          cuda_cudart
          cuda_cupti
          cuda_nvcc
          cuda_nvml_dev
          cuda_nvrtc
          cuda_nvtx
          cusparselt
          libcublas
          libcufft
          libcufile
          libcurand
          libcusolver
          libcusparse
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        inherit packages;
        shellHook =
          let
            libPath =
              with pkgs; lib.makeLibraryPath (packages ++ [ addDriverRunpath.driverLink ]);
          in
          ''
            export LD_LIBRARY_PATH=${libPath}:$NIX_LD_LIBRARY_PATH
          '';
      };
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixpkgs-fmt;
    };
}
