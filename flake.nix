{
  description = "rl-mistake-analysis development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          trainingPython = pkgs.python312.withPackages (ps: [
            ps.pip
            ps.scikit-learn
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              gitMinimal
              onnxruntime
              trainingPython
              ruff
              rustc
              rustfmt
              uv
            ];

            ORT_LIB_PATH = "${pkgs.lib.getLib pkgs.onnxruntime}/lib";
            ORT_PREFER_DYNAMIC_LINK = "1";
            UV_PYTHON = "${trainingPython}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
            UV_LINK_MODE = "copy";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.onnxruntime
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];
            shellHook = ''
              export PYTHONPATH="${trainingPython}/${pkgs.python312.sitePackages}"
            '';
          };
        });
    };
}
