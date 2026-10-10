let
  flake = builtins.getFlake "git+file://${toString ./.}";
in
flake.devShells.${builtins.currentSystem}.default.overrideAttrs (old: {
  UV_CACHE_DIR = "${toString ./.}/.uv-cache";
  shellHook = old.shellHook + ''
    uv sync --locked --group notebooks --group training --group tracking --no-install-package scikit-learn --no-install-package pip || exit 1
    source .venv/bin/activate
  '';
})
