let
  flake = builtins.getFlake "git+file://${toString ./.}";
in
flake.devShells.${builtins.currentSystem}.default.overrideAttrs (_: {
  shellHook = ''
    uv sync --locked --group notebooks || exit 1
    source .venv/bin/activate
  '';
})
