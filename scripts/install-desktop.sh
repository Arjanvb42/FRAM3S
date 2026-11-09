#!/usr/bin/env bash
# Adds FRAM3 to the app menu (GNOME/KDE). No sudo needed.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
mkdir -p "$APPS" "$ICONS"
chmod +x "$DIR/scripts/fram3-launch.sh" "$DIR/fram3.sh"
cp "$DIR/web/icon.svg" "$ICONS/fram3.svg"
cat > "$APPS/fram3.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=FRAM3
GenericName=3D-scanstudio
Comment=Van video of foto's naar een 3D-model, bewerken en exporteren
Exec="$DIR/scripts/fram3-launch.sh"
Icon=fram3
Terminal=false
Categories=Graphics;3DGraphics;
Keywords=3D;scan;fotogrammetrie;STL;3MF;OBJ;printen;
StartupWMClass=FRAM3
DESKTOP
update-desktop-database "$APPS" >/dev/null 2>&1 || true
gtk-update-icon-cache -q "${ICONS%/scalable/apps}" >/dev/null 2>&1 || true
echo "FRAM3 staat nu in je appmenu."
